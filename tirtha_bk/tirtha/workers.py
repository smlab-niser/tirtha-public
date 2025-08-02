import json
import os
import shutil
import pytz
from datetime import datetime
from pathlib import Path
from subprocess import STDOUT, CalledProcessError, check_output
from rich.console import Console
from typing import Optional
from django.conf import settings
from django.db import IntegrityError
from django.utils import timezone
from django.core.mail import EmailMultiAlternatives
from django.template.loader import render_to_string

# Local imports
from tirtha.models import ARK, Contribution, Image, Mesh, Run

from .alicevision import AliceVision
from .postprocess import PostProcess
from .utils import Logger
from .utilsark import generate_noid, noid_check_digit


STATIC = Path(settings.STATIC_ROOT)
MEDIA = Path(settings.MEDIA_ROOT)
LOG_DIR = Path(settings.LOG_DIR)
ARCHIVE_ROOT = Path(settings.ARCHIVE_ROOT)
GS_MAX_ITER = settings.GS_MAX_ITER
MESHOPS_MIN_IMAGES = settings.MESHOPS_MIN_IMAGES
ALICEVISION_DIRPATH = settings.ALICEVISION_DIRPATH
NSFW_MODEL_DIRPATH = settings.NSFW_MODEL_DIRPATH
# VGGT_SCRIPT_PATH = settings.VGGT_SCRIPT_PATH
# VGGT_ENV_PATH = settings.VGGT_ENV_PATH
MANIQA_MODEL_FILEPATH = settings.MANIQA_MODEL_FILEPATH
OBJ2GLTF_PATH = settings.OBJ2GLTF_PATH
GLTFPACK_PATH = settings.GLTFPACK_PATH
COLMAP_PATH = settings.COLMAP_PATH
BASE_URL = settings.BASE_URL
ARK_NAAN = settings.ARK_NAAN
ARK_SHOULDER = settings.ARK_SHOULDER


class BaseOps:
    def __init__(self, meshID: str, kind: str = "aV") -> None:
        """
        Base class for all operations

        Parameters
        ----------
        meshID : str
            Mesh ID
        kind : str
            Kind of operation, one of ['aV', 'GS'], by default 'aV'

        """
        self.meshID = meshID
        self.mesh = mesh = Mesh.objects.get(ID=meshID)
        self.meshVID = mesh.verbose_id
        self.meshStr = f"{self.meshVID} <=> {self.meshID}"  # Used in logging

        # Create new Run
        self.kind = kind
        self.run = run = Run.objects.create(mesh=mesh, kind=kind)
        run.save()  # Creates run directory
        self.runID = runID = run.ID
        self.runDir = STATIC / "models" / Path(run.directory)

        # Set up Logger
        self.log_path = LOG_DIR / f"{kind}Ops/{meshID}" / self.runDir.stem
        if not self.log_path.exists():
            self.log_path.mkdir(parents=True, exist_ok=True)

        self.cls = cls = self.__class__
        # NOTE: Attributes with self.__class__ will only work if the class is instantiated.
        cls.logger = Logger(log_path=self.log_path, name=f"{cls.__name__}_{runID[:8]}")
        cls.logger.info(
            f"ID {meshID} has Verbose ID (VID) {self.meshVID}. Using VID for logging."
        )

        # Source (images) & run directories
        self.imageDir = MEDIA / f"models/{meshID}/images/good"

        cls.logger.info(f"Created new {kind} Run {runID} for mesh {self.meshStr}.")
        cls.logger.info(f"{kind} Run directory: {self.runDir}")
        cls.logger.info(f"{kind} Run log file: {cls.logger._log_file}")
        self._update_mesh_status("Processing")

        # Use image filenames (UUIDs in DB) to fetch images & corresponding contributors
        self.imageFiles = sorted(self.imageDir.glob("*"))
        self.imageUUIDs = [imageFile.stem for imageFile in self.imageFiles]
        try:
            # NOTE: This does not raise an error if the image is not found in the DB
            # CHECK: Test ain_bulk()
            self.images = Image.objects.in_bulk(
                self.imageUUIDs, field_name="ID"
            ).values()
            self.contributions = [image.contribution for image in self.images]
            self.contributors = [
                contribution.contributor for contribution in self.contributions
            ]

            # Set the `images` & `contributors` attributes of the Run
            # CHECK: Test `aset()`
            run.images.set(self.images)
            run.contributors.set(self.contributors)
            run.save()
            cls.logger.info(
                f"Set the `images` & `contributors` attributes of {kind} Run {runID}."
            )
        except Exception as e:
            self._handle_error(excep=e, caller="Fetching images & contributors")

        # Specify the run order (else alphabetical order)
        self._run_order_suffix = [
            "run_cleanup",
            "run_ark",
            "run_finalize",
        ]

    def _run_all(self) -> None:
        """
        Runs all the steps with default parameters.

        Raises
        ------
        ValueError
            If `_run_order` is not defined
        Exception
            If any exception is raised during the execution

        """
        try:
            if self._run_order:
                for step in self._run_order:
                    getattr(self, step)()
            else:
                raise ValueError("_run_order is not defined.")
        except Exception as e:
            self._handle_error(excep=e, caller="_run_all")

    def _check_output(self, out: Path, src: str) -> None:
        if not out.is_file():
            self._handle_error(
                FileNotFoundError(f"No output was produced by {src} at {out}."),
                src,
            )

    def _handle_error(self, excep: Exception, caller: str) -> None:
        """
        Handles errors during the worker execution

        Parameters
        ----------
        excep : Exception
            The exception that was raised
        caller : str
            The name of the function that raised the exception

        Raises
        ------
        excep : Exception
            The exception that was raised

        """
        # Log the error
        self.logger.error(f"{caller} step failed.")
        self.logger.error(f"{excep}", exc_info=True)

        # Update statuses to 'Error'
        self._update_mesh_status("Error")
        self.run.ended_at = timezone.now()
        self.run.save()
        self._update_run_status("Error")

        raise excep

    def _update_mesh_status(self, status: str) -> None:
        """
        Updates the mesh status in the DB

        Parameters
        ----------
        status : str
            The status to update the mesh to

        """
        self.mesh.status = status
        self.mesh.save()  # NOTE: Consider the effect on signals.py when saving Mesh or any other model
        self.logger.info(
            f"Updated mesh.status to '{status}' for mesh {self.meshStr}..."
        )

    def _update_run_status(self, status: str) -> None:
        """
        Updates the run status in the DB

        Parameters
        ----------
        status : str
            The status to update the run to

        """
        self.run.status = status
        self.run.save()
        self.logger.info(
            f"Updated {self.kind} run.status to '{status}' for run {self.runID}..."
        )

    @classmethod  # NOTE: Won't be picklable as a regular method
    def _serialRunner(cls, cmd: str, log_file: Path):
        """
        Run a command serially and log the output

        Parameters
        ----------
        cmd : str
            Command to run
        log_file : Path
            Path to the log file

        """
        logger = Logger(log_file.stem, log_file.parent)
        log_path = Path(log_file).resolve()
        try:
            cls.logger.info(f"Starting command execution. Log file: {log_path}.")
            logger.info(f"Command:\n{cmd}")
            output = check_output(cmd, shell=True, stderr=STDOUT)
            logger.info(f"Output:\n{output.decode().strip()}")
            cls.logger.info(f"Finished command execution. Log file: {log_path}.")
        except CalledProcessError as error:
            logger.error(f"\n{error.output.decode().strip()}")
            cls.logger.error(
                f"Error in command execution for {logger.name}. Check log file: {log_path}."
            )
            # NOTE: Won't appear in VSCode's Jupyter Notebook (May 2024)
            error.add_note(f"{logger.name} failed. Check log file: {log_path}.")
            raise error

    def run_cleanup(self) -> None:
        """
        Does the following:
        1. Cleans up older errored-out runs.
        2. Copies current run's output to "published".
        2. Archives current run.
        NOTE: Errored-out runs are not deleted during their run, but only during the next run.
        This is done to allow for debugging.

        """
        kind = self.kind
        out_map = {"aV": ".glb", "GS": ".splat", "Point": ".ply"}
        out_type = out_map[kind]
        # out_type = ".glb" if kind == "aV" else ".splat"  # Filetype of final output
        meshStr = self.meshStr
        curr_runID = self.runID
        arcDir = ARCHIVE_ROOT / self.meshID / f"{kind.lower()}cache" / self.runDir.stem

        self.logger.info(f"Cleaning up runs for mesh {meshStr}.")
        try:
            # 1. Delete errored-out runs (all kinds)
            self.logger.info(
                f"Looking up & deleting errored-out runs for mesh {meshStr}..."
            )
            runs = Run.objects.filter(mesh=self.mesh, status="Error").order_by(
                "-ended_at"
            )
            if len(runs) > 0:
                for run in runs:
                    runDir = STATIC / "models" / Path(run.directory)
                    self.logger.info(
                        f"{run.kind} Run {run.ID} for mesh {meshStr} has errors. Deleting..."
                    )
                    shutil.rmtree(runDir)  # Delete folder
                    run.delete()  # Delete from DB
                    self.logger.info(
                        f"Deleted {run.kind} run {run.ID} for mesh {meshStr}."
                    )
            self.logger.info(
                f"Deleted {len(runs)} errored-out runs for mesh {meshStr}."
            )

            # 2. Copy current run's output to STATIC / "models" / meshID / "published" / output_name
            self.logger.info(
                f"Copying output for {kind} run {curr_runID} for mesh {meshStr}..."
            )

            out_file_mapper = {
                "aV": "decimatedOptGLB.glb",
                "GS": "postprocessed.splat",
                "Point": "Point_Voxel.ply",
            }
            out_file = out_file_mapper[kind]
            src = self.opt_path / out_file
            self.arkURL = (
                f"models/{self.meshID}/published/{self.meshID}_{curr_runID}{out_type}"
            )
            dest = STATIC / self.arkURL
            shutil.copy2(src, dest)
            self.logger.info(
                f"Copied output for {kind} run {curr_runID} for mesh {meshStr}."
            )
            # 3. Move everything else to arcDir
            self.logger.info(
                f"Archiving {kind} run {curr_runID} for mesh {meshStr} to {arcDir}."
            )
            shutil.move(self.runDir, arcDir)  # Move run folder to archive
            self.run.directory = str(arcDir)  # Update run directory
            self._update_run_status("Archived")  # Update run status & save
            self.logger.info(
                f"Archived {kind} run {curr_runID} for mesh {meshStr} to {arcDir}."
            )

        except Exception as e:
            self.logger.error(f"Error cleaning up runs for mesh {meshStr}.")
            self._handle_error(e, "run_cleanup")
        self.logger.info(f"Finished cleaning up runs for mesh {meshStr}.")

    def run_ark(self, ark_len: int = 16) -> None:
        """
        Runs the ark generator to generate a unique ark for the run.

        Parameters
        ----------
        ark_len : int
            Length of the ark to be generated
            Default: 16

        """
        kind = self.kind
        arkURL, run, mesh, meshStr = self.arkURL, self.run, self.mesh, self.meshStr
        naan, shoulder = ARK_NAAN, ARK_SHOULDER
        run.ended_at = timezone.now()

        # Create metadata
        self.logger.info(
            f"Creating metadata for ARK for {kind} run {run.ID} for mesh {meshStr}..."
        )
        def_commit = (
            "This ARK was generated & is managed by Project Tirtha (https://smlab.niser.ac.in/project/tirtha/). "
            + "We are committed to maintaining this ARK as per our Terms of Use (https://smlab.niser.ac.in/project/tirtha/#terms) "
            + "and Privacy Policy (https://smlab.niser.ac.in/project/tirtha/#privacy)."
        )

        metadata = {
            "monument": {
                "name": str(mesh.name),
                "location": f"{mesh.district}, {mesh.state}, {mesh.country}",  # LATE_EXP: f"{mesh.lat}, {mesh.lon}"
                "verbose_id": str(mesh.verbose_id),
                "thumbnail": f"{BASE_URL}{mesh.thumbnail.url}",
                "description": str(mesh.description),
                "completed": True if mesh.completed else False,
            },
            "run": {
                "ID": str(run.ID),
                "ended_at": str(run.ended_at),
                "contributors": list(run.contributors.values_list("name", flat=True)),
                "images": int(run.images.count()),
            },
            "notice": def_commit,
        }
        metadata_json = json.dumps(metadata)
        self.logger.info(
            f"Created metadata for ARK for {kind} run {run.ID} for mesh {meshStr}..."
        )

        # Generate ark - NOTE: Adapted from arklet
        self.logger.info(
            f"Generating ARK for {kind} run {run.ID} for mesh {meshStr}..."
        )
        ark, collisions = None, 0
        while True:
            noid = generate_noid(ark_len)
            base_ark_string = f"{naan}{shoulder}{noid}"
            check_digit = noid_check_digit(base_ark_string)
            ark_string = f"{base_ark_string}{check_digit}"
            try:
                ark = ARK.objects.create(
                    ark=ark_string,
                    naan=naan,
                    shoulder=shoulder,
                    assigned_name=f"{noid}{check_digit}",
                    url=f"{BASE_URL}/static/{arkURL}",
                    metadata=metadata_json,
                )
                self.logger.info(f"ARK URL: {BASE_URL}/static/{arkURL}")
                break
            except IntegrityError:
                collisions += 1
                continue
        msg = f"Generated ARK for {kind} run {run.ID} for mesh {meshStr} after {collisions} collision(s)."
        if collisions == 0:
            self.logger.info(msg)
        else:
            self.logger.warning(msg)

        # Update run
        run.ark = ark
        run.save()
        self.arkStr = str(ark_string)

    def run_finalize(self) -> None:
        """
        Finalizes current run

        """
        kind = self.kind
        self.logger.info(f"Finalizing {kind} run {self.runID} for mesh {self.meshStr}.")
        self.mesh.reconstructed_at = datetime.now(pytz.timezone("Asia/Kolkata"))
        self.logger.info(f"{kind} Run {self.runID} finished for mesh {self.meshStr}.")
        self._update_mesh_status("Live")
        self.logger.info(
            f"Finished finalizing {kind} run {self.runID} for mesh {self.meshStr}."
        )


class MeshOps(BaseOps):
    """
    Mesh processing pipeline. Does the following:
    - Runs the aliceVision pipeline on a given `models.Mesh`
    - Runs `obj2gltf` to convert the obj file to gltf
    - Runs `gltfpack` (meshopt) to optimize the gltf file
    - Publishes the final output to the Tirtha site

    """

    def __init__(self, meshID: str) -> None:
        super().__init__(meshID=meshID, kind="aV")

        # Check if executables exist
        self.aV_exec = Path(ALICEVISION_DIRPATH)
        self.obj2gltf_exec = Path(OBJ2GLTF_PATH)
        self.gltfpack_exec = Path(GLTFPACK_PATH)
        self._check_exec(path=self.aV_exec)
        self._check_exec(exe=self.obj2gltf_exec)
        self._check_exec(exe=self.gltfpack_exec)

        # Logger setup for `AliceVision`
        MeshOps.av_logger = Logger(
            log_path=self.runDir,
            name=f"alicevision__{self.runID}",
        )

        # Specify the run order (else alphabetical order)
        self._run_order = [
            "run_aliceVision",
            "run_obj2gltf",
            "run_meshopt",
        ] + self._run_order_suffix

    def _check_exec(self, exe: str = None, path: Path = None) -> None:
        """
        Checks if the executables exist and are valid

        Parameters
        ----------
        exe : str, optional
            Name of the executable, by default None
        path : Path, optional
            Path to the executable, by default None

        Raises
        ------
        FileNotFoundError
            If the path to any of the executables is not found

        """
        if path is not None and not path.exists():
            self.logger.error(f"aliceVision executable not found: {path}")
            raise FileNotFoundError(f"aliceVision executable not found: {path}")

        if exe is not None:
            exe = shutil.which(exe)
            if not exe:
                self.logger.error(f"Executable not found: {exe}")
                raise FileNotFoundError(f"Executable not found: {exe}")

    def run_aliceVision(self) -> None:
        """
        Runs the aliceVision pipeline

        """
        mesh = self.mesh
        meshStr = self.meshStr
        self.logger.info(f"Creating aliceVision worker for mesh {meshStr}...")
        try:
            aV = AliceVision(
                exec_path=self.aV_exec,
                input_dir=self.imageDir,
                cache_dir=self.runDir,
                # If a prior run had errored out, set the current run to produce full tracebacks
                verboseLevel="trace" if mesh.status == "Error" else "info",
                logger=MeshOps.av_logger,
            )
        except Exception:
            self._handle_error(
                Exception(f"aliceVision worker failed for mesh {meshStr}."),
                "aliceVision",
            )

        try:
            self.logger.info(
                f"Running the aliceVision pipeline(s) for mesh {meshStr}..."
            )
            self.logger.info(f"Check log file: {aV.logger._log_file}")
            rotation = [mesh.rotaX, mesh.rotaY, mesh.rotaZ]
            orientMesh = mesh.orientMesh

            # Check if center_image exists
            center_image = mesh.center_image
            matches = list(self.imageDir.glob(f"{center_image}.*"))
            if not matches:
                center_image = None
                mesh.center_image = ""
                mesh.save()

            aV._run_all(
                center_image=center_image,
                denoise=mesh.denoise,  # NOTE: Denoising smooths out too much at the moment
                rotation=rotation,
                orientMesh=orientMesh,
                estimateSpaceMinObservationAngle=mesh.minObsAng,
            )

            # Set for further processing
            self.textured_path = self.runDir / "15_texturing/"

            self.logger.info(
                f"Finished running aliceVision pipeline for mesh {meshStr}."
            )

        except Exception:
            self._handle_error(
                Exception(f"aliceVision pipeline failed for mesh {meshStr}."),
                "aliceVision",
            )

    def run_obj2gltf(self, options: Optional[dict] = {"secure": "true"}) -> None:
        """
        Runs obj2gltf to create .glb in `static_dir` from .obj in `textured_path`
        For decimated mesh only

        Parameters
        ----------
        options : Optional[dict]
            Additional key-val options to pass to the obj2gltf executable
            Check obj2gltf docs for more info.
            Default: {"secure": "true"}

        """
        out_path = self.runDir / "obj2gltf/"
        out_path.mkdir(parents=True, exist_ok=True)
        cmd_init = f"{self.obj2gltf_exec} -i "

        if options:
            keyvals = [f"--{k} {v}" for k, v in options.items()]
            opts = " ".join(keyvals)

        # Decimated mesh only
        inp = self.textured_path / "texturedDecimatedMesh/texturedMesh.obj"
        out = out_path / "decimatedGLB.glb"
        cmd = cmd_init + f"{inp} -o {out} {opts}"
        log_path = out_path / "obj2gltf.log"
        self.logger.info(f"Running obj2gltf for mesh {self.meshStr}.")
        self.logger.info(f"Command: {cmd}")
        self._serialRunner(cmd, log_path)

        self._check_output(out, "obj2gltf")

        # Set for further processing
        self.glb_path = out_path

        self.logger.info(f"Finished running obj2gltf for mesh {self.meshStr}.")

    def run_meshopt(
        self,
        options: Optional[dict] = {
            "cc": "",  # No value needed
            "tc": "",  # No value needed
            "si": 0.6,  # Simplification factor
        },
    ) -> None:
        """
        Runs gltfpack's meshopt to compress and optimize glTF files
        For decimated mesh only

        Parameters
        ----------
        options : Optional[dict]
            Additional key-val options to pass to the gltfpack executable
            Check `gltfpack` docs for more info.
            Default: {'cc': '', 'tc': '', 'si': 0.6}

        """
        out_path = self.runDir / "meshopt/"
        out_path.mkdir(parents=True, exist_ok=True)
        cmd_init = f"{self.gltfpack_exec} -i "

        if "si" in options.keys():
            if not (0 < options["si"] <= 1):
                self._handle_error(ValueError("'si' must be > 0 and <= 1"), "meshopt")

        if options:
            keyvals = [f"-{k} {v}" for k, v in options.items()]
            opts = " ".join(keyvals)

        # Decimated mesh only
        inp = self.glb_path / "decimatedGLB.glb"
        out = out_path / "decimatedOptGLB.glb"
        cmd = cmd_init + f"{inp} -o {out} {opts}"
        log_path = out_path / "meshopt.log"
        self.logger.info(f"Running meshopt (gltfpack) for mesh {self.meshStr}.")
        self.logger.info(f"Commands: {cmd}")
        self._serialRunner(cmd, log_path)

        self._check_output(out, "meshopt")

        # Set for further processing
        self.opt_path = out_path

        self.logger.info(
            f"Finished running meshopt (gltfpack) for mesh {self.meshStr}."
        )


class GSOps(BaseOps):
    """
    Runs the Gaussian splatting pipeline

    """

    def __init__(self, meshID: str) -> None:
        super().__init__(meshID=meshID, kind="GS")

        # Check if nerfstudio is installed
        from importlib.util import find_spec

        if find_spec("nerfstudio"):
            self.logger.info("nerf_studio is installed.")
        else:
            self.logger.error("nerf_studio is not installed.")
            self._handle_error(ImportError("nerf_studio is not installed."), "GSOps")

        # Specify the run order (else alphabetical order)
        self._run_order = [
            "run_splatfacto",
            "run_postprocess",
        ] + self._run_order_suffix

    def run_splatfacto(self) -> None:
        """
        Creates Gaussian Splats using the `splatfacto` library

        """
        alpha_cull_thresh = settings.ALPHA_CULL_THRESH
        cull_post_dens = settings.CULL_POST_DENS

        log_path = self.log_path / "splatfacto.log"
        sf_train_log_path = self.log_path / "splatfacto_train.log"
        output_path = self.runDir / "output/"

        # Process data
        self.logger.info("Processing data for Splatfacto...")
        cmd = (
            "ns-process-data images --data "
            + str(self.imageDir)
            + " --output-dir "
            + str(output_path)
            + " --colmap-cmd "
            + str(COLMAP_PATH)
        )
        self._serialRunner(cmd, log_path)
        self.logger.info("Processed data for Splatfacto.")

        # Create GS
        self.logger.info("Creating GS using Splatfacto...")
        # NOTE: See https://docs.nerf.studio/nerfology/methods/splat.html#quality-and-regularization
        reg_opts = (
            # Threshold to delete translucent gaussians - lower values remove more (usually better quality)
            " --pipeline.model.cull_alpha_thresh="
            + str(alpha_cull_thresh)
            # Disable culling after 15K steps
            + " --pipeline.model.continue_cull_post_densification="
            + str(cull_post_dens)
            # Less spiky Gaussians
            + " --pipeline.model.use_scale_regularization True"
        )
        cmd = (
            "ns-train splatfacto "
            + reg_opts
            + " --data "
            + str(output_path)
            + " --output-dir "
            + str(output_path)
            + ' --timestamp ""'
            + f" --max-num-iterations {GS_MAX_ITER} "
            # Quit after GS creation
            # Also see: https://docs.nerf.studio/quickstart/viewer_quickstart.html#accessing-over-an-ssh-connection
            + "--viewer.quit-on-train-completion True "
            + "> "
            + str(sf_train_log_path)
        )
        self.logger.info(f"Check log file: {sf_train_log_path}.")
        self._serialRunner(cmd, log_path)
        self.logger.info("Created GS using Splatfacto.")

        # Export GS
        self.logger.info("Exporting GS from Splatfacto...")
        cmd = (
            "ns-export gaussian-splat --load-config "
            + str(output_path)
            + "/output/splatfacto/config.yml "
            + " --output-dir "
            + str(output_path)
        )
        self._serialRunner(cmd, log_path)
        self.logger.info("Exported GS from Splatfacto.")

        export_path = self.runDir / "output/splat.ply"
        self._check_output(export_path, "run_splatfacto")

        self.logger.info("Finished running splatfacto.")

    def run_postprocess(self) -> None:
        """
        Applies post-processing to the generated Gaussian Splat and converts to `.splat`

        """
        # Paths
        out_path = self.runDir / "output/postprocessed/"
        out_path.mkdir(parents=True, exist_ok=True)
        inp = self.runDir / "output/splat.ply"
        out = out_path / "postprocessed.splat"

        try:
            self.logger.info(f"Post-processing GS for {self.meshStr}.")
            postproc = PostProcess(
                input_path=inp,
                output_path=out,
                runDir=self.runDir,
                log_path=self.log_path,
            )
            self.logger.info(f"Check log file: {postproc.log_path}.")
            postproc.run_ops()
        except Exception as e:
            self._handle_error(e, "run_postprocess")

        self._check_output(out, "run_postprocess")

        # Set for further processing
        self.opt_path = out_path

        self.logger.info(f"Finished post-processing GS for {self.meshStr}.")


class PointOps(BaseOps):
    """
    Runs the Gaussian splatting pipeline

    """

    def __init__(self, meshID: str) -> None:
        super().__init__(meshID=meshID, kind="Point")
        self.opt_path = ""
        # Specify the run order (else alphabetical order)
        self._run_order = [
            "run_vggt",
        ] + self._run_order_suffix

    def run_vggt(self) -> None:
        """
        Creates PointCloud using VGGT
        """
        log_path = self.log_path / "VGGT.log"
        output_path = self.runDir / "output/"
        output_path.mkdir(parents=True, exist_ok=True)

        # Full paths from settings
        # vggt_script = settings.VGGT_SCRIPT_PATH
        # vggt_env = settings.VGGT_ENV_PATH
        # venv_python = Path(vggt_env) / "bin/python"

        # if not venv_python.exists():
        #     self._handle_error(
        #         FileNotFoundError(f"Python not found in {venv_python}"), "run_vggt"
        #     )

        # self.logger.info("Running VGGT pipeline...")

        # # Compose the command
        # cmd = f"{venv_python} {vggt_script} --image_dir {self.imageDir} --output_dir {output_path} --binary --prediction_mode 'Pointmap Branch'"

        # self._serialRunner(cmd, log_path)

        # # Check for expected output file
        # expected_ply = output_path / "Point.ply"
        # expected_voxel_ply = output_path / "Point_Voxel.ply"
        # self._check_output(expected_ply, "run_vggt")

        # self.logger.info("Finished running VGGT pipeline.")

        # self.opt_path = output_path


"""
Tasks

"""


def prerun_check(contrib_id: str, recons_type: str) -> tuple[bool, str]:
    """
    Checks if a contribution is ready for processing.

    Parameters
    ----------
    contrib_id : str
        Contribution ID
    recons_type : str
        Reconstruction type
        Options: 'GS', 'aV'

    Returns
    -------
    tuple[bool, str]
        (status, message)

    """
    contrib = Contribution.objects.get(ID=contrib_id)
    mesh = contrib.mesh
    images_count = len(os.listdir(MEDIA / f"models/{mesh.ID}/images/good"))

    # Check if mesh is already being processed or completed
    if mesh.completed:
        return False, "Mesh already completed."
    if mesh.status == "Processing":
        return False, "Mesh already processing."
    if images_count < MESHOPS_MIN_IMAGES:
        return (
            False,
            f"Not enough images to process mesh. Only {images_count} good images found.",
        )
    if mesh.reconstructed_at and (contrib.processed_at < mesh.reconstructed_at):
        runs = mesh.runs.filter(status="Archived").order_by("-ended_at")
        if runs and (runs[0].kind == recons_type):
            return (
                False,
                f"Mesh already reconstructed using {recons_type} using current contribution.",
            )

    return True, "Mesh ready for processing."


def ops_runner(contrib_id: str, kind: str) -> None:
    """
    Runs the appropriate operations on a `models.Mesh` and publishes the results.

    Parameters
    ----------
    contrib_id : str
        Contribution ID
    kind : str
        Kind of operation to run
        Options: 'aV' (AliceVision) or 'GS' (Gaussian Splatting)

    """
    ops_map = {"aV": MeshOps, "GS": GSOps, "Point": PointOps}
    OP = ops_map[kind]
    op_name = OP.__name__

    contrib = Contribution.objects.get(ID=contrib_id)
    meshID = str(contrib.mesh.ID)
    meshVID = str(contrib.mesh.verbose_id)

    cons = Console()  # This appears as normal printed logs in celery logs.
    cons.rule(f"{op_name} Runner Start")
    cons.print(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Triggered by {contrib} for Mesh {meshVID} <=> {meshID}."
    )
    cons.print(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Starting {op_name} on Mesh {meshVID} <=> {meshID}."
    )

    try:
        op = OP(meshID=meshID)
        cons.print(f"Check {op.log_path} for more details.")
        op._run_all()
        cons.print(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Finished {op_name} on {meshVID} <=> {meshID}."
        )
        """
        #NOTE: Sending email to the contributor when the reconstruction is done.
        """
        reconstruction = {
            "aV": "AliceVision",
            "GS": "Gaussian Splatting",
            "Point": "Point Cloud Reconstruction",
        }
        template = render_to_string(
            "tirtha/email_template.html",
            {
                "name": contrib.contributor.name,
                "meshVID": meshVID,
                "meshID": meshID,
                "reconstruction": reconstruction[kind],
                "reconstruction_link": f"{BASE_URL}/models/{meshID}",  # NOTE: Add proper link when ready
                "reconstruction_name": contrib.mesh.name,
            },
        )
        email = EmailMultiAlternatives(
            subject="Thank you for your contribution!",
            body=(
                f"Hi {contrib.contributor.name},\n\n"
                f"Your reconstruction '{contrib.mesh.name}' has been successfully completed using the {reconstruction[kind]} method.\n"
                f"View it here: {f'{BASE_URL}/models/{meshID}'}\n\n"
                "Thank you for contributing to Project Tirtha!"
            ),
            from_email=settings.EMAIL_HOST_USER,
            to=[contrib.contributor.email],
            reply_to=[],
        )

        # Attach the HTML version
        email.attach_alternative(template, "text/html")

        try:
            email.send()
        except Exception as e:
            print(f"Failed to send email: {e}")

    except Exception as e:
        cons.print(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: ERROR encountered in {op_name} for {'meshVID'} <=> {'meshID'}!"
        )
        cons.print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {e}")
        raise e
    cons.rule(f"{op_name} Runner End")
