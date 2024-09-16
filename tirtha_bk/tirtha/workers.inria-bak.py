"""
This backup copy of workers.py contains GS code from Inria.

"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from subprocess import STDOUT, CalledProcessError, check_output
from typing import Optional

# For .splat conversion
import numpy as np
from io import BytesIO
from plyfile import PlyData

# ImageOps
import cv2
import pytz
from django.conf import settings
from django.db import IntegrityError
from django.utils import timezone

# from nn_models.MANIQA.batch_predict import MANIQAScore  # Local import # FIXME: TODO: Uncomment once fixed.
# from nsfw_detector import predict  # Local package installation # FIXME: TODO: Uncomment once fixed.
from rich.console import Console
from silence_tensorflow import silence_tensorflow  # To suppress TF warnings

# FIXME: TODO: Uncomment once fixed.
# os.environ[
#     "TF_FORCE_GPU_ALLOW_GROWTH"
# ] = "true"  # To force nsfw_detector model to occupy only necessary GPU memory
# silence_tensorflow()  # To suppress TF warnings

# Local imports
from tirtha.models import ARK, Contribution, Image, Mesh, Run

from .alicevision import AliceVision
from .utils import Logger
from .utilsark import generate_noid, noid_check_digit

from .preprocessing import Preprocess
from .optimization import Optimization
from .filter import Filter


STATIC = Path(settings.STATIC_ROOT)
MEDIA = Path(settings.MEDIA_ROOT)
LOG_DIR = Path(settings.LOG_DIR)
ARCHIVE_ROOT = Path(settings.ARCHIVE_ROOT)
GS_SAVE_ITERS = settings.GS_SAVE_ITERS
GS_MAX_ITER = settings.GS_MAX_ITER
GS_CONVERTER = settings.GS_CONVERTER_PATH
MESHOPS_MIN_IMAGES = settings.MESHOPS_MIN_IMAGES
ALICEVISION_DIRPATH = settings.ALICEVISION_DIRPATH
NSFW_MODEL_DIRPATH = settings.NSFW_MODEL_DIRPATH
MANIQA_MODEL_FILEPATH = settings.MANIQA_MODEL_FILEPATH
OBJ2GLTF_PATH = settings.OBJ2GLTF_PATH
GLTFPACK_PATH = settings.GLTFPACK_PATH
BASE_URL = settings.BASE_URL
ARK_NAAN = settings.ARK_NAAN
ARK_SHOULDER = settings.ARK_SHOULDER
COLMAP = settings.SPLATFACTO_COLMAP
# COLMAP_FOLDER = setting.COLMAP_FOLDER


class MeshOps:
    """
    Mesh processing pipeline. Does the following:
    - Runs the aliceVision pipeline on a given `models.Mesh`
    - Runs `obj2gltf` to convert the obj file to gltf
    - Runs `gltfpack` (meshopt) to optimize the gltf file
    - Publishes the final output to the Tirtha site

    """

    def __init__(self, meshID: str):
        self.meshID = meshID
        self.mesh = mesh = Mesh.objects.get(ID=meshID)
        self.meshVID = mesh.verbose_id
        self.meshStr = f"{self.meshVID} <=> {self.meshID}"  # Used in logging

        # Create new Run
        self.run = run = Run.objects.create(mesh=mesh, kind="aV")
        run.save()  # Creates run directory
        self.runID = runID = run.ID
        self.runDir = STATIC / "models" / Path(run.directory)

        # Set up Logger
        self.log_path = LOG_DIR / f"MeshOps/{meshID}/" / self.runDir.stem
        if not self.log_path.exists():
            self.log_path.mkdir(parents=True, exist_ok=True)

        self.cls = cls = self.__class__
        # NOTE: Attributes with self.__class__ will only work if the class is instantiated.
        cls.logger = Logger(log_path=self.log_path, name=f"{cls.__name__}_{runID[:8]}")
        cls.logger.info(
            f"ID {meshID} has Verbose ID (VID) {self.meshVID}. Using VID for logging."
        )

        # Source (images) & run directories
        self.imageDir = MEDIA / f"models/{meshID}/images/good/"

        cls.logger.info(f"Created new Run {runID} for mesh {self.meshStr}.")
        cls.logger.info(f"Run directory: {self.runDir}")
        cls.logger.info(f"Run log file: {cls.logger._log_file}")
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
                f"Set the `images` & `contributors` attributes of Run {runID}."
            )
        except Exception as e:
            self._handle_error(excep=e, caller="Fetching images & contributors")

        # Executables
        self.aV_exec = Path(ALICEVISION_DIRPATH)
        self.obj2gltf_exec = Path(OBJ2GLTF_PATH)
        self.gltfpack_exec = Path(GLTFPACK_PATH)
        self._check_exec(path=self.aV_exec)
        self._check_exec(exe=self.obj2gltf_exec)
        self._check_exec(exe=self.gltfpack_exec)

        # Logger setup
        MeshOps.av_logger = Logger(
            log_path=self.runDir,
            name=f"alicevision__{self.runID}",
        )

        # Specify the run order (else alphabetical order)
        self._run_order = [
            "run_aliceVision",
            "run_obj2gltf",
            "run_meshopt",
            "run_cleanup",
            "run_ark",
            "run_finalize",
        ]

    def _check_exec(self, exe: str = None, path: Path = None):
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

    def _update_mesh_status(self, status: str):
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

    def _update_run_status(self, status: str):
        """
        Updates the run status in the DB

        Parameters
        ----------
        status : str
            The status to update the run to

        """
        self.run.status = status
        self.run.save()
        self.logger.info(f"Updated run.status to '{status}' for run {self.runID}...")

    def _handle_error(self, excep: Exception, caller: str):
        """
        Handles errors during the worker execution

        Parameters
        ----------
        excep : Exception
            The exception that was raised
        caller : str
            The name of the function that raised the exception

        """
        # Log the error
        self.logger.error(f"{caller} step failed for mesh {self.meshStr}.")
        self.logger.error(f"{excep}", exc_info=True)

        # Update statuses to 'Error'
        self._update_mesh_status("Error")
        self.run.ended_at = timezone.now()
        self.run.save()
        self._update_run_status("Error")

        raise excep

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

    def _run_all(self):
        """
        Runs all the steps with default parameters.

        Raises
        ------
        ValueError
            If `_run_order` is not defined

        """
        if self._run_order:
            for step in self._run_order:
                getattr(self, step)()
        else:
            raise ValueError("_run_order is not defined.")

    def run_aliceVision(self):
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
        except Exception as e:
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

        except Exception as e:
            self._handle_error(
                Exception(f"aliceVision pipeline failed for mesh {meshStr}."),
                "aliceVision",
            )

    def run_obj2gltf(self, options: Optional[dict] = {"secure": "true"}):
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
        log_path = out_path / f"obj2gltf.log"
        self.logger.info(f"Running obj2gltf for mesh {self.meshStr}.")
        self.logger.info(f"Command: {cmd}")
        self._serialRunner(cmd, log_path)

        # Check if output was produced
        if not out.is_file():
            self._handle_error(
                FileNotFoundError(f"Output not produced for {out}."), "meshopt"
            )

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
    ):
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
        log_path = out_path / f"meshopt.log"
        self.logger.info(f"Running meshopt (gltfpack) for mesh {self.meshStr}.")
        self.logger.info(f"Commands: {cmd}")
        self._serialRunner(cmd, log_path)

        # Check if output was produced
        if not out.is_file():
            self._handle_error(
                FileNotFoundError(f"Output not produced for {out}."), "meshopt"
            )

        # Set for further processing
        self.opt_path = out_path

        self.logger.info(
            f"Finished running meshopt (gltfpack) for mesh {self.meshStr}."
        )

    def run_cleanup(self):
        """
        Does the following:
        1. Cleans up older errored-out runs.
        2. Copies current run's .glb to "published".
        2. Archives current run.
        NOTE: Errored-out runs are not deleted during their run, but only during the next run.
        This is done to allow for debugging.

        """
        meshStr = self.meshStr
        curr_runID = self.runID
        arcDir = ARCHIVE_ROOT / self.meshID / "cache" / self.runDir.stem

        self.logger.info(f"Cleaning up runs for mesh {meshStr}.")
        try:
            # 1. Delete errored-out runs (including GS runs)
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
                        f"Run {run.ID} for mesh {meshStr} has errors. Deleting..."
                    )
                    shutil.rmtree(runDir)  # Delete folder
                    run.delete()  # Delete from DB
                    self.logger.info(f"Deleted run {run.ID} for mesh {meshStr}.")
            self.logger.info(
                f"Deleted {len(runs)} errored-out runs for mesh {meshStr}."
            )

            # 2. Copy current run's .glb to STATIC / "models" / meshID / "published" / run.id.glb
            self.logger.info(f"Copying .glb for run {curr_runID} for mesh {meshStr}...")
            src = self.opt_path / "decimatedOptGLB.glb"
            self.arkURL = (
                f"models/{self.meshID}/published/{self.meshID}_{curr_runID}.glb"
            )
            dest = STATIC / self.arkURL
            shutil.copy2(src, dest)
            self.logger.info(f"Copied .glb for run {curr_runID} for mesh {meshStr}.")

            # 3. Move everything else to arcDir
            self.logger.info(
                f"Archiving run {curr_runID} for mesh {meshStr} to {arcDir}."
            )
            shutil.move(self.runDir, arcDir)  # Move run folder to archive
            self.run.directory = str(arcDir)  # Update run directory
            self._update_run_status("Archived")  # Update run status & save
            self.logger.info(
                f"Archived run {curr_runID} for mesh {meshStr} to {arcDir}."
            )

        except Exception as e:
            self.logger.error(f"Error cleaning up runs for mesh {meshStr}.")
            self._handle_error(e, "Archiver")
        self.logger.info(f"Finished cleaning up runs for mesh {meshStr}.")

    def run_ark(self, ark_len=16):
        """
        Runs the ark generator to generate a unique ark for the run.

        Parameters
        ----------
        ark_len : int
            Length of the ark to be generated
            Default: 16

        """
        arkURL, run, mesh, meshStr = self.arkURL, self.run, self.mesh, self.meshStr
        naan, shoulder = ARK_NAAN, ARK_SHOULDER
        run.ended_at = timezone.now()

        # Create metadata
        self.logger.info(
            f"Creating metadata for ARK for run {run.ID} for mesh {meshStr}..."
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
            f"Created metadata for ARK for run {run.ID} for mesh {meshStr}..."
        )

        # Generate ark - NOTE: Adapted from arklet
        self.logger.info(f"Generating ARK for run {run.ID} for mesh {meshStr}...")
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
        msg = f"Generated ARK for run {run.ID} for mesh {meshStr} after {collisions} collision(s)."
        if collisions == 0:
            self.logger.info(msg)
        else:
            self.logger.warning(msg)

        # Update run
        run.ark = ark
        run.save()
        self.arkStr = str(ark_string)

    def run_finalize(self):
        """
        Finalizes current run

        """
        self.logger.info(f"Finalizing run {self.runID} for mesh {self.meshStr}.")
        self.mesh.reconstructed_at = datetime.now(pytz.timezone("Asia/Kolkata"))
        self.logger.info(f"Run {self.runID} finished for mesh {self.meshStr}.")
        self._update_mesh_status("Live")
        self.logger.info(
            f"Finished finalizing run {self.runID} for mesh {self.meshStr}."
        )


# LATE_EXP: FIXME: This can be made faster:
# 1. By incorporating with the task itself, and removing the need for model load for each new contribution.
# 2. By batching images and inferencing on patches.
class ImageOps:
    """
    Image pre-processing pipeline. Does the following:
    - Passes images through a NSFW filter [1] and moves flagged images into `images/nsfw` folder for manual moderation.
    - Sorts images into `images/[good / bad]` by Contrast-to-Noise ratio, Dynamic Range & results from No-Reference Image
    Quality Assessment (MANIQA [2]) for each image in a contribution.

    Parameters
    ----------
    contrib_id : str
        Contribution ID

    References
    ----------
    .. [1] NSFW Detector, Laborde, Gant; https://github.com/GantMan/nsfw_model
    .. [2] Yang, S., Wu, T., Shi, S., Lao, S., Gong, Y., Cao, M., Wang, J.,
        & Yang, Y. (2022). MANIQA: Multi-dimension Attention Network for
        No-Reference Image Quality Assessment. In Proceedings of the IEEE/CVF
        Conference on Computer Vision and Pattern Recognition (pp. 1191-1200).

    """

    def __init__(self, contrib_id: str):
        self.logger = Logger(
            log_path=Path(LOG_DIR) / "ImageOps",
            name=f"{self.__class__.__name__}_{contrib_id[:8]}",
        )

        try:
            self.logger.info(
                f"Accessing DB to get the contribution object for Contribution ID, {contrib_id}..."
            )
            self.contribution = Contribution.objects.get(ID=contrib_id)
        except Contribution.DoesNotExist as excep:
            self.logger.error(
                f"Contribution with ID {contrib_id} does not exist in the DB.",
                exc_info=True,
            )
            raise ValueError(f"Contribution {contrib_id} not found.") from excep

        self.mesh = self.contribution.mesh
        self.images = self.contribution.images.all()
        self.size = len(self.images)
        if self.size == 0:
            msg = f"No images found for contribution {self.contribution.ID}."
            self.logger.error(msg)
            raise ValueError(msg)
        self.logger.info(
            f"Found {self.size} images for contribution {self.contribution.ID}."
        )

        if NSFW_MODEL_DIRPATH is not None and not NSFW_MODEL_DIRPATH.exists():
            self.logger.error(f"nsfw_detector model not found at {NSFW_MODEL_DIRPATH}.")
            raise FileNotFoundError(
                f"nsfw_detector model not found at {NSFW_MODEL_DIRPATH}."
            )

        # Thresholds & Weights
        self.thresholds = {"CS": 0.95, "DR": 100, "CNR": 17.5, "MANIQA": 0.6}

        self.weights = {  # NOTE: TODO: Unused for now
            "DR": 0.1,
            "CNR": 0.15,
            "MANIQA": 0.75,
        }

    def check_content_safety(self, img_path: str):
        """
        Runs content safety filters on 1 image

        """
        local_cs_model = predict.load_model(NSFW_MODEL_DIRPATH)
        local_cs_res = predict.classify(local_cs_model, img_path).values()

        for val in local_cs_res:
            pos = val["neutral"] + val["drawings"]
            # neg = val['hentai'] + val['porn'] + val['sexy']

        return True if pos > self.thresholds["CS"] else False

    def check_images(self):
        """
        Checks images for NSFW content & quality and sorts them into
        `images/[good / bad]` folders.
        NOTE: No sRGB linearization is done here, as decision boundaries
        or thresholds are hard to delineate in linear sRGB space.

        """
        lg = self.logger

        # FIXME: TODO: Till the VRAM + concurrency issue is fixed, skip image checks.
        # FIXME: TODO: Remove once fixed.
        lg.info(
            f"NOTE: Skipping image checks for contribution {self.contribution.ID} due to VRAM + concurrency issues. FIXME:"
        )

        def _update_image(img, label, remark):
            lg.info(f"Updating image {img.ID} with label {label} and remark {remark}.")
            img.label = label
            img.remark = remark
            img.save()  # `pre_save`` signal handles moving file to the correct folder
            lg.info(f"Updated image {img.ID} with label {label} and remark {remark}.")

        # manr = MANIQAScore(ckpt_pth=MANIQA_MODEL_FILEPATH, cpu_num=32, num_crops=20)
        # FIXME: TODO: Uncomment once fixed.
        for idx, img in enumerate(self.images):
            lg.info(f"Checking image {img.ID} | [{idx}/{self.size}]...")
            # img_path = str((MEDIA / img.image.name).resolve()) # FIXME: TODO: Uncomment once fixed.

            # FIXME: TODO: Remove (till `continue`) once fixed
            # Skip & move image to good folder
            _update_image(img, "good", f"PASS -- SKIPPED")
            continue

            # Content safety check
            if not self.check_content_safety(img_path):
                _update_image(
                    img, "nsfw", "NSFW content detected by local NSFW filter."
                )
                continue

            # Quality check
            rgb_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
            # DR
            dr = (gray_img.max() - gray_img.min()) * 100 / 255
            if dr < self.thresholds["DR"]:
                _update_image(
                    img,
                    "bad",
                    f"FAIL -- DR: {dr:.4f}; Rejected by DR threshold: {self.thresholds['DR']}.",
                )
                continue

            # CNR
            cnr = gray_img.std() ** 2 / gray_img.mean()
            if cnr < self.thresholds["CNR"]:
                _update_image(
                    img,
                    "bad",
                    f"FAIL -- DR: {dr:.4f}, CNR: {cnr:.4f}; Rejected by CNR threshold: {self.thresholds['CNR']}.",
                )
                continue

            # MANIQA
            iqa_score = float(manr.predict_one(img_path).detach().cpu().numpy())
            if iqa_score < self.thresholds["MANIQA"]:
                _update_image(
                    img,
                    "bad",
                    f"FAIL -- DR: {dr:.4f}, CNR: {cnr:.4f}, MANIQA: {iqa_score:.4f}; Rejected by MANIQA threshold: {self.thresholds['MANIQA']}.",
                )
                continue

            # If all pass, add dr, cnr, iqa_score as a remark to Image & move to good folder
            _update_image(
                img,
                "good",
                f"PASS -- DR: {dr:.4f}, CNR: {cnr:.4f}, MANIQA: {iqa_score:.4f}; Thresholds: {self.thresholds}.",
            )

        lg.info(f"Finished checking images for contribution {self.contribution.ID}.")
        lg.info("Marking contribution as `processed` & updating `processed_at`.")
        Contribution.objects.filter(
            ID=self.contribution.ID
        ).update(  # Does not trigger signals
            processed=True, processed_at=timezone.now()
        )
        lg.info("Finished updating contribution.")


class GSOps:
    """
    TODO: @Anubhav Create a base class and reuse a ton of code
    Runs the Gaussian splatting pipeline

    """

    def __init__(self, meshID: str, saving_iterations: int):
        self.nerf_use = False  # Whether nerfstudio is used for Gaussian Splatting
        self.meshID = meshID
        self.saving_iterations = saving_iterations
        self.mesh = mesh = Mesh.objects.get(ID=meshID)
        self.meshVID = mesh.verbose_id
        self.meshStr = f"{self.meshVID} <=> {self.meshID}"  # Used in logging

        # Create new Run
        self.run = run = Run.objects.create(mesh=mesh, kind="GS")
        run.save()  # Creates run directory
        self.runID = runID = run.ID
        self.runDir = STATIC / "models" / Path(run.directory)

        # Set up Logger
        self.log_path = LOG_DIR / f"GSOps/{meshID}" / self.runDir.stem
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

        cls.logger.info(f"Created new GS Run {runID} for mesh {self.meshStr}.")
        cls.logger.info(f"GS Run directory: {self.runDir}")
        cls.logger.info(f"GS Run log file: {cls.logger._log_file}")
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
                f"Set the `images` & `contributors` attributes of Run {runID}."
            )
        except Exception as e:
            self._handle_error(excep=e, caller="Fetching images & contributors")

        # Specify the run order (else alphabetical order)
        # self._run_order = [
        #     "run_preprocess",
        #     "run_optimization",
        #     "run_filtering",
        #     "run_convert",
        #     "run_cleanup",
        #     "run_ark",
        #     "run_finalize",
        # ]
        self._run_order = [
            "run_splatfacto",
            "run_filtering",
            "run_convert",
            "run_cleanup",
            "run_ark",
            "run_finalize",
        ]

    def _update_mesh_status(self, status: str):
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

    def _update_run_status(self, status: str):
        """
        Updates the run status in the DB

        Parameters
        ----------
        status : str
            The status to update the run to

        """
        self.run.status = status
        self.run.save()
        self.logger.info(f"Updated run.status to '{status}' for run {self.runID}...")

    def _handle_error(self, excep: Exception, caller: str):
        """
        Handles errors during the worker execution

        Parameters
        ----------
        excep : Exception
            The exception that was raised
        caller : str
            The name of the function that raised the exception

        """
        # Log the error
        self.logger.error(f"{caller} step failed for mesh {self.meshStr}.")
        self.logger.error(f"{excep}", exc_info=True)

        # Update statuses to 'Error'
        self._update_mesh_status("Error")
        self.run.ended_at = timezone.now()
        self.run.save()
        self._update_run_status("Error")

        raise excep

    @classmethod  # NTE: Won't be picklable as a regular method
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

    def run_preprocess(self):
        """
        Runs the preprocessing pipeline on the imageset

        """
        self.logger.info(f"Preprocessing images for mesh {self.meshStr}.")
        process_images = Preprocess(
            self.meshID, self.imageDir, self.runDir, self.log_path
        )
        self.logger.info(f"Check log file: {process_images.log_path}.")

        # Run preprocessing
        process_images._run_all()

    def run_optimization(self):
        """
        Runs the main GS optimization pipeline on the imageset

        """
        self.logger.info(f"Creating GS for mesh {self.meshStr}.")
        optimization = Optimization(self.saving_iterations, self.runDir, self.log_path)
        self.logger.info(f"Check log file: {optimization.log_path}.")

        # Run the optimization pipeline
        optimization._run_all()

    def run_filtering(self):
        """
        Runs the filtering pipeline on the generated Gaussian splat point clouds
        # TODO: @Anubhav Include the relevant code in the repo instead of using the python lib / command.

        """

        # Paths and condtions to either use splatfactio by nerf or gaussia splatting by inria for filtering
        if self.nerf_use:
            out_path = self.runDir / "output/filtered/"
            out_path.mkdir(parents=True, exist_ok=True)
            inp = self.runDir / "output/splat.ply"
            out = out_path / "filtered.ply"

        else:
            out_path = self.runDir / "output/filtered/"
            out_path.mkdir(parents=True, exist_ok=True)
            inp = (
                self.runDir
                / f"output/point_cloud/iteration_{GS_MAX_ITER}/point_cloud.ply"
            )
            out = out_path / "filtered.ply"

        self.logger.info(f"Filtering the GS for floaters {self.meshStr}.")
        filter = Filter(inp, out, self.runDir, self.log_path)
        self.logger.info(f"Check log file: {filter.log_path}.")

        # Build command
        # cmd_init = f"{GS_CONVERTER} -i "
        # opts = "-f 3dgs --density_filter --remove_flyers"
        # cmd = cmd_init + f"{inp} -o {out} {opts}"
        # log_path = self.log_path / f"filtering.log"

        # self.logger.info(f"Running filtering for generated point clouds {self.meshStr}.")
        # self.logger.info(f"Command: {cmd}")
        # self._serialRunner(cmd, log_path)

        # Check if output was produced
        if not out.is_file():
            self._handle_error(
                FileNotFoundError(f"Output not produced for {out}."), "Filtering"
            )

        self.logger.info(f"Finished filtering GS for mesh: {self.meshStr}.")

    def run_convert(self):
        """
        Taken from https://github.com/antimatter15/splat/blob/main/convert.py
        Licensed under the MIT License

        """
        # Paths
        input_file = self.runDir / "output/filtered/filtered.ply"
        output_file = self.runDir / "output/filtered/filtered.splat"

        self.logger.info("Converting .ply to .splat...")
        self.logger.info(f"Input file: {input_file}.")

        # TODO: @Anubhav Optimize & rewrite + add filtering with params here
        plydata = PlyData.read(input_file)
        vert = plydata["vertex"]
        sorted_indices = np.argsort(
            -np.exp(vert["scale_0"] + vert["scale_1"] + vert["scale_2"])
            / (1 + np.exp(-vert["opacity"]))
        )
        buffer = BytesIO()
        for idx in sorted_indices:
            v = plydata["vertex"][idx]
            position = np.array([v["x"], v["y"], v["z"]], dtype=np.float32)
            scales = np.exp(
                np.array(
                    [v["scale_0"], v["scale_1"], v["scale_2"]],
                    dtype=np.float32,
                )
            )
            rot = np.array(
                [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]],
                dtype=np.float32,
            )
            SH_C0 = 0.28209479177387814
            color = np.array(
                [
                    0.5 + SH_C0 * v["f_dc_0"],
                    0.5 + SH_C0 * v["f_dc_1"],
                    0.5 + SH_C0 * v["f_dc_2"],
                    1 / (1 + np.exp(-v["opacity"])),
                ]
            )
            buffer.write(position.tobytes())
            buffer.write(scales.tobytes())
            buffer.write((color * 255).clip(0, 255).astype(np.uint8).tobytes())
            buffer.write(
                ((rot / np.linalg.norm(rot)) * 128 + 128)
                .clip(0, 255)
                .astype(np.uint8)
                .tobytes()
            )

        splat_data = buffer.getvalue()

        # Save
        self.logger.info(f"Saving to {output_file}.")
        with open(output_file, "wb") as f:
            f.write(splat_data)
        self.logger.info("Converted .ply to .splat.")

    def run_cleanup(self):
        """
        Does the following:
        1. Cleans up older errored-out runs.
        2. Copies current Gaussian run's .splat to "published".
        2. Archives current run.
        NOTE: Errored-out runs are not deleted during their run, but only during the next run.
        This is done to allow for debugging.

        """
        meshStr = self.meshStr
        curr_runID = self.runID
        arcDir = ARCHIVE_ROOT / self.meshID / "gscache" / self.runDir.stem

        self.logger.info(f"Cleaning up runs for mesh {meshStr}.")
        try:
            # 1. Delete errored-out runs (including aV runs)
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
                        f"GS Run {run.ID} for mesh {meshStr} has errors. Deleting..."
                    )
                    shutil.rmtree(runDir)  # Delete folder
                    run.delete()  # Delete from DB
                    self.logger.info(f"Deleted run {run.ID} for mesh {meshStr}.")
            self.logger.info(
                f"Deleted {len(runs)} errored-out runs for mesh {meshStr}."
            )

            # 2. Copy current run's .splat to STATIC / "models" / meshID / "published" / run.id.splat
            self.logger.info(
                f"Copying .splat for GS run {curr_runID} for mesh {meshStr}..."
            )
            src = self.runDir / f"output/filtered/filtered.splat"
            self.arkURL = (
                f"models/{self.meshID}/published/{self.meshID}_{curr_runID}.splat"
            )
            dest = STATIC / self.arkURL
            shutil.copy2(src, dest)
            self.logger.info(
                f"Copied .splat for GS run {curr_runID} for mesh {meshStr}."
            )

            # 3. Move everything else to arcDir
            self.logger.info(
                f"Archiving GS run {curr_runID} for mesh {meshStr} to {arcDir}."
            )
            shutil.move(self.runDir, arcDir)  # Move run folder to archive
            self.run.directory = str(arcDir)  # Update run directory
            self._update_run_status("Archived")  # Update run status & save
            self.logger.info(
                f"Archived run {curr_runID} for mesh {meshStr} to {arcDir}."
            )

        except Exception as e:
            self.logger.error(f"Error cleaning up runs for mesh {meshStr}.")
            self._handle_error(e, "Archiver")
        self.logger.info(f"Finished cleaning up runs for mesh {meshStr}.")

    def run_ark(self, ark_len=16):
        """
        Runs the ark generator to generate a unique ark for the run.

        Parameters
        ----------
        ark_len : int
            Length of the ark to be generated
            Default: 16

        """
        arkURL, run, mesh, meshStr = self.arkURL, self.run, self.mesh, self.meshStr
        naan, shoulder = ARK_NAAN, ARK_SHOULDER
        run.ended_at = timezone.now()

        # Create metadata
        self.logger.info(
            f"Creating metadata for ARK for Gaussian splatting run {run.ID} for mesh {meshStr}..."
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
            f"Created metadata for ARK for Gaussian splatting run {run.ID} for mesh {meshStr}..."
        )

        # Generate ark - NOTE: Adapted from arklet
        self.logger.info(
            f"Generating ARK for Gaussian splatting run {run.ID} for mesh {meshStr}..."
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
        msg = f"Generated ARK for Gaussian splatting run {run.ID} for mesh {meshStr} after {collisions} collision(s)."
        if collisions == 0:
            self.logger.info(msg)
        else:
            self.logger.warning(msg)

        # Update run
        run.ark = ark
        run.save()
        self.arkStr = str(ark_string)

    def run_finalize(self):
        """
        Finalizes current run

        """
        self.logger.info(f"Finalizing GS run {self.runID} for mesh {self.meshStr}.")
        self.mesh.reconstructed_at = datetime.now(pytz.timezone("Asia/Kolkata"))
        self.logger.info(f"GS Run {self.runID} finished for mesh {self.meshStr}.")
        self._update_mesh_status("Live")
        self.logger.info(
            f"Finished finalizing GS run {self.runID} for mesh {self.meshStr}."
        )

    def run_splatfacto(self):
        """
        Runs COLMAP and creates Gaussian Splats using the `splatfacto` library

        """
        self.nerf_use = True
        # Image path
        # image_path = Path(MEDIA / self.runDir)
        # output path
        output_path = self.runDir / "output/"

        # colmap command
        cmd_init = f"ns-process-data images --data "
        opts = " --output-dir "
        cmd = cmd_init + str(self.imageDir) + f"{opts}" + str(output_path)
        log_path = self.log_path / f"splatfacto_colmap.log"

        self.logger.info(f"Running colamap by splatfacto.")
        self.logger.info(f"Command: {cmd}")
        self._serialRunner(cmd, log_path)

        train_inp = "ns-train splatfacto --data "
        log_path = self.log_path / f"splatfacto_train.log"
        cmd = (
            train_inp
            + str(output_path)
            + " --output-dir "
            + str(output_path)
            + ' --timestamp ""'
            + f" --max-num-iterations {GS_MAX_ITER} "
            + "--viewer.quit-on-train-completion True "
            + "> "
            + str(log_path)
        )

        self.logger.info("Running GS by splatfacto.")
        self.logger.info(f"Command: {cmd}")
        self._serialRunner(cmd, log_path)

        export_inp = "ns-export gaussian-splat --load-config "
        log_path = self.log_path / "splatfacto_export.log"
        cmd = (
            export_inp
            + str(output_path)
            + "/output/splatfacto/config.yml "
            + " --output-dir "
            + str(output_path)
        )

        self.logger.info("Running splatfacto...")
        self.logger.info(f"Command: {cmd}")
        self._serialRunner(cmd, log_path)
        # Check if output was produced - TODO:
        # if not image_path.is_file():
        #     self._handle_error(
        #         FileNotFoundError(f"Input is not not valid for {image_path}."), "splatfacto_colmap"
        #     )

        self.logger.info("Finished running splatfacto.")

    def _run_all(self):
        """
        Runs all the steps with default parameters.

        Raises
        ------
        ValueError
            If `_run_order` is not defined

        """
        if self._run_order:
            for step in self._run_order:
                getattr(self, step)()
        else:
            raise ValueError("_run_order is not defined.")


"""
Tasks

"""


def prerun_check(contrib_id):
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
        return False, "Mesh already reconstructed using current contribution."

    return True, "Mesh ready for processing."


def mo_runner(contrib_id: str):
    """
    Runs MeshOps on a `models.Mesh` and publishes the results.

    """
    contrib = Contribution.objects.get(ID=contrib_id)
    meshID = str(contrib.mesh.ID)
    meshVID = str(contrib.mesh.verbose_id)

    cons = Console()  # This appears as normal printed logs in celery logs.
    cons.rule("MeshOps Runner Start")
    cons.print(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Triggered by {contrib.ID} for Mesh {meshVID} <=> {meshID}."
    )
    cons.print(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Starting MeshOps on Mesh {meshVID} <=> {meshID}."
    )
    try:
        mO = MeshOps(meshID=meshID)
        cons.print(f"Check {mO.log_path} for more details.")
        mO._run_all()
        cons.print(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Finished MeshOps on {meshVID} <=> {meshID}."
        )  # <----------------
    except Exception as e:
        cons.print(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: ERROR encountered in MeshOps for {meshVID} <=> {meshID}!"
        )
        cons.print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {e}")
        raise e
    cons.rule("MeshOps Runner End")


def to_runner(contrib_id: str):
    """
    Runs GSOps on a `models.Mesh` and publishes the results.

    """
    contrib = Contribution.objects.get(ID=contrib_id)
    meshID = str(contrib.mesh.ID)
    meshVID = str(contrib.mesh.verbose_id)

    cons = Console()  # This appears as normal printed logs in celery logs.
    cons.rule("GSOps Runner Start")
    cons.print(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Triggered by {contrib.ID} for Mesh {meshVID} <=> {meshID}."
    )
    cons.print(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Starting GSOps on Mesh {meshVID} <=> {meshID}."
    )
    try:
        tO = GSOps(meshID=meshID, saving_iterations=GS_SAVE_ITERS)
        cons.print(f"Check {tO.log_path} for more details.")
        tO._run_all()
        cons.print(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Finished GSOps on {meshVID} <=> {meshID}."
        )  # <----------------
    except Exception as e:
        cons.print(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: ERROR encountered in GSOps for {meshVID} <=> {meshID}!"
        )
        cons.print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {e}")
        raise e
    cons.rule("GSOps Runner End")
