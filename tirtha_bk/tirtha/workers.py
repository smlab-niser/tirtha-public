import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from subprocess import STDOUT, CalledProcessError, check_output
from typing import Optional

# ImageOps
import cv2
import pytz
from django.conf import settings
from django.db import IntegrityError
from django.utils import timezone
from nn_models.MANIQA.batch_predict import MANIQAScore  # Local import
from nsfw_detector import predict  # Local package installation
from rich.console import Console
from silence_tensorflow import silence_tensorflow  # To suppress TF warnings

os.environ[
    "TF_FORCE_GPU_ALLOW_GROWTH"
] = "true"  # To force nsfw_detector model to occupy only necessary GPU memory
silence_tensorflow()  # To suppress TF warnings

# Local imports
from tirtha.models import ARK, Contribution, Image, Mesh, Run

from .alicevision import AliceVision
from .utils import Logger
from .utilsark import generate_noid, noid_check_digit

STATIC = Path(settings.STATIC_ROOT)
MEDIA = Path(settings.MEDIA_ROOT)
LOG_DIR = Path(settings.LOG_DIR)
ARCHIVE_ROOT = Path(settings.ARCHIVE_ROOT)
MESHOPS_MIN_IMAGES = settings.MESHOPS_MIN_IMAGES
ALICEVISION_DIRPATH = settings.ALICEVISION_DIRPATH
NSFW_MODEL_DIRPATH = settings.NSFW_MODEL_DIRPATH
OBJ2GLTF_PATH = settings.OBJ2GLTF_PATH
GLTFPACK_PATH = settings.GLTFPACK_PATH
BASE_URL = settings.BASE_URL
ARK_NAAN = settings.ARK_NAAN
ARK_SHOULDER = settings.ARK_SHOULDER


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
        self.run = run = Run.objects.create(mesh=mesh)
        run.save()  # Creates run directory
        self.runID = runID = run.ID

        # Set up Logger
        self.log_path = LOG_DIR / f"MeshOps/{meshID}/"
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
        self.runDir = STATIC / "models" / Path(run.directory)

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
            # NOTE: Won't appear in VSCode's Jupyter Notebook (March 2023)
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
        arcDir = ARCHIVE_ROOT / self.meshID / "cache" / curr_runID

        self.logger.info(f"Cleaning up runs for mesh {meshStr}.")
        try:
            # 1. Delete errored-out runs
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
            "This ARK was generated & is managed by Project Tirtha (https://tirtha.niser.ac.in). "
            + "We are committed to maintaining this ARK as per our Terms of Use (https://tirtha.niser.ac.in/#terms) "
            + "and Privacy Policy (https://tirtha.niser.ac.in/#privacy)."
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

        def _update_image(img, label, remark):
            lg.info(f"Updating image {img.ID} with label {label} and remark {remark}.")
            img.label = label
            img.remark = remark
            img.save()  # `pre_save`` signal handles moving file to the correct folder
            lg.info(f"Updated image {img.ID} with label {label} and remark {remark}.")

        manr = MANIQAScore(cpu_num=16, num_crops=20)
        for idx, img in enumerate(self.images):
            lg.info(f"Checking image {img.ID} | [{idx}/{self.size}]...")
            img_path = str((MEDIA / img.image.name).resolve())

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
