# import os
import cv2
from pathlib import Path
from django.utils import timezone
from django.conf import settings

# from silence_tensorflow import silence_tensorflow

# Local imports
from tirtha.models import Contribution

# TODO: FIXME: Uncomment when fixed
# TODO: CHECK: for standard model repos, from where we can inference directly instead of local setup.
# from nn_models.MANIQA.batch_predict import MANIQAScore  # Local import
# from nsfw_detector import predict  # Local package installation

from .utils import Logger


# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = (
#     "true"  # To force nsfw_detector model to occupy only necessary GPU memory
# )
# silence_tensorflow()  # To suppress TF warnings


MEDIA = Path(settings.MEDIA_ROOT)
LOG_DIR = Path(settings.LOG_DIR)
ARCHIVE_ROOT = Path(settings.ARCHIVE_ROOT)
GS_MAX_ITER = settings.GS_MAX_ITER
MESHOPS_MIN_IMAGES = settings.MESHOPS_MIN_IMAGES
ALICEVISION_DIRPATH = settings.ALICEVISION_DIRPATH
NSFW_MODEL_DIRPATH = settings.NSFW_MODEL_DIRPATH
MANIQA_MODEL_FILEPATH = settings.MANIQA_MODEL_FILEPATH
OBJ2GLTF_PATH = settings.OBJ2GLTF_PATH
GLTFPACK_PATH = settings.GLTFPACK_PATH
BASE_URL = settings.BASE_URL
ARK_NAAN = settings.ARK_NAAN
ARK_SHOULDER = settings.ARK_SHOULDER


# TODOLATER: FIXME: This can be made faster:
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

    def __init__(self, contrib_id: str) -> None:
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

        # FIXME: TODO: Uncomment once fixed.
        for idx, img in enumerate(self.images):
            lg.info(f"Checking image {img.ID} | [{idx}/{self.size}]...")

            # FIXME: TODO: Remove (till `continue`) once fixed
            # Skip & move image to good folder
            _update_image(img, "good", "PASS -- SKIPPED")
            continue
            manr = MANIQAScore(
                ckpt_pth=MANIQA_MODEL_FILEPATH, cpu_num=32, num_crops=20
            )  # FIXME: TODO: Move out of loop
            img_path = str(
                (MEDIA / img.image.name).resolve()
            )  # FIXME: TODO: Uncomment once fixed.

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
