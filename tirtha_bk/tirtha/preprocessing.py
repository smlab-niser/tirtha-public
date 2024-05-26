"""
The contents of this file are subject to the terms of the Gaussian-Splatting LICENSE present in ./scene/LICENSE.md.

"""
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import shutil
from pathlib import Path
from subprocess import (
    STDOUT,
    CalledProcessError,
    check_output,
)
from django.conf import settings

# Local imports
from .utils import Logger


STATIC = Path(settings.STATIC_ROOT)
MEDIA = Path(settings.MEDIA_ROOT)


class Preprocess:
    def __init__(self, meshID, imageDir, runDir, log_path):
        self.camera = "OPENCV"
        self.imageDir = imageDir
        self.colmap_command = settings.COLMAP_PATH
        self.runDir = runDir
        self.sourcepath = Path(MEDIA / f"models/{meshID}")
        self.log_path = log_path
        self.log_path.mkdir(parents=True, exist_ok=True)
        
        Preprocess.logger = Logger(
            log_path=self.log_path,
            name=f"preprocess",
        )
        print(self.log_path)
        self.use_gpu = True

         # Create directories
        Path(self.runDir / "distorted/sparse").mkdir(parents=True, exist_ok=True)
        Path(self.runDir / "sparse/0").mkdir(parents=True, exist_ok=True)
        

    @staticmethod  # NOTE: Won't be picklable as a regular method
    def _serialRunner(cmd: str, log_file: Path):
        """
        Run a command serially and log the output

        Parameters
        ----------
        cmd : str
            Command to run
        log_file : Path
            Path to the log file

        Raises
        ------
        CalledProcessError
            If the command fails

        """

        logger = Logger(log_file.stem, log_file.parent)
        log_path = Path(log_file).resolve()
        try:
            Preprocess.logger.info(f"Starting command execution. Log file: {log_path}.")
            logger.info(f"Command:\n{cmd}")
            output = check_output(cmd, shell=True, stderr=STDOUT)
            logger.info(f"Output:\n{output.decode().strip()}")
            Preprocess.logger.info(
                f"Finished command execution. Log file: {log_path}."
            )
        except CalledProcessError as error:
            logger.error(f"\n{error.output.decode().strip()}")
            Preprocess.logger.error(
                f"Error in command execution for {logger.name}. Check log file: {log_path}."
            )
            # Preprocess.state = {
            #     "error": True,
            #     "source": logger.name,
            #     "log_file": log_path,
            # }
            # error.add_note(
            #     f"{logger.name} failed. Check log file: {log_path}."
            # )  # NOTE: Won't appear in VSCode's Jupyter Notebook (May 2024)
            # raise error

    def colmap(self):
        log_file = Path(self.log_path / "colmap")
        log_file.mkdir(parents=True, exist_ok=True)

        ## Feature extraction
        feat_extracton_cmd = str(self.colmap_command) + " feature_extractor "\
            "--database_path " + str(self.runDir) + "/distorted/database.db \
            --image_path " + str(self.imageDir) + " \
            --ImageReader.single_camera 1 \
            --ImageReader.camera_model " + str(self.camera) + " \
            --SiftExtraction.use_gpu " + str(self.use_gpu)
        self._serialRunner(feat_extracton_cmd, log_file / "feature_extractor.log")

        ## Feature matching
        feat_matching_cmd = str(self.colmap_command) + " exhaustive_matcher \
            --database_path " + str(self.runDir) + "/distorted/database.db \
            --SiftMatching.use_gpu " + str(self.use_gpu)
        self._serialRunner(feat_matching_cmd, log_file / "exhaustive_matcher.log") 

        ### Bundle adjustment
        # The default Mapper tolerance is unnecessarily large,
        # decreasing it speeds up bundle adjustment steps.
        mapper_cmd = (str(self.colmap_command) + " mapper \
            --database_path " + str(self.runDir) + "/distorted/database.db \
            --image_path "  + str(self.imageDir) + " \
            --output_path "  + str(self.runDir) + "/distorted/sparse \
            --Mapper.ba_global_function_tolerance=0.000001")
        self._serialRunner(mapper_cmd, log_file / "mapper.log") 

        ### Image undistortion
        ## We need to undistort our images into ideal pinhole intrinsics.
        img_undist_cmd = (self.colmap_command + " image_undistorter \
            --image_path " + str(self.imageDir) + " \
            --input_path " + str(self.runDir) + "/distorted/sparse/0 \
            --output_path " + str(self.runDir) + " \
            --output_type COLMAP")
        self._serialRunner(img_undist_cmd, log_file / "image_undistorter.log") 

        files = list(Path(str(self.runDir)+"/sparse").iterdir())

        # Copy each file from the source directory to the destination directory
        for file in files:
            if file == '0':
                continue
            source_file = Path(self.runDir, "sparse", file)
            source_file.mkdir(parents=True, exist_ok=True)
            destination_file = Path(self.runDir, "sparse", "0", file)
            destination_file.mkdir(parents=True, exist_ok=True)
            shutil.move(source_file, destination_file)

    def _run_all(self):
        self.colmap()
