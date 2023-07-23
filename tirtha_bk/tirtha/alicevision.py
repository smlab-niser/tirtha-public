"""
Python API for AliceVision (Meshroom)
For Internal Use

"""
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count
from pathlib import Path
from subprocess import (
    PIPE,
    STDOUT,
    CalledProcessError,
    Popen,
    TimeoutExpired,
    check_output,
)
from time import sleep
from typing import Dict, Iterable, Optional, Union

# Local imports
from .utils import Logger

# NOTE: Tweak as needed
CAMERAINIT_MAX_RETRIES = 5
CAMERAINIT_RETRY_INTERVAL = 1  # seconds
CAMERAINIT_MAX_RUNTIME = 2  # seconds


@dataclass
class AliceVision:
    # NOTE: Path to a folder containinng files from both Meshroom/aliceVision/bin + Meshroom/aliceVision/lib
    # Oherwise, it raises a .so file not found error. .so files reside in lib.
    # `cameraSensors.db` & `vlfeat_K80L3.SIFT.tree` should also be in the this folder
    exec_path: Union[str, Path]  # settings.ALICEVISION_PATH
    input_dir: Union[str, Path]  # Folder with images
    cache_dir: Union[str, Path]  # Folder to store the node outputs | runDir
    logger: Logger  # Logger object from the calling script

    # Optional
    verboseLevel: Optional[
        str
    ] = "info"  # among "trace", "debug", "info", "warning", "error", "fatal" (decreasing verbosity)
    descPresets: Optional[Dict] = field(
        default_factory=lambda: {
            "Preset": "normal",
            "Quality": "normal",
            "Types": "dspsift",  # NOTE: String of comma separated values (no spaces & no trailing comma) NOTE: dspsift is absent from docs, but available in MR21/23.
        }
    )

    def __post_init__(self):
        """
        Initializes the path inputs as `pathlib.Path` objects.
        Creates the output directory if it doesn't exist.
        Stores the node executable names in a dict.

        Raises
        ------
        FileNotFoundError
            If `input_dir` doesn't exist.
        ValueError
            If any attribute of `descPresets` is invalid.

        """
        self.cpu_count = cpu_count()
        # NOTE: Tweak as needed
        # Minimum number of images to process before splitting into blocks
        self.minBlockSize = self.cpu_count * 2
        # Max number of cores to use
        # NOTE: nodes are multithreaded, so using all cores may not be optimal
        self.maxCores = self.cpu_count // 2

        # Tracks the state of the pipeline (intended for when we want to continue despite errors)
        # NOTE: Defined as class variable to be accessible from the static method `_serialRunner`
        AliceVision.state = {"error": False, "source": None, "log_file": None}
        AliceVision.logger = self.logger

        self.exec_path = Path(self.exec_path)
        self.input_dir = Path(self.input_dir)
        if not self.input_dir.exists():
            err = f"Image folder not found at {self.input_dir}."
            self.logger.error(err)
            raise FileNotFoundError(err)

        # check if input_dir is empty
        if not any(self.input_dir.iterdir()):
            err = f"Image folder is empty at {self.input_dir}."
            self.logger.error(err)
            raise FileNotFoundError(err)

        self.cache_dir = Path(self.cache_dir)
        if (
            not self.cache_dir.exists()
        ):  # TODO: Redundant check, since Run.save() creates runDir.
            self.cache_dir.mkdir(parents=True)

        # Check describer presets
        describerPreset, describerQuality, describerTypes = self.descPresets.values()
        allowed_types = [
            "sift",
            "sift_float",
            "sift_upright",
            "dspsift",
            "akaze",
            "akaze_liop",
            "akaze_mldb",
            "cctag3",
            "cctag3",
            "akaze_ocv",
        ]
        describerTypesList = describerTypes.split(",")
        for describerType in describerTypesList:
            if describerType not in allowed_types:
                err = f"Invalid describerType. Allowed values are {allowed_types}. Ensure that there are no spaces in the string, e.g. 'dspsift,akaze,cctag3'."
                self.logger.error(err)
                raise ValueError(err)

        allowed_presets = ["low", "medium", "normal", "high", "ultra"]
        if (
            describerPreset not in allowed_presets
            or describerQuality not in allowed_presets
        ):
            err = f"Invalid describerPreset or describerQuality. Allowed values are {allowed_presets}."
            self.logger.error(err)
            raise ValueError(err)

        # Dict with node executable names
        self._nodes = {
            "cameraInit": "aliceVision_cameraInit",
            "featureExtraction": "aliceVision_featureExtraction",
            "imageMatching": "aliceVision_imageMatching",
            "featureMatching": "aliceVision_featureMatching",
            "structureFromMotion": "aliceVision_incrementalSfM",
            "sfmTransform": "aliceVision_utils_sfmTransform",
            "prepareDenseScene": "aliceVision_prepareDenseScene",
            "depthMapEstimation": "aliceVision_depthMapEstimation",
            "depthMapFiltering": "aliceVision_depthMapFiltering",
            "meshing": "aliceVision_meshing",
            "meshFiltering": "aliceVision_meshFiltering",
            "meshDecimate": "aliceVision_meshDecimate",
            "meshResampling": "aliceVision_meshResampling",
            "meshDenoising": "aliceVision_meshDenoising",
            "texturing": "aliceVision_texturing",
        }

    @property
    def inputSize(self):
        """
        Number of files in `input_dir`

        Returns
        -------
        int
            Number of files in `input_dir`

        """
        return len([i for i in Path(self.input_dir).iterdir() if i.is_file()])
        # return len(next(os.walk(self.input_dir))[2]) # CHECK: Performance

    @property
    def blockSize(self):
        """
        Number of images to process in each block

        Returns
        -------
        int
            `block_size` or number of images to process in each block

        """
        # NOTE: We can also maintain a dict with varying block sizes for different nodes
        # Or, different ratios with respect to the max block size
        if self.inputSize <= self.minBlockSize:
            return self.inputSize
        return self.inputSize // self.maxCores

    @property
    def numBlocks(self):
        """
        Number of blocks to process

        Returns
        -------
        int
            Number of blocks to process

        """
        return (
            self.inputSize // self.blockSize
        ) + 1  # NOTE: +1 to account for the remainder

    @classmethod
    def _check_state(cls):
        """
        Checks the state of the pipeline

        Raises
        ------
        RuntimeError
            If `cls.state["error"]` is True

        """
        cls = AliceVision
        if cls.state["error"]:
            msg = f"Skipping due to error in {cls.state['source']}. Check logs at {cls.state['log_file']}."
            cls.logger.error(msg)
            raise RuntimeError(msg)

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
            AliceVision.logger.info(
                f"Starting command execution. Log file: {log_path}."
            )
            logger.info(f"Command:\n{cmd}")
            output = check_output(cmd, shell=True, stderr=STDOUT)
            logger.info(f"Output:\n{output.decode().strip()}")
            AliceVision.logger.info(
                f"Finished command execution. Log file: {log_path}."
            )
        except CalledProcessError as error:
            logger.error(f"\n{error.output.decode().strip()}")
            AliceVision.logger.error(
                f"Error in command execution for {logger.name}. Check log file: {log_path}."
            )
            AliceVision.state = {
                "error": True,
                "source": logger.name,
                "log_file": log_path,
            }
            error.add_note(
                f"{logger.name} failed. Check log file: {log_path}."
            )  # NOTE: Won't appear in VSCode's Jupyter Notebook (March 2023)
            raise error

    def _parallelRunner(self, cmd: str, log_path: Path, caller: str):
        """
        Run a command in parallel and log the output

        Parameters
        ----------
        cmd : str
            Command to run
        log_file : Path
            Path to the log file
        caller : str
            Name of the caller function

        """
        block_size = self.blockSize

        cmds, logs = [], []
        for i in range(self.numBlocks):
            logs.extend([log_path / f"{caller}.{i}.log"])
            cmds.append(f"{cmd} --rangeStart {i * block_size} --rangeSize {block_size}")
        cmds_and_logs = list(zip(cmds, logs))

        with Pool(self.cpu_count) as pool:
            pool.starmap(self._serialRunner, cmds_and_logs)  # NOTE: Blocking call

    def _timeoutRunner(self, cmd: Iterable, timeout: int):
        """
        Run a command with a timeout
        NOTE: Mainly for cameraInit

        Parameters
        ----------
        cmd : Iterable
            Command to run
        timeout : int
            Timeout in seconds

        Returns
        -------
        str
            Output of the command

        Raises
        ------
        CalledProcessError
            If the command fails
        TimeoutExpired
            If the command times out

        """
        process = Popen(cmd, stdout=PIPE, stderr=PIPE)

        try:
            stdout, stderr = process.communicate(timeout=timeout)
            if process.returncode == 0:
                return stdout.decode("utf-8").strip()
            else:
                raise CalledProcessError(
                    process.returncode, cmd, stderr.decode("utf-8")
                )
        except TimeoutExpired:
            process.kill()
            raise

    def _check_input(
        self,
        cmd: str,
        inp: Union[str, Path],
        alt: Optional[Union[str, Path]] = None,
        arg: Optional[str] = "-i",
    ):
        """
        Check if the input file exists and updates the command with the input file path.

        Parameters
        ----------
        cmd : str
            Command (string) to assemble
        inp : Union[str, Path]
            Input .sfm file
        alt : Optional[Union[str, Path]]
            Alternative input file
            Default: None
        arg : Optional[str]
            Argument to use for the input file
            Default: '-i'

        Raises
        ------
        FileNotFoundError
            If `inp` is not found at provided paths.

        """
        inp = inp or alt
        if not Path(inp).exists():
            raise FileNotFoundError(f"Input file not found at {inp}.")
        cmd += f" {arg} {inp}"

        return cmd, inp

    def _add_desc_presets(self, cmd: str, addAll: bool = False):
        """
        Add describer presets to the command.

        Parameters
        ----------
        cmd : str
            Command (string) to assemble
        addAll : bool
            Add all presets or just the describer types
            Default: False

        """
        describerPreset, describerQuality, describerTypes = self.descPresets.values()

        cmd += f" -d {describerTypes}"
        if addAll:
            cmd += f" -p {describerPreset}"
            cmd += f" --describerQuality {describerQuality}"

        return cmd

    def _check_value(
        self, cmd: str, name: str, value: Union[int, float], rng: Iterable
    ):
        """
        Checks if value is in provided range.

        Parameters
        ----------
        cmd : str
            Command (string) to assemble
        name : str
            Name of the parameter
        value : Union[int, float]
            Value to check.
        rng : Iterable
            Range for checking with start and
            end values as first and second elements.

        Raises
        ------
        ValueError
            If `value` is not within `rng`.

        """
        if not (rng[0] < value < rng[1]):
            raise ValueError(f"Value must be between {rng[0]} and {rng[1]}.")
        cmd += f" --{name} {value}"

        return cmd

    def cameraInit(self):
        """
        Initializes the camera intrinsics from a set of images and
        save the result in a sfmData file, named `cameraInit.sfm`.
        Uses `aliceVision_cameraInit`.
        Default location for the output file is
        `self.cache_dir`/01_cameraInit/cameraInit.sfm.

        NOTE: Ensure that the `cameraSensors.db` file is in the same
        folder as the `aliceVision_cameraInit` file (`exec_path`).

        """
        self._check_state()

        node_path = self.exec_path / self._nodes["cameraInit"]
        out_path = self.cache_dir / "01_cameraInit/"
        out_path.mkdir(parents=True, exist_ok=True)
        out_file = out_path / "cameraInit.sfm"

        # Assemble command
        cmd = f"{node_path} --imageFolder {self.input_dir} -o {out_file} --verboseLevel {self.verboseLevel}"

        # Check & add other inputs
        sensorDatabase = self.exec_path / "cameraSensors.db"
        cmd, sensorDatabase = self._check_input(cmd, sensorDatabase, arg="-s")

        # Set up logger
        log_file = out_path / "cameraInit.log"
        logger = Logger(log_file.stem, log_file.parent)
        log_path = Path(log_file).resolve()

        # Run command
        ret_int = CAMERAINIT_RETRY_INTERVAL
        self.logger.info(f"Starting command execution. Log file: {log_path}.")
        logger.info(f"Command:\n{cmd}")
        cmd = cmd.split()  # Split command into list of strings
        for i in range(CAMERAINIT_MAX_RETRIES):
            try:
                output = self._timeoutRunner(
                    cmd, CAMERAINIT_MAX_RUNTIME
                )  # NOTE: Blocking call

                # Check if cameraInit.sfm is created
                if not out_file.exists():
                    logger.debug(
                        f"cameraInit did not create cameraInit.sfm. Retrying in {ret_int}s..."
                    )
                    sleep(ret_int)
                    continue

                logger.info(f"Output:\n{output}")
                self.logger.info(
                    f"Finished command execution after {i} retries. Log file: {log_path}."
                )
                break
            except CalledProcessError as error:
                # NOTE: Logging, but continuing execution till CAMERAINIT_MAX_RETRIES is reached
                logger.error(f"\n{error.output.strip()}")
                self.logger.error(
                    f"Error in command execution for {logger.name}. Check log file: {log_path}."
                )
            except TimeoutExpired:
                logger.debug(
                    f"cameraInit timed out {i + 1} time(s). Retrying in {ret_int}s..."
                )
                sleep(ret_int)
        else:
            logger.error(
                f"cameraInit did not finish after {CAMERAINIT_MAX_RETRIES} retries."
            )
            self.logger.error(
                f"cameraInit did not finish after {CAMERAINIT_MAX_RETRIES} retries."
            )
            self.logger.error(
                f"Error in command execution for {logger.name}. Check log file: {log_path}."
            )
            AliceVision.state = {
                "error": True,
                "source": logger.name,
                "log_file": log_path,
            }

    def featureExtraction(self, inputSfm: Optional[Union[str, Path]] = None):
        """
        Extracts features from a set of images using `aliceVision_featureExtraction`.
        Default location for the output file is `self.cache_dir`/02_featureExtraction/cameraInit.sfm.
        Defaults to 'dspsift' feature extraction on GPU (`forceCpuExtraction` is False).

        Parameters
        ----------
        inputSfm : Optional[Union[str, Path]]
            Optional, Path to the input sfm file
            Default: `self.cache_dir/01_CameraInit/cameraInit.sfm`

        """
        self._check_state()

        node_path = self.exec_path / self._nodes["featureExtraction"]
        out_path = self.cache_dir / "02_featureExtraction/"
        out_path.mkdir(parents=True, exist_ok=True)

        # Assemble command
        cmd = f"{node_path} -o {out_path} --verboseLevel {self.verboseLevel}"

        # Check & add input file
        cmd, inputSfm = self._check_input(
            cmd, inputSfm, self.cache_dir / "01_cameraInit/cameraInit.sfm"
        )

        # Add describer presets
        cmd = self._add_desc_presets(cmd, addAll=True)

        # Add other arguments
        cmd += f" --forceCpuExtraction 0 --maxThreads 0"  # NOTE: maxThreads 0 means "automatic"

        self._parallelRunner(cmd, out_path, "featureExtraction")

    def imageMatching(
        self,
        inputSfm: Optional[Union[str, Path]] = None,
        featuresFolders: Optional[Union[str, Path]] = None,
    ):
        """
        Matches features between images using `aliceVision_featureMatching`.

        Parameters
        ----------
        inputSfm : Optional[Union[str, Path]]
            Optional, Path to the input sfm file
            Default: `self.cache_dir/01_CameraInit/cameraInit.sfm`
        featuresFolders : Optional[Union[str, Path]]
            Optional, Path to the folder containing features
            Default: `self.cache_dir/02_featureExtraction/`

        """
        self._check_state()

        node_path = self.exec_path / self._nodes["imageMatching"]
        out_path = self.cache_dir / "03_imageMatching/"
        out_path.mkdir(parents=True, exist_ok=True)
        out_file = out_path / "imageMatches.txt"
        tree_path = self.exec_path / "vlfeat_K80L3.SIFT.tree"

        # Assemble command
        cmd = f"{node_path} -o {out_file} --verboseLevel {self.verboseLevel}"

        # Check & add input file
        cmd, inputSfm = self._check_input(
            cmd, inputSfm, self.cache_dir / "01_cameraInit/cameraInit.sfm"
        )

        # Check & add featuresFolders
        cmd, featuresFolders = self._check_input(
            cmd, featuresFolders, alt=self.cache_dir / "02_featureExtraction/", arg="-f"
        )

        # Check & add path to tree file
        cmd, tree_path = self._check_input(cmd, tree_path, arg="-t")

        log_file = out_path / "imageMatching.log"
        self._serialRunner(cmd, log_file)

    def featureMatching(
        self,
        inputSfm: Optional[Union[str, Path]] = None,
        featuresFolders: Optional[Union[str, Path]] = None,
        imagePairsList: Optional[Union[str, Path]] = None,
    ):
        """
        Matches features between images using `aliceVision_featureMatching`.

        Parameters
        ----------
        inputSfm : Optional[Union[str, Path]]
            Optional, Path to the input sfm file
            Default: `self.cache_dir/01_CameraInit/cameraInit.sfm`
        featuresFolders : Optional[Union[str, Path]]
            Optional, Path to the folder containing features
            Default: `self.cache_dir/02_featureExtraction/`
        imagePairsList : Optional[Union[str, Path]]
            Optional, Path to the file containing image pairs
            Default: `self.cache_dir/03_imageMatching/imageMatches.txt`

        """
        self._check_state()

        node_path = self.exec_path / self._nodes["featureMatching"]
        out_path = self.cache_dir / "04_featureMatching/"
        out_path.mkdir(parents=True, exist_ok=True)

        # Assemble command
        cmd = f"{node_path} -o {out_path} --verboseLevel {self.verboseLevel}"

        # Check & add input file
        cmd, inputSfm = self._check_input(
            cmd, inputSfm, self.cache_dir / "01_cameraInit/cameraInit.sfm"
        )

        # Check & add featuresFolders
        cmd, featuresFolders = self._check_input(
            cmd, featuresFolders, alt=self.cache_dir / "02_featureExtraction/", arg="-f"
        )

        # Check & add path to imagePairsList
        cmd, imagePairsList = self._check_input(
            cmd,
            imagePairsList,
            alt=self.cache_dir / "03_imageMatching/imageMatches.txt",
            arg="-l",
        )

        # Add describer types
        cmd = self._add_desc_presets(cmd)

        # Add other arguments
        cmd += f" --guidedMatching 1"  # NOTE: guidedMatching set to True

        self._parallelRunner(cmd, out_path, "featureMatching")

    def structureFromMotion(
        self,
        inputSfm: Optional[Union[str, Path]] = None,
        featuresFolders: Optional[Union[str, Path]] = None,
        matchesFolders: Optional[Union[str, Path]] = None,
    ):
        """
        Computes structure from motion using `aliceVision_incrementalSfM`.

        Parameters
        ----------
        inputSfm : Optional[Union[str, Path]]
            Optional, Path to the input sfm file
            Default: `self.cache_dir/01_CameraInit/cameraInit.sfm`
        featuresFolders : Optional[Union[str, Path]]
            Optional, Path to the folder containing features
            Default: `self.cache_dir/02_featureExtraction/`
        matchesFolders : Optional[Union[str, Path]]
            Optional, Path to the folder containing matches
            Default: `self.cache_dir/04_featureMatching/`

        """
        self._check_state()

        node_path = self.exec_path / self._nodes["structureFromMotion"]
        out_path = self.cache_dir / "05_structureFromMotion/"
        out_path.mkdir(parents=True, exist_ok=True)
        out_file = out_path / "sfm.abc"

        # Assemble command
        cmd = f"{node_path} --verboseLevel {self.verboseLevel}"

        # Add output paths
        outputViewsAndPoses = out_path / "cameras.sfm"
        cmd += f" -o {out_file} --outputViewsAndPoses {outputViewsAndPoses} --extraInfoFolder {out_path}"

        # Check & add input file
        cmd, inputSfm = self._check_input(
            cmd, inputSfm, self.cache_dir / "01_cameraInit/cameraInit.sfm"
        )

        # Check & add featuresFolders
        cmd, featuresFolders = self._check_input(
            cmd, featuresFolders, alt=self.cache_dir / "02_featureExtraction/", arg="-f"
        )

        # Check & add matchesFolders
        cmd, matchesFolders = self._check_input(
            cmd, matchesFolders, alt=self.cache_dir / "04_featureMatching/", arg="-m"
        )

        # Add describer presets
        cmd = self._add_desc_presets(cmd)

        log_file = out_path / "structureFromMotion.log"
        self._serialRunner(cmd, log_file)

    def sfmTransform(
        self,
        inputSfm: Optional[Union[str, Path]] = None,
        outputViewsAndPoses: Optional[Union[str, Path]] = None,
        transformation: Optional[str] = None
        # applyRotation: Optional[bool], # Maybe in general API
        # applyScale: Optional[bool],
        # applyTranslation: Optional[bool]
    ):
        """
        Transforms the SfM data using `aliceVision_utils_sfmTransform`.

        Parameters
        ----------
        inputSfm : Optional[Union[str, Path]]
            Optional, Path to the input sfm file
            Default: `self.cache_dir/05_structureFromMotion/sfm.abc`
        outputViewsAndPoses : Optional[Union[str, Path]]
            Optional, Path to the output sfm file
            Default: `self.cache_dir/05_structureFromMotion/cameras.sfm`
        transformation : Optional[str]
            Optional, Name of the image to align with (no extensions)
            If not provided, method = `auto_from_cameras` (AliceVision default)
            will be used, else, the method will be set to `from_single_camera`
            using the provided image.
            Default: `None`

        """
        self._check_state()

        node_path = self.exec_path / self._nodes["sfmTransform"]
        out_path = self.cache_dir / "06_sfmTransform/"
        out_path.mkdir(parents=True, exist_ok=True)
        out_file = out_path / "sfmTrans.abc"

        # Assemble command
        cmd = f"{node_path} -o {out_file} --verboseLevel {self.verboseLevel}"

        # Check & add input file
        cmd, inputSfm = self._check_input(
            cmd, inputSfm, self.cache_dir / "05_structureFromMotion/sfm.abc"
        )

        # Add method and transformation
        if transformation:
            method = "from_single_camera"  # NOTE: default is `auto_from_cameras`
            cmd += f" --method {method} --transformation {transformation}"

        # Check & add outputViewsAndPoses
        cmd, outputViewsAndPoses = self._check_input(
            cmd,
            outputViewsAndPoses,
            alt=self.cache_dir / "05_structureFromMotion/cameras.sfm",
            arg="--outputViewsAndPoses",
        )

        # Add transformation presets
        applyScale, applyRotation, applyTranslation = [0, 1, 1]
        cmd += f" --applyScale {applyScale} --applyRotation {applyRotation} --applyTranslation {applyTranslation}"

        log_file = out_path / "sfmTransform.log"
        self._serialRunner(cmd, log_file)

    def sfmRotate(
        self,
        inputSfm: Optional[Union[str, Path]] = None,
        outputViewsAndPoses: Optional[Union[str, Path]] = None,
        rotation: Optional[Iterable[float]] = [0.0, 0.0, 0.0],
        orientMesh: Optional[bool] = False,
    ):
        """
        Rotates the SfM data using `aliceVision_utils_sfmTransform`.

        Parameters
        ----------
        inputSfm : Optional[Union[str, Path]]
            Optional, Path to the input sfm file
            Default: `self.cache_dir/05_structureFromMotion/sfm.abc`
        outputViewsAndPoses : Optional[Union[str, Path]]
            Optional, Path to the output sfm file
            Default: `self.cache_dir/05_structureFromMotion/cameras.sfm`
        rotation : Optional[Iterable]
            Optional, Euler rotation around X, Y, Z in degrees.
            Default: `[0., 0., 0.]`
        orientMesh : Optional[bool]
            Optional, If True, the mesh will be rotated as well.

        Raises
        ------
        ValueError
            If rotation is not between 0 and 360 degrees.

        """
        self._check_state()

        node_path = self.exec_path / self._nodes["sfmTransform"]
        out_path = self.cache_dir / "07_sfmRotate/"
        out_path.mkdir(parents=True, exist_ok=True)
        out_file = out_path / "sfmRota.abc"

        # Assemble command
        cmd = f"{node_path} -o {out_file} --verboseLevel {self.verboseLevel}"

        # Check & add input file
        cmd, inputSfm = self._check_input(
            cmd, inputSfm, self.cache_dir / "06_sfmTransform/sfmTrans.abc"
        )

        # Add method and transformation
        for r in rotation:
            if not (0 <= r <= 360):
                raise ValueError(
                    f"Rotation must be between 0 and 360 degrees, got {r}."
                )
        rx, ry, rz = rotation
        transformation = f"0,0,0,{rx},{ry},{rz},1" if orientMesh else f"0,0,0,0,0,0,1"
        cmd += f" --method manual --manualTransform {transformation}"

        # Check & add outputViewsAndPoses
        cmd, outputViewsAndPoses = self._check_input(
            cmd,
            outputViewsAndPoses,
            alt=self.cache_dir / "05_structureFromMotion/cameras.sfm",
            arg="--outputViewsAndPoses",
        )

        log_file = out_path / "sfmRotate.log"
        self._serialRunner(cmd, log_file)

    def prepareDenseScene(
        self,
        inputSfm: Optional[Union[str, Path]] = None,
        # imagesFolders: Optional[Union[str, Path]], # Maybe in general API
    ):
        """
        Prepares a dense scene using `aliceVision_prepareDenseScene`.

        Parameters
        ----------
        inputSfm : Optional[Union[str, Path]]
            Optional, Path to the input sfm file
            Default: `self.cache_dir/07_sfmRotate/sfmRota.abc`

        """
        self._check_state()

        node_path = self.exec_path / self._nodes["prepareDenseScene"]
        out_path = self.cache_dir / "08_prepareDenseScene/"
        out_path.mkdir(parents=True, exist_ok=True)

        # Assemble command
        cmd = f"{node_path} -o {out_path} --verboseLevel {self.verboseLevel}"

        # Check & add input file
        cmd, inputSfm = self._check_input(
            cmd, inputSfm, self.cache_dir / "07_sfmRotate/sfmRota.abc"
        )

        self._parallelRunner(cmd, out_path, "prepareDenseScene")

    def depthMapEstimation(
        self,
        inputSfm: Optional[Union[str, Path]] = None,
        imagesFolders: Optional[Union[str, Path]] = None,
    ):
        """
        Computes depth maps using `aliceVision_depthMapEstimation`.

        Parameters
        ----------
        inputSfm : Optional[Union[str, Path]]
            Optional, Path to the input sfm file
            Default: `self.cache_dir/07_sfmRotate/sfmRota.abc`
        imagesFolders : Optional[Union[str, Path]]
            Optional, Path to the images folder
            Default: `self.cache_dir/08_prepareDenseScene/`

        """
        self._check_state()

        node_path = self.exec_path / self._nodes["depthMapEstimation"]
        out_path = self.cache_dir / "09_depthMapEstimation/"
        out_path.mkdir(parents=True, exist_ok=True)

        # Assemble command
        cmd = f"{node_path} -o {out_path} --verboseLevel {self.verboseLevel}"

        # Check & add input file
        cmd, inputSfm = self._check_input(
            cmd, inputSfm, self.cache_dir / "07_sfmRotate/sfmRota.abc"
        )

        # Check & add imagesFolders
        cmd, imagesFolders = self._check_input(
            cmd,
            imagesFolders,
            alt=self.cache_dir / "08_prepareDenseScene/",
            arg="--imagesFolder",
        )

        # Add other arguments
        cmd += f" --nbGPUs 0"  # NOTE: `nbGPUs` = 0 means, use all available GPUs

        self._parallelRunner(cmd, out_path, "depthMapEstimation")

    def depthMapFiltering(
        self,
        inputSfm: Optional[Union[str, Path]] = None,
        depthMapsFolder: Optional[Union[str, Path]] = None,
    ):
        """
        Filters depth maps using `aliceVision_depthMapFiltering`.

        Parameters
        ----------
        inputSfm : Optional[Union[str, Path]]
            Optional, Path to the input sfm file
            Default: `self.cache_dir/07_sfmRotate/sfmRota.abc`
        depthMapsFolder : Optional[Union[str, Path]]
            Optional, Path to the depth maps folder
            Default: `self.cache_dir/09_depthMapEstimation/`

        """
        self._check_state()

        node_path = self.exec_path / self._nodes["depthMapFiltering"]
        out_path = self.cache_dir / "10_depthMapFiltering/"
        out_path.mkdir(parents=True, exist_ok=True)

        # Assemble command
        cmd = f"{node_path} -o {out_path} --verboseLevel {self.verboseLevel}"

        # Check & add input file
        cmd, inputSfm = self._check_input(
            cmd, inputSfm, self.cache_dir / "07_sfmRotate/sfmRota.abc"
        )

        # Check & add depthMapsFolder
        cmd, depthMapsFolder = self._check_input(
            cmd,
            depthMapsFolder,
            alt=self.cache_dir / "09_depthMapEstimation/",
            arg="--depthMapsFolder",
        )

        self._parallelRunner(cmd, out_path, "depthMapFiltering")

    def meshing(
        self,
        inputSfm: Optional[Union[str, Path]] = None,
        depthMapsFolder: Optional[Union[str, Path]] = None,
        estimateSpaceMinObservationAngle: Optional[int] = 30,
        # Maybe in general API
        # maxPoints: Optional[int] = 0,
        # maxInputPoints: Optional[int] = 0,
        # seed: Optional[int] = 0
    ):
        """
        Computes a mesh using `aliceVision_meshing`.

        Parameters
        ----------
        inputSfm : Optional[Union[str, Path]]
            Optional, Path to the input sfm file
            Default: `self.cache_dir/07_sfmRotate/sfmRota.abc`
        depthMapsFolder : Optional[Union[str, Path]]
            Optional, Path to the depth maps folder
            Default: `self.cache_dir/10_depthMapFiltering/`
        estimateSpaceMinObservationAngle : Optional[int]
            Optional, Minimum angle of observation for a point to be used in the meshing.
            Default: 30 # NOTE: 10 in MR

        """
        self._check_state()

        node_path = self.exec_path / self._nodes["meshing"]
        out_path = self.cache_dir / "11_meshing/"
        out_path.mkdir(parents=True, exist_ok=True)
        out_dense_sfm = out_path / "densePointCloud.abc"
        out_mesh = out_path / "rawMesh.obj"

        # Assemble command
        cmd = f"{node_path} --output {out_dense_sfm} --outputMesh {out_mesh} --verboseLevel {self.verboseLevel}"

        # Check & add input file
        cmd, inputSfm = self._check_input(
            cmd, inputSfm, self.cache_dir / "07_sfmRotate/sfmRota.abc"
        )

        # Check & add depthMapsFolder
        cmd, depthMapsFolder = self._check_input(
            cmd,
            depthMapsFolder,
            alt=self.cache_dir / "10_depthMapFiltering/",
            arg="--depthMapsFolder",
        )

        # Check & add estimateSpaceMinObservationAngle
        cmd = self._check_value(
            cmd,
            "estimateSpaceMinObservationAngle",
            estimateSpaceMinObservationAngle,
            [0, 120],
        )

        log_file = out_path / "meshing.log"
        self._serialRunner(cmd, log_file)

    def meshFiltering(
        self,
        inputMesh: Optional[Union[str, Path]] = None,
        keepLargestMeshOnly: Optional[Union[int, bool]] = 1,
    ):
        """
        Filters a mesh using `aliceVision_meshFiltering`.

        Parameters
        ----------
        inputMesh : Optional[Union[str, Path]]
            Optional, Path to the input mesh file
            Default: `self.cache_dir/11_meshing/rawMesh.obj`
        keepLargestMeshOnly : Optional[Union[int, bool]]
            Optional, Keep only the largest mesh.
            Default: 1

        """
        self._check_state()

        node_path = self.exec_path / self._nodes["meshFiltering"]
        out_path = self.cache_dir / "12_meshFiltering/"
        out_path.mkdir(parents=True, exist_ok=True)
        out_mesh = out_path / "filteredMesh.obj"

        keepLargestMeshOnly = int(keepLargestMeshOnly)

        # Assemble command
        cmd = f"{node_path} -o {out_mesh} --verboseLevel {self.verboseLevel}"

        # Check & add input file
        cmd, inputMesh = self._check_input(
            cmd, inputMesh, self.cache_dir / "11_meshing/rawMesh.obj"
        )

        # Add other arguments
        cmd += f" --keepLargestMeshOnly {keepLargestMeshOnly}"

        log_file = out_path / "meshFiltering.log"
        self._serialRunner(cmd, log_file)

    def meshDecimate(
        self,
        inputMesh: Optional[Union[str, Path]] = None,
        simplificationFactor: Optional[float] = 0.3,
        # Maybe in general API
        # nbVertices: Optional[int] = 0,
        # maxVertices: Optional[int] = 0,
    ):
        """
        Decimates a mesh using `aliceVision_meshDecimate`.

        Parameters
        ----------
        inputMesh : Optional[Union[str, Path]]
            Optional, Path to the input mesh file
            Default: `self.cache_dir/12_meshFiltering/filteredMesh.obj`
        simplificationFactor : Optional[float]
            Optional, Simplification factor.
            Default: 0.3

        """
        self._check_state()

        node_path = self.exec_path / self._nodes["meshDecimate"]
        out_path = self.cache_dir / "13_meshDecimate/"
        out_path.mkdir(parents=True, exist_ok=True)
        out_mesh = out_path / "decimatedMesh.obj"

        # Assemble command
        cmd = f"{node_path} -o {out_mesh} --verboseLevel {self.verboseLevel}"

        # Check & add input file
        cmd, inputMesh = self._check_input(
            cmd, inputMesh, self.cache_dir / "12_meshFiltering/filteredMesh.obj"
        )

        # Check & add simplificationFactor
        cmd = self._check_value(
            cmd, "simplificationFactor", simplificationFactor, [0, 1]
        )

        log_file = out_path / "meshDecimate.log"
        self._serialRunner(cmd, log_file)

    # FIXME: Resampled meshes cannot be textured using the dense.abc generated by `Meshing`. Leaves untextured areas.
    # def meshResampling(
    #     self,
    #     useDecimated: Optional[bool] = False,
    #     inputMesh: Optional[Union[str, Path]] = None,
    #     simplificationFactor: Optional[float] = 0.3,
    #     # Maybe in general API
    #     # nbVertices: Optional[int] = 0,
    #     # maxVertices: Optional[int] = 0,
    # ):
    #     """
    #     Resample a mesh.

    #     Parameters
    #     ----------
    #     useDecimated : Optional[bool]
    #         Optional, *Also* use the decimated mesh.
    #         Default: False
    #     inputMesh : Optional[Union[str, Path]]
    #         Optional, Path to the input mesh file
    #         Default: `self.cache_dir/12_meshFiltering/filteredMesh.obj`
    #     simplificationFactor : Optional[float]
    #         Optional, Simplification factor.
    #         Default: 0.3

    #     Raises
    #     ------
    #     ValueError
    #         If `simplificationFactor` is not between 0 and 1.

    #     """
    #     node_path = self.exec_path / self._nodes["meshResampling"]
    #     out_path = self.cache_dir / "14_meshResampling/"

    #     # Assemble command
    #     cmd = f"{node_path}  --verboseLevel {self.verboseLevel}"

    #     # Check & add simplificationFactor
    #     cmd = self._check_value(cmd, "simplificationFactor", simplificationFactor, [0, 1])

    #     # Resample decimated mesh
    #     if useDecimated:
    #         out_mesh = out_path / "resampledDecimatedMesh.obj"
    #         cmd += f" -o {out_mesh}"

    #         # Check & add input file
    #         cmd, inputMesh = self._check_input(cmd, inputMesh, self.cache_dir / "13_meshDecimate/decimatedMesh.obj")

    #         self._serialRunner(cmd)

    #     # Resample raw mesh
    #     out_mesh = out_path / "resampledRawMesh.obj"
    #     cmd += f" -o {out_mesh}"

    #     # Check & add input file
    #     cmd, inputMesh = self._check_input(cmd, inputMesh, self.cache_dir / "12_meshFiltering/filteredMesh.obj")

    #     self._serialRunner(cmd)

    def meshDenoising(
        self,
        useDecimated: Optional[bool] = True,
        inputMesh: Optional[Union[str, Path]] = None,
        lmd: Optional[float] = 2.0,
        eta: Optional[float] = 1.5,
    ):
        """
        Denoises a mesh using `aliceVision_meshDenoising`.
        NOTE: Larger values of lambda or eta result in smoother meshes

        Parameters
        ----------
        useDecimated : Optional[bool]
            Optional, *Also* use the decimated mesh.
            Default: True
        inputMesh : Optional[Union[str, Path]]
            Optional, Path to the input mesh file
            Default: `self.cache_dir/12_meshFiltering/filteredMesh.obj`
        lmd : Optional[float]
            Optional, Smoothing parameter.
            Default: 2.
        eta : Optional[float]
            Optional, Smoothing parameter.
            Default: 1.5

        """
        self._check_state()

        node_path = self.exec_path / self._nodes["meshDenoising"]
        out_path = self.cache_dir / "14_meshDenoising/"
        out_path.mkdir(parents=True, exist_ok=True)

        # Assemble commands
        cmds = []
        cmd = f"{node_path} --verboseLevel {self.verboseLevel}"

        # Check & add denoising arguments
        cmd = self._check_value(cmd, "lambda", lmd, [0, 10])
        cmd = self._check_value(cmd, "eta", eta, [0, 20])

        # Denoise decimated mesh
        if useDecimated:
            out_mesh = out_path / "denoisedDecimatedMesh.obj"
            local_cmd = cmd + f" -o {out_mesh}"
            local_cmd, _ = self._check_input(
                local_cmd,
                inputMesh,
                self.cache_dir / "13_meshDecimate/decimatedMesh.obj",
            )
            cmds.append(local_cmd)

        # Denoise raw mesh
        out_mesh = out_path / "denoisedRawMesh.obj"
        cmd += f" -o {out_mesh}"
        cmd, _ = self._check_input(
            cmd, inputMesh, self.cache_dir / "12_meshFiltering/filteredMesh.obj"
        )
        cmds.append(cmd)

        logs = [
            out_path / f"meshDenoising.deci.log",
            out_path / f"meshDenoising.raw.log",
        ]
        cmds_and_logs = list(zip(cmds, logs))
        with Pool(2) as pool:
            pool.starmap(self._serialRunner, cmds_and_logs)

    def texturing(
        self,
        useDecimated: Optional[bool] = True,
        denoise: Optional[bool] = False,
        inputDenseSfm: Optional[Union[str, Path]] = None,
        inputMesh: Optional[Union[str, Path]] = None,
        unwrapMethod: Optional[str] = "basic",
        textureSide: Optional[int] = 2048,  # NOTE: 2048 for decimated & * 2 for raw
        # Maybe in general API
        # imagesFolder: Optional[Union[str, Path]] = None, # NOTE: Replaces `inputDenseSfm` if used
        # downscale: Optional[int] = 1,
        # fillHoles: Optional[Union[int, bool]] = 1
    ):
        """
        Textures a mesh using `aliceVision_texturing`.

        Parameters
        ----------
        useDecimated : Optional[bool]
            Optional, *Also* use the decimated mesh for texturing.
            Default: True
        denoise : Optional[bool]
            Optional, Whether to denoise the Mesh
            NOTE: Denoising can result in extreme smoothing at the moment
            Default: False
        inputDenseSfm : Optional[Union[str, Path]]
            Optional, Path to the input dense SfM file
            Default: `self.cache_dir/11_meshing/densePointCloud.abc`
        inputMesh : Optional[Union[str, Path]]
            Optional, Path to the input mesh file
            Default: `self.cache_dir/12_meshFiltering/filteredMesh.obj`
        unwrapMethod : Optional[str]
            Optional, Unwrap method.
            Default: `basic`. NOTE: Try `basic` if stuck with `LSCM` (for < 600K faces).
        textureSide : Optional[int]
            Optional, Texture resolution.
            Default: 2048 for decimated and 4096 for raw (scaled by 2)

        """
        self._check_state()

        node_path = self.exec_path / self._nodes["texturing"]
        out_path = self.cache_dir / "15_texturing/"
        out_path.mkdir(parents=True, exist_ok=True)

        # Assemble commands
        cmds = []
        cmd = f"{node_path} --verboseLevel {self.verboseLevel}"

        # Add other arguments
        cmd += f" --unwrapMethod {unwrapMethod}"

        # Check & add input files
        cmd, inputDenseSfm = self._check_input(
            cmd, inputDenseSfm, self.cache_dir / "11_meshing/densePointCloud.abc"
        )

        # (Denoised) Decimated mesh
        if useDecimated:
            out_mesh_dir = out_path / "texturedDecimatedMesh/"
            out_mesh_dir.mkdir(parents=True, exist_ok=True)
            local_cmd = (
                cmd + f" -o {out_mesh_dir} --textureSide {textureSide}"
            )  # 2048 for decimated

            alt = self.cache_dir / "13_meshDecimate/decimatedMesh.obj"
            if denoise:
                alt = self.cache_dir / "14_meshDenoising/denoisedDecimatedMesh.obj"

            local_cmd, _ = self._check_input(
                local_cmd, inputMesh, alt=alt, arg="--inputMesh"
            )
            cmds.append(local_cmd)

        # (Denoised) Raw mesh
        out_mesh_dir = out_path / "texturedRawMesh/"
        out_mesh_dir.mkdir(parents=True, exist_ok=True)
        cmd += f" -o {out_mesh_dir} --textureSide {textureSide * 2}"  # 4096 or twice decimated for raw

        alt = self.cache_dir / "12_meshFiltering/filteredMesh.obj"
        if denoise:
            alt = self.cache_dir / "14_meshDenoising/denoisedRawMesh.obj"

        cmd, _ = self._check_input(cmd, inputMesh, alt=alt, arg="--inputMesh")
        cmds.append(cmd)

        logs = [out_path / f"texturing.deci.log", out_path / f"texturing.raw.log"]
        cmds_and_logs = list(zip(cmds, logs))
        with Pool(2) as pool:
            pool.starmap(self._serialRunner, cmds_and_logs)

    def _run_all(
        self,
        denoise: Optional[bool] = False,
        center_image: Optional[str] = None,
        rotation: Optional[Iterable[float]] = [0.0, 0.0, 0.0],
        orientMesh: Optional[bool] = False,
        estimateSpaceMinObservationAngle: Optional[int] = 30,
    ):
        """
        Runs all the steps with default parameters.

        Parameters
        ----------
        denoise : Optional[bool]
            Optional, Whether to denoise the Mesh
            Default: False
            NOTE: Denoising can result in extreme smoothing at the moment
        center_image : Optional[str]
            Optional, Path to the image to be used as the center of the reconstruction.
            Default: None
        rotation : Optional[Iterable[float]]
            Optional, Euler rotation around X, Y, Z in degrees.
            Default: `[0., 0., 0.]`
        orientMesh : Optional[bool]
            Optional, Whether to orient the mesh using sfmRotate or use <model-viewer>.
        estimateSpaceMinObservationAngle : Optional[int]
            Optional, Minimum angle between two observations to consider them as a valid pair.

        """
        self.cameraInit()
        self.featureExtraction()
        self.imageMatching()
        self.featureMatching()
        self.structureFromMotion()
        self.sfmTransform(transformation=center_image)
        self.sfmRotate(rotation=rotation, orientMesh=orientMesh)
        self.prepareDenseScene()
        self.depthMapEstimation()
        self.depthMapFiltering()
        self.meshing(estimateSpaceMinObservationAngle=estimateSpaceMinObservationAngle)
        self.meshFiltering()
        self.meshDecimate()
        # self.meshResampling()
        if denoise:
            self.meshDenoising()
        self.texturing(denoise=denoise)
