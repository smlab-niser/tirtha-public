"""
Shared pytest fixtures for AliceVision tests.
"""

import pytest
import tempfile
import shutil
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tirtha.alicevision import AliceVision


class MockLogger:
    """Mock logger for testing."""

    def __init__(self):
        self.messages = {"info": [], "error": [], "debug": [], "warning": []}

    def info(self, msg):
        self.messages["info"].append(msg)

    def error(self, msg):
        self.messages["error"].append(msg)

    def debug(self, msg):
        self.messages["debug"].append(msg)

    def warning(self, msg):
        self.messages["warning"].append(msg)


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    return MockLogger()


@pytest.fixture
def sample_images_dir():
    """Provide path to actual sample images from AlagumGanesh."""
    images_path = Path("/home/faiz/TirthaProject/tirtha-public/AlagumGanesh/images/use")
    if images_path.exists() and any(images_path.iterdir()):
        return images_path
    return None


@pytest.fixture
def real_exec_path():
    """Provide path to actual AliceVision executables in bin21."""
    exec_path = Path(__file__).parent.parent.parent / "bin21"
    if exec_path.exists():
        return exec_path
    else:
        pytest.skip(f"AliceVision executables not found at {exec_path}")


@pytest.fixture
def temp_exec_path():
    """Create a temporary directory with mock AliceVision executables."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create mock executables
        executables = [
            "aliceVision_cameraInit",
            "aliceVision_featureExtraction",
            "aliceVision_imageMatching",
            "aliceVision_featureMatching",
            "aliceVision_incrementalSfM",
            "aliceVision_utils_sfmTransform",
            "aliceVision_prepareDenseScene",
            "aliceVision_depthMapEstimation",
            "aliceVision_depthMapFiltering",
            "aliceVision_meshing",
            "aliceVision_meshFiltering",
            "aliceVision_meshDecimate",
            "aliceVision_meshResampling",
            "aliceVision_meshDenoising",
            "aliceVision_texturing",
        ]

        for exe in executables:
            (temp_path / exe).touch()
            (temp_path / exe).chmod(0o755)

        # Create required database files
        (temp_path / "cameraSensors.db").touch()
        (temp_path / "vlfeat_K80L3.SIFT.tree").touch()

        yield temp_path


@pytest.fixture
def temp_input_dir(sample_images_dir):
    """Create a temporary input directory with sample images."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Copy some sample images if they exist
        if sample_images_dir and sample_images_dir.exists():
            for i, img_file in enumerate(sample_images_dir.glob("*.jpeg")):
                if i < 20:  # Copy only first 20 images for testing
                    shutil.copy2(img_file, temp_path)
        else:
            # Create dummy image files for testing
            for i in range(20):
                dummy_img = temp_path / f"test_image_{i}.jpg"
                dummy_img.write_bytes(b"fake_image_data")

        yield temp_path


@pytest.fixture
def empty_input_dir():
    """Create an empty input directory for testing error conditions."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def basic_alicevision(real_exec_path, temp_input_dir, temp_cache_dir, mock_logger):
    """Create a basic AliceVision instance for testing with real executables."""
    return AliceVision(
        exec_path=real_exec_path,
        input_dir=temp_input_dir,
        cache_dir=temp_cache_dir,
        logger=mock_logger,
    )


@pytest.fixture
def basic_alicevision_mock_exec(
    temp_exec_path, temp_input_dir, temp_cache_dir, mock_logger
):
    """Create a basic AliceVision instance for testing with mock executables."""
    return AliceVision(
        exec_path=temp_exec_path,
        input_dir=temp_input_dir,
        cache_dir=temp_cache_dir,
        logger=mock_logger,
    )


@pytest.fixture
def alicevision_with_custom_presets(
    real_exec_path, temp_input_dir, temp_cache_dir, mock_logger
):
    """Create AliceVision instance with custom descriptor presets."""
    desc_presets = {"Preset": "high", "Quality": "ultra", "Types": "sift,akaze"}
    return AliceVision(
        exec_path=real_exec_path,
        input_dir=temp_input_dir,
        cache_dir=temp_cache_dir,
        logger=mock_logger,
        descPresets=desc_presets,
    )


@pytest.fixture(params=["invalid_preset", "invalid_quality", "invalid_descriptor"])
def invalid_type(request):
    """Invalid descriptor preset types for testing."""
    invalid_configs = {
        "invalid_preset": {"Preset": "invalid", "Quality": "normal", "Types": "sift"},
        "invalid_quality": {"Preset": "normal", "Quality": "invalid", "Types": "sift"},
        "invalid_descriptor": {
            "Preset": "normal",
            "Quality": "normal",
            "Types": "invalid_type",
        },
    }
    return invalid_configs[request.param]


@pytest.fixture
def large_sample_images_dir(sample_images_dir):
    """Create a large sample images directory for performance testing."""
    return sample_images_dir  # For now, just return the regular sample directory


@pytest.fixture
def pipeline_stage_prerequisites(basic_alicevision):
    """Create all common pipeline prerequisite files."""
    # Stage 1: Camera Init
    camera_init_path = basic_alicevision.cache_dir / "01_cameraInit"
    camera_init_path.mkdir(parents=True, exist_ok=True)
    (camera_init_path / "cameraInit.sfm").touch()

    # Stage 2: Feature Extraction
    feature_path = basic_alicevision.cache_dir / "02_featureExtraction"
    feature_path.mkdir(parents=True, exist_ok=True)

    # Stage 3: Image Matching
    image_matching_path = basic_alicevision.cache_dir / "03_imageMatching"
    image_matching_path.mkdir(parents=True, exist_ok=True)
    (image_matching_path / "imageMatches.txt").touch()

    # Stage 5: Structure from Motion
    sfm_path = basic_alicevision.cache_dir / "05_structureFromMotion"
    sfm_path.mkdir(parents=True, exist_ok=True)
    (sfm_path / "sfm.abc").touch()

    # Stage 7: Prepare Dense Scene (alternative paths)
    dense_scene_path = basic_alicevision.cache_dir / "07_prepareDenseScene"
    dense_scene_path.mkdir(parents=True, exist_ok=True)
    (dense_scene_path / "densePointCloud.abc").touch()

    # Stage 11: Meshing
    meshing_path = basic_alicevision.cache_dir / "11_meshing"
    meshing_path.mkdir(parents=True, exist_ok=True)
    (meshing_path / "densePointCloud.abc").touch()
    (meshing_path / "mesh.obj").touch()
    (meshing_path / "rawMesh.obj").touch()

    # Stage 12: Mesh Filtering
    mesh_filtering_path = basic_alicevision.cache_dir / "12_meshFiltering"
    mesh_filtering_path.mkdir(parents=True, exist_ok=True)
    (mesh_filtering_path / "filteredMesh.obj").touch()

    # Stage 13: Mesh Decimate
    mesh_decimate_path = basic_alicevision.cache_dir / "13_meshDecimate"
    mesh_decimate_path.mkdir(parents=True, exist_ok=True)
    (mesh_decimate_path / "decimatedMesh.obj").touch()

    return basic_alicevision


@pytest.fixture
def minimal_image_set(sample_images_dir, tmp_path):
    """Create a minimal set of 20 images for testing."""
    minimal_dir = tmp_path / "test_images"
    minimal_dir.mkdir()

    # Copy only 20 images for pipeline testing
    sample_images = list(sample_images_dir.glob("*.jpeg"))[:20]

    for img in sample_images:
        shutil.copy2(img, minimal_dir / img.name)

    return minimal_dir
