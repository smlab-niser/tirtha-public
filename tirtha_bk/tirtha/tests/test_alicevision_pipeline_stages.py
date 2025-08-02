"""
Pipeline stages tests for AliceVision class (stages 1-15).
"""

import pytest
from unittest.mock import Mock, patch
from subprocess import TimeoutExpired

from tirtha.alicevision import AliceVision


class TestAliceVisionPipelineStages:
    """Test AliceVision pipeline stages."""

    # Stage 1: Camera Initialization Tests
    @patch("tirtha.alicevision.AliceVision._timeoutRunner")
    def test_01_camera_init_success(self, mock_timeout_runner, basic_alicevision):
        """Test successful camera initialization."""

        def create_output(*args, **kwargs):
            # Create camera init output
            output_path = basic_alicevision.cache_dir / "01_cameraInit"
            output_path.mkdir(parents=True, exist_ok=True)
            (output_path / "cameraInit.sfm").touch()
            return "Success"

        mock_timeout_runner.side_effect = create_output
        basic_alicevision.cameraInit()

        # Check that output directory and file were created
        assert (basic_alicevision.cache_dir / "01_cameraInit").exists()
        assert (basic_alicevision.cache_dir / "01_cameraInit/cameraInit.sfm").exists()
        mock_timeout_runner.assert_called()

    @patch("tirtha.alicevision.AliceVision._timeoutRunner")
    def test_01_camera_init_timeout_retry(self, mock_timeout_runner, basic_alicevision):
        """Test camera initialization with timeout and retry."""
        call_count = 0

        def side_effect_func(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:  # First 2 calls timeout
                raise TimeoutExpired("cmd", 2)
            else:  # 3rd call succeeds and creates output
                output_path = basic_alicevision.cache_dir / "01_cameraInit"
                output_path.mkdir(parents=True, exist_ok=True)
                (output_path / "cameraInit.sfm").touch()
                return "Success"

        mock_timeout_runner.side_effect = side_effect_func
        basic_alicevision.cameraInit()

        # Should have been called at least 3 times due to retries
        assert mock_timeout_runner.call_count >= 3
        assert (basic_alicevision.cache_dir / "01_cameraInit/cameraInit.sfm").exists()

    @patch("tirtha.alicevision.AliceVision._timeoutRunner")
    def test_01_camera_init_max_retries_exceeded(
        self, mock_timeout_runner, basic_alicevision
    ):
        """Test camera initialization when max retries are exceeded."""
        # Reset state to ensure clean test
        AliceVision.state = {"error": False, "source": None, "log_file": None}

        # Always timeout
        mock_timeout_runner.side_effect = TimeoutExpired("cmd", 2)

        # Should set error state but not raise exception
        basic_alicevision.cameraInit()

        # Verify error state was set
        assert AliceVision.state["error"]
        assert AliceVision.state["source"] is not None

    # Stage 2: Feature Extraction Tests
    @patch("tirtha.alicevision.AliceVision._parallelRunner")
    def test_02_feature_extraction_with_camera_init_output(
        self, mock_parallel_runner, basic_alicevision
    ):
        """Test feature extraction using camera init output."""
        # Create camera init output (simulating stage 1 completion)
        camera_init_path = basic_alicevision.cache_dir / "01_cameraInit"
        camera_init_path.mkdir(parents=True, exist_ok=True)
        (camera_init_path / "cameraInit.sfm").touch()

        def create_output(*args, **kwargs):
            # Create feature extraction output
            feature_path = basic_alicevision.cache_dir / "02_featureExtraction"
            feature_path.mkdir(parents=True, exist_ok=True)
            return None

        mock_parallel_runner.side_effect = create_output
        basic_alicevision.featureExtraction()

        mock_parallel_runner.assert_called_once()
        assert (basic_alicevision.cache_dir / "02_featureExtraction").exists()

    @patch("tirtha.alicevision.AliceVision._parallelRunner")
    def test_02_feature_extraction_custom_input(
        self, mock_parallel_runner, basic_alicevision, temp_cache_dir
    ):
        """Test feature extraction with custom input file."""
        custom_input = temp_cache_dir / "custom_input.sfm"
        custom_input.touch()

        mock_parallel_runner.return_value = None
        basic_alicevision.featureExtraction(inputSfm=custom_input)

        mock_parallel_runner.assert_called_once()

    # Stage 3: Image Matching Tests
    @patch("tirtha.alicevision.AliceVision._serialRunner")
    def test_03_image_matching_with_feature_output(
        self, mock_serial_runner, basic_alicevision
    ):
        """Test image matching using feature extraction output."""
        # Create previous stage outputs (simulating stages 1-2 completion)
        camera_init_path = basic_alicevision.cache_dir / "01_cameraInit"
        camera_init_path.mkdir(parents=True, exist_ok=True)
        (camera_init_path / "cameraInit.sfm").touch()

        feature_path = basic_alicevision.cache_dir / "02_featureExtraction"
        feature_path.mkdir(parents=True, exist_ok=True)

        def create_output(*args, **kwargs):
            # Create image matching output
            match_path = basic_alicevision.cache_dir / "03_imageMatching"
            match_path.mkdir(parents=True, exist_ok=True)
            (match_path / "imageMatches.txt").touch()

        mock_serial_runner.side_effect = create_output
        basic_alicevision.imageMatching()

        mock_serial_runner.assert_called_once()
        assert (basic_alicevision.cache_dir / "03_imageMatching").exists()

    @patch("tirtha.alicevision.AliceVision._serialRunner")
    def test_03_image_matching_custom_inputs(
        self, mock_serial_runner, basic_alicevision, temp_cache_dir
    ):
        """Test image matching with custom inputs."""
        custom_sfm = temp_cache_dir / "custom.sfm"
        custom_features = temp_cache_dir / "custom_features"
        custom_sfm.touch()
        custom_features.mkdir(exist_ok=True)

        mock_serial_runner.return_value = None
        basic_alicevision.imageMatching(
            inputSfm=custom_sfm, featuresFolders=custom_features
        )

        mock_serial_runner.assert_called_once()

    # Stage 4: Feature Matching Tests
    @patch("tirtha.alicevision.AliceVision._parallelRunner")
    def test_04_feature_matching_with_image_matching_output(
        self, mock_parallel_runner, basic_alicevision
    ):
        """Test feature matching using image matching output."""
        # Create previous stage outputs
        camera_init_path = basic_alicevision.cache_dir / "01_cameraInit"
        camera_init_path.mkdir(parents=True, exist_ok=True)
        (camera_init_path / "cameraInit.sfm").touch()

        feature_path = basic_alicevision.cache_dir / "02_featureExtraction"
        feature_path.mkdir(parents=True, exist_ok=True)

        match_path = basic_alicevision.cache_dir / "03_imageMatching"
        match_path.mkdir(parents=True, exist_ok=True)
        (match_path / "imageMatches.txt").touch()

        def create_output(*args, **kwargs):
            # Create feature matching output
            feat_match_path = basic_alicevision.cache_dir / "04_featureMatching"
            feat_match_path.mkdir(parents=True, exist_ok=True)

        mock_parallel_runner.side_effect = create_output
        basic_alicevision.featureMatching()

        mock_parallel_runner.assert_called_once()
        assert (basic_alicevision.cache_dir / "04_featureMatching").exists()

    # Stage 5: Structure from Motion Tests
    @patch("tirtha.alicevision.AliceVision._serialRunner")
    def test_05_structure_from_motion_with_feature_matching_output(
        self, mock_serial_runner, basic_alicevision
    ):
        """Test structure from motion using feature matching output."""
        # Create previous stage outputs
        camera_init_path = basic_alicevision.cache_dir / "01_cameraInit"
        camera_init_path.mkdir(parents=True, exist_ok=True)
        (camera_init_path / "cameraInit.sfm").touch()

        feature_path = basic_alicevision.cache_dir / "02_featureExtraction"
        feature_path.mkdir(parents=True, exist_ok=True)

        feat_match_path = basic_alicevision.cache_dir / "04_featureMatching"
        feat_match_path.mkdir(parents=True, exist_ok=True)

        def create_output(*args, **kwargs):
            # Create SfM output
            sfm_path = basic_alicevision.cache_dir / "05_structureFromMotion"
            sfm_path.mkdir(parents=True, exist_ok=True)
            (sfm_path / "sfm.abc").touch()
            (sfm_path / "cameras.sfm").touch()

        mock_serial_runner.side_effect = create_output
        basic_alicevision.structureFromMotion()

        mock_serial_runner.assert_called_once()
        assert (basic_alicevision.cache_dir / "05_structureFromMotion").exists()

    # Stage 6: SfM Transform Tests
    @patch("tirtha.alicevision.AliceVision._serialRunner")
    def test_06_sfm_transform_with_sfm_output(
        self, mock_serial_runner, basic_alicevision
    ):
        """Test SfM transform using structure from motion output."""
        # Create SfM output
        sfm_path = basic_alicevision.cache_dir / "05_structureFromMotion"
        sfm_path.mkdir(parents=True, exist_ok=True)
        (sfm_path / "sfm.abc").touch()
        (sfm_path / "cameras.sfm").touch()

        def create_output(*args, **kwargs):
            # Create SfM transform output
            transform_path = basic_alicevision.cache_dir / "06_sfmTransform"
            transform_path.mkdir(parents=True, exist_ok=True)
            (transform_path / "sfmTrans.abc").touch()

        mock_serial_runner.side_effect = create_output
        basic_alicevision.sfmTransform()

        mock_serial_runner.assert_called_once()
        assert (basic_alicevision.cache_dir / "06_sfmTransform").exists()

    @patch("tirtha.alicevision.AliceVision._serialRunner")
    def test_06_sfm_transform_with_transformation_parameter(
        self, mock_serial_runner, basic_alicevision
    ):
        """Test SfM transform with transformation parameter."""
        # Create SfM output
        sfm_path = basic_alicevision.cache_dir / "05_structureFromMotion"
        sfm_path.mkdir(parents=True, exist_ok=True)
        (sfm_path / "sfm.abc").touch()
        (sfm_path / "cameras.sfm").touch()

        mock_serial_runner.return_value = None
        basic_alicevision.sfmTransform(transformation="test_image")

        mock_serial_runner.assert_called_once()

    # Stage 7: SfM Rotate Tests
    @patch("tirtha.alicevision.AliceVision._serialRunner")
    def test_07_sfm_rotate_with_transform_output(
        self, mock_serial_runner, basic_alicevision
    ):
        """Test SfM rotate using transform output."""
        # Create transform output
        transform_path = basic_alicevision.cache_dir / "06_sfmTransform"
        transform_path.mkdir(parents=True, exist_ok=True)
        (transform_path / "sfmTrans.abc").touch()

        sfm_path = basic_alicevision.cache_dir / "05_structureFromMotion"
        sfm_path.mkdir(parents=True, exist_ok=True)
        (sfm_path / "cameras.sfm").touch()

        def create_output(*args, **kwargs):
            # Create SfM rotate output
            rotate_path = basic_alicevision.cache_dir / "07_sfmRotate"
            rotate_path.mkdir(parents=True, exist_ok=True)
            (rotate_path / "sfmRota.abc").touch()

        mock_serial_runner.side_effect = create_output
        basic_alicevision.sfmRotate()

        mock_serial_runner.assert_called_once()
        assert (basic_alicevision.cache_dir / "07_sfmRotate").exists()

    @pytest.mark.parametrize(
        "rotation", [[0, 0, 0], [90, 0, 0], [0, 90, 0], [0, 0, 90]]
    )
    @patch("tirtha.alicevision.AliceVision._serialRunner")
    def test_07_sfm_rotate_with_different_rotations(
        self, mock_serial_runner, basic_alicevision, rotation
    ):
        """Test SfM rotate with different rotation values."""
        # Create transform output
        transform_path = basic_alicevision.cache_dir / "06_sfmTransform"
        transform_path.mkdir(parents=True, exist_ok=True)
        (transform_path / "sfmTrans.abc").touch()

        sfm_path = basic_alicevision.cache_dir / "05_structureFromMotion"
        sfm_path.mkdir(parents=True, exist_ok=True)
        (sfm_path / "cameras.sfm").touch()

        mock_serial_runner.return_value = None
        basic_alicevision.sfmRotate(rotation=rotation)

        mock_serial_runner.assert_called_once()

    @patch("tirtha.alicevision.AliceVision._serialRunner")
    def test_07_sfm_rotate_invalid_rotation_raises_error(
        self, mock_serial_runner, basic_alicevision
    ):
        """Test SfM rotate with invalid rotation values."""
        # Create required input files first
        sfm_transform_path = basic_alicevision.cache_dir / "06_sfmTransform"
        sfm_transform_path.mkdir(parents=True, exist_ok=True)
        (sfm_transform_path / "sfmTrans.abc").touch()

        with pytest.raises(ValueError, match="Rotation must be between 0 and 360"):
            basic_alicevision.sfmRotate(rotation=[400, 0, 0])

    # Stage 8: Prepare Dense Scene Tests
    @patch("tirtha.alicevision.AliceVision._parallelRunner")
    def test_08_prepare_dense_scene_with_rotate_output(
        self, mock_parallel_runner, basic_alicevision
    ):
        """Test prepare dense scene using rotate output."""
        # Create rotate output
        rotate_path = basic_alicevision.cache_dir / "07_sfmRotate"
        rotate_path.mkdir(parents=True, exist_ok=True)
        (rotate_path / "sfmRota.abc").touch()

        def create_output(*args, **kwargs):
            # Create prepare dense scene output
            dense_path = basic_alicevision.cache_dir / "08_prepareDenseScene"
            dense_path.mkdir(parents=True, exist_ok=True)

        mock_parallel_runner.side_effect = create_output
        basic_alicevision.prepareDenseScene()

        mock_parallel_runner.assert_called_once()
        assert (basic_alicevision.cache_dir / "08_prepareDenseScene").exists()

    # Stage 9: Depth Map Estimation Tests
    @patch("tirtha.alicevision.AliceVision._parallelRunner")
    def test_09_depth_map_estimation_with_dense_scene_output(
        self, mock_parallel_runner, basic_alicevision
    ):
        """Test depth map estimation using dense scene output."""
        # Create dense scene output
        dense_path = basic_alicevision.cache_dir / "08_prepareDenseScene"
        dense_path.mkdir(parents=True, exist_ok=True)

        rotate_path = basic_alicevision.cache_dir / "07_sfmRotate"
        rotate_path.mkdir(parents=True, exist_ok=True)
        (rotate_path / "sfmRota.abc").touch()

        def create_output(*args, **kwargs):
            # Create depth map estimation output
            depth_path = basic_alicevision.cache_dir / "09_depthMapEstimation"
            depth_path.mkdir(parents=True, exist_ok=True)

        mock_parallel_runner.side_effect = create_output
        basic_alicevision.depthMapEstimation()

        mock_parallel_runner.assert_called_once()
        assert (basic_alicevision.cache_dir / "09_depthMapEstimation").exists()

    # Stage 10: Depth Map Filtering Tests
    @patch("tirtha.alicevision.AliceVision._parallelRunner")
    def test_10_depth_map_filtering_with_estimation_output(
        self, mock_parallel_runner, basic_alicevision
    ):
        """Test depth map filtering using estimation output."""
        # Create depth map estimation output
        depth_path = basic_alicevision.cache_dir / "09_depthMapEstimation"
        depth_path.mkdir(parents=True, exist_ok=True)

        rotate_path = basic_alicevision.cache_dir / "07_sfmRotate"
        rotate_path.mkdir(parents=True, exist_ok=True)
        (rotate_path / "sfmRota.abc").touch()

        def create_output(*args, **kwargs):
            # Create depth map filtering output
            filter_path = basic_alicevision.cache_dir / "10_depthMapFiltering"
            filter_path.mkdir(parents=True, exist_ok=True)

        mock_parallel_runner.side_effect = create_output
        basic_alicevision.depthMapFiltering()

        mock_parallel_runner.assert_called_once()
        assert (basic_alicevision.cache_dir / "10_depthMapFiltering").exists()

    # Stage 11: Meshing Tests
    @patch("tirtha.alicevision.AliceVision._serialRunner")
    def test_11_meshing_with_filtering_output(
        self, mock_serial_runner, basic_alicevision
    ):
        """Test meshing using filtering output."""
        # Create depth map filtering output
        filter_path = basic_alicevision.cache_dir / "10_depthMapFiltering"
        filter_path.mkdir(parents=True, exist_ok=True)

        rotate_path = basic_alicevision.cache_dir / "07_sfmRotate"
        rotate_path.mkdir(parents=True, exist_ok=True)
        (rotate_path / "sfmRota.abc").touch()

        def create_output(*args, **kwargs):
            # Create meshing output
            mesh_path = basic_alicevision.cache_dir / "11_meshing"
            mesh_path.mkdir(parents=True, exist_ok=True)
            (mesh_path / "rawMesh.obj").touch()
            (mesh_path / "densePointCloud.abc").touch()

        mock_serial_runner.side_effect = create_output
        basic_alicevision.meshing()

        mock_serial_runner.assert_called_once()
        assert (basic_alicevision.cache_dir / "11_meshing").exists()

    @pytest.mark.parametrize("angle", [10, 30, 60, 90])
    @patch("tirtha.alicevision.AliceVision._serialRunner")
    def test_11_meshing_with_different_observation_angles(
        self, mock_serial_runner, basic_alicevision, angle
    ):
        """Test meshing with different observation angles."""
        # Create required inputs
        filter_path = basic_alicevision.cache_dir / "10_depthMapFiltering"
        filter_path.mkdir(parents=True, exist_ok=True)

        rotate_path = basic_alicevision.cache_dir / "07_sfmRotate"
        rotate_path.mkdir(parents=True, exist_ok=True)
        (rotate_path / "sfmRota.abc").touch()

        mock_serial_runner.return_value = None
        basic_alicevision.meshing(estimateSpaceMinObservationAngle=angle)

        mock_serial_runner.assert_called_once()

    # Stage 12: Mesh Filtering Tests
    @patch("tirtha.alicevision.AliceVision._serialRunner")
    def test_12_mesh_filtering_with_meshing_output(
        self, mock_serial_runner, basic_alicevision
    ):
        """Test mesh filtering using meshing output."""
        # Create meshing output
        mesh_path = basic_alicevision.cache_dir / "11_meshing"
        mesh_path.mkdir(parents=True, exist_ok=True)
        (mesh_path / "rawMesh.obj").touch()

        def create_output(*args, **kwargs):
            # Create mesh filtering output
            filter_path = basic_alicevision.cache_dir / "12_meshFiltering"
            filter_path.mkdir(parents=True, exist_ok=True)
            (filter_path / "filteredMesh.obj").touch()

        mock_serial_runner.side_effect = create_output
        basic_alicevision.meshFiltering()

        mock_serial_runner.assert_called_once()
        assert (basic_alicevision.cache_dir / "12_meshFiltering").exists()

    # Stage 13: Mesh Decimate Tests
    @patch("tirtha.alicevision.AliceVision._serialRunner")
    def test_13_mesh_decimate_with_filtering_output(
        self, mock_serial_runner, basic_alicevision
    ):
        """Test mesh decimate using filtering output."""
        # Create mesh filtering output
        filter_path = basic_alicevision.cache_dir / "12_meshFiltering"
        filter_path.mkdir(parents=True, exist_ok=True)
        (filter_path / "filteredMesh.obj").touch()

        def create_output(*args, **kwargs):
            # Create mesh decimate output
            decimate_path = basic_alicevision.cache_dir / "13_meshDecimate"
            decimate_path.mkdir(parents=True, exist_ok=True)
            (decimate_path / "decimatedMesh.obj").touch()

        mock_serial_runner.side_effect = create_output
        basic_alicevision.meshDecimate()

        mock_serial_runner.assert_called_once()
        assert (basic_alicevision.cache_dir / "13_meshDecimate").exists()

    @pytest.mark.parametrize("factor", [0.1, 0.3, 0.5, 0.8])
    @patch("tirtha.alicevision.AliceVision._serialRunner")
    def test_13_mesh_decimate_with_different_simplification_factors(
        self, mock_serial_runner, basic_alicevision, factor
    ):
        """Test mesh decimate with different simplification factors."""
        # Create mesh filtering output
        filter_path = basic_alicevision.cache_dir / "12_meshFiltering"
        filter_path.mkdir(parents=True, exist_ok=True)
        (filter_path / "filteredMesh.obj").touch()

        mock_serial_runner.return_value = None
        basic_alicevision.meshDecimate(simplificationFactor=factor)

        mock_serial_runner.assert_called_once()

    # Stage 14: Mesh Denoising Tests
    @patch("tirtha.alicevision.Pool")
    @patch("tirtha.alicevision.AliceVision._serialRunner")
    def test_14_mesh_denoising_with_decimate_output(
        self, mock_serial_runner, mock_pool, basic_alicevision
    ):
        """Test mesh denoising using decimate output."""
        # Create mesh decimate output
        decimate_path = basic_alicevision.cache_dir / "13_meshDecimate"
        decimate_path.mkdir(parents=True, exist_ok=True)
        (decimate_path / "decimatedMesh.obj").touch()

        filter_path = basic_alicevision.cache_dir / "12_meshFiltering"
        filter_path.mkdir(parents=True, exist_ok=True)
        (filter_path / "filteredMesh.obj").touch()

        # Mock the pool context manager
        mock_pool_instance = Mock()
        mock_pool.return_value.__enter__ = Mock(return_value=mock_pool_instance)
        mock_pool.return_value.__exit__ = Mock(return_value=None)

        def create_output(*args, **kwargs):
            # Create mesh denoising output
            denoise_path = basic_alicevision.cache_dir / "14_meshDenoising"
            denoise_path.mkdir(parents=True, exist_ok=True)
            (denoise_path / "denoisedDecimatedMesh.obj").touch()
            (denoise_path / "denoisedRawMesh.obj").touch()

        mock_pool_instance.starmap.side_effect = create_output

        basic_alicevision.meshDenoising()

        mock_pool.assert_called_once_with(2)
        assert (basic_alicevision.cache_dir / "14_meshDenoising").exists()

    @pytest.mark.parametrize("lmd,eta", [(1.0, 1.0), (2.0, 1.5), (3.0, 2.0)])
    @patch("tirtha.alicevision.Pool")
    @patch("tirtha.alicevision.AliceVision._serialRunner")
    def test_14_mesh_denoising_with_different_parameters(
        self, mock_serial_runner, mock_pool, basic_alicevision, lmd, eta
    ):
        """Test mesh denoising with different lambda and eta parameters."""
        # Create required inputs
        decimate_path = basic_alicevision.cache_dir / "13_meshDecimate"
        decimate_path.mkdir(parents=True, exist_ok=True)
        (decimate_path / "decimatedMesh.obj").touch()

        filter_path = basic_alicevision.cache_dir / "12_meshFiltering"
        filter_path.mkdir(parents=True, exist_ok=True)
        (filter_path / "filteredMesh.obj").touch()

        # Mock the pool
        mock_pool_instance = Mock()
        mock_pool.return_value.__enter__ = Mock(return_value=mock_pool_instance)
        mock_pool.return_value.__exit__ = Mock(return_value=None)

        basic_alicevision.meshDenoising(lmd=lmd, eta=eta)

        mock_pool.assert_called_once_with(2)

    # Stage 15: Texturing Tests
    @patch("tirtha.alicevision.Pool")
    @patch("tirtha.alicevision.AliceVision._serialRunner")
    def test_15_texturing_with_mesh_outputs(
        self, mock_serial_runner, mock_pool, basic_alicevision
    ):
        """Test texturing using mesh outputs."""
        # Create mesh outputs
        mesh_path = basic_alicevision.cache_dir / "11_meshing"
        mesh_path.mkdir(parents=True, exist_ok=True)
        (mesh_path / "densePointCloud.abc").touch()

        decimate_path = basic_alicevision.cache_dir / "13_meshDecimate"
        decimate_path.mkdir(parents=True, exist_ok=True)
        (decimate_path / "decimatedMesh.obj").touch()

        filter_path = basic_alicevision.cache_dir / "12_meshFiltering"
        filter_path.mkdir(parents=True, exist_ok=True)
        (filter_path / "filteredMesh.obj").touch()

        # Mock the pool
        mock_pool_instance = Mock()
        mock_pool.return_value.__enter__ = Mock(return_value=mock_pool_instance)
        mock_pool.return_value.__exit__ = Mock(return_value=None)

        def create_output(*args, **kwargs):
            # Create texturing output
            texture_path = basic_alicevision.cache_dir / "15_texturing"
            texture_path.mkdir(parents=True, exist_ok=True)
            (texture_path / "texturedDecimatedMesh").mkdir(exist_ok=True)
            (texture_path / "texturedRawMesh").mkdir(exist_ok=True)

        mock_pool_instance.starmap.side_effect = create_output

        basic_alicevision.texturing()

        mock_pool.assert_called_once_with(2)
        assert (basic_alicevision.cache_dir / "15_texturing").exists()

    @pytest.mark.parametrize("unwrap_method", ["basic", "LSCM"])
    @patch("tirtha.alicevision.Pool")
    def test_15_texturing_with_different_unwrap_methods(
        self, mock_pool, basic_alicevision, unwrap_method
    ):
        """Test texturing with different unwrap methods."""
        # Create required inputs
        mesh_path = basic_alicevision.cache_dir / "11_meshing"
        mesh_path.mkdir(parents=True, exist_ok=True)
        (mesh_path / "densePointCloud.abc").touch()

        decimate_path = basic_alicevision.cache_dir / "13_meshDecimate"
        decimate_path.mkdir(parents=True, exist_ok=True)
        (decimate_path / "decimatedMesh.obj").touch()

        filter_path = basic_alicevision.cache_dir / "12_meshFiltering"
        filter_path.mkdir(parents=True, exist_ok=True)
        (filter_path / "filteredMesh.obj").touch()

        # Mock the pool
        mock_pool_instance = Mock()
        mock_pool.return_value.__enter__ = Mock(return_value=mock_pool_instance)
        mock_pool.return_value.__exit__ = Mock(return_value=None)

        basic_alicevision.texturing(unwrapMethod=unwrap_method)

        mock_pool.assert_called_once_with(2)

    @pytest.mark.parametrize("texture_side", [1024, 2048, 4096])
    @patch("tirtha.alicevision.Pool")
    def test_15_texturing_with_different_texture_sizes(
        self, mock_pool, basic_alicevision, texture_side
    ):
        """Test texturing with different texture sizes."""
        # Create required inputs
        mesh_path = basic_alicevision.cache_dir / "11_meshing"
        mesh_path.mkdir(parents=True, exist_ok=True)
        (mesh_path / "densePointCloud.abc").touch()

        decimate_path = basic_alicevision.cache_dir / "13_meshDecimate"
        decimate_path.mkdir(parents=True, exist_ok=True)
        (decimate_path / "decimatedMesh.obj").touch()

        filter_path = basic_alicevision.cache_dir / "12_meshFiltering"
        filter_path.mkdir(parents=True, exist_ok=True)
        (filter_path / "filteredMesh.obj").touch()

        # Mock the pool
        mock_pool_instance = Mock()
        mock_pool.return_value.__enter__ = Mock(return_value=mock_pool_instance)
        mock_pool.return_value.__exit__ = Mock(return_value=None)

        basic_alicevision.texturing(textureSide=texture_side)

        mock_pool.assert_called_once_with(2)

    # ===============================
    # MIGRATED PIPELINE INTEGRATION TESTS
    # ===============================

    @pytest.mark.parametrize(
        "custom_sfm,custom_features",
        [(True, False), (False, True), (True, True), (False, False)],
    )
    @patch("tirtha.alicevision.AliceVision._serialRunner")
    def test_03_image_matching_various_inputs(
        self,
        mock_serial_runner,
        basic_alicevision,
        temp_cache_dir,
        custom_sfm,
        custom_features,
    ):
        """Test image matching with various input combinations."""
        # Create default files (previous stage outputs)
        camera_init_path = basic_alicevision.cache_dir / "01_cameraInit"
        camera_init_path.mkdir(parents=True, exist_ok=True)
        (camera_init_path / "cameraInit.sfm").touch()

        feature_path = basic_alicevision.cache_dir / "02_featureExtraction"
        feature_path.mkdir(parents=True, exist_ok=True)

        # Create custom files if needed
        input_sfm = None
        features_folder = None

        if custom_sfm:
            input_sfm = temp_cache_dir / "custom.sfm"
            input_sfm.touch()

        if custom_features:
            features_folder = temp_cache_dir / "custom_features"
            features_folder.mkdir(exist_ok=True)

        mock_serial_runner.return_value = None
        basic_alicevision.imageMatching(
            inputSfm=input_sfm, featuresFolders=features_folder
        )

        mock_serial_runner.assert_called_once()

    @patch("tirtha.alicevision.AliceVision._parallelRunner")
    def test_04_feature_matching_custom_inputs(
        self, mock_parallel_runner, basic_alicevision, temp_cache_dir
    ):
        """Test feature matching with custom input files."""
        # Create custom input files
        custom_sfm = temp_cache_dir / "custom.sfm"
        custom_sfm.touch()

        custom_features = temp_cache_dir / "custom_features"
        custom_features.mkdir(exist_ok=True)

        custom_pairs = temp_cache_dir / "custom_pairs.txt"
        custom_pairs.touch()

        mock_parallel_runner.return_value = None
        basic_alicevision.featureMatching(
            inputSfm=custom_sfm,
            featuresFolders=custom_features,
            imagePairsList=custom_pairs,
        )

        mock_parallel_runner.assert_called_once()

    @patch("tirtha.alicevision.AliceVision._serialRunner")
    def test_05_structure_from_motion_with_previous_outputs(
        self, mock_serial_runner, basic_alicevision
    ):
        """Test structure from motion using previous stage outputs."""
        # Create previous stage outputs (simulating stages 1-4 completion)
        camera_init_path = basic_alicevision.cache_dir / "01_cameraInit"
        camera_init_path.mkdir(parents=True, exist_ok=True)
        (camera_init_path / "cameraInit.sfm").touch()

        feature_path = basic_alicevision.cache_dir / "02_featureExtraction"
        feature_path.mkdir(parents=True, exist_ok=True)

        matching_path = basic_alicevision.cache_dir / "04_featureMatching"
        matching_path.mkdir(parents=True, exist_ok=True)

        def create_output(*args, **kwargs):
            # Create structure from motion outputs
            sfm_path = basic_alicevision.cache_dir / "05_structureFromMotion"
            sfm_path.mkdir(parents=True, exist_ok=True)
            (sfm_path / "sfm.abc").touch()
            (sfm_path / "cameras.sfm").touch()
            return None

        mock_serial_runner.side_effect = create_output
        basic_alicevision.structureFromMotion()

        mock_serial_runner.assert_called_once()
        assert (basic_alicevision.cache_dir / "05_structureFromMotion").exists()
        assert (basic_alicevision.cache_dir / "05_structureFromMotion/sfm.abc").exists()
        assert (
            basic_alicevision.cache_dir / "05_structureFromMotion/cameras.sfm"
        ).exists()

    @patch("tirtha.alicevision.AliceVision._serialRunner")
    def test_05_structure_from_motion_custom_inputs(
        self, mock_serial_runner, basic_alicevision, temp_cache_dir
    ):
        """Test structure from motion with custom input files."""
        # Create custom input files
        custom_sfm = temp_cache_dir / "custom.sfm"
        custom_sfm.touch()

        custom_features = temp_cache_dir / "custom_features"
        custom_features.mkdir(exist_ok=True)

        custom_matches = temp_cache_dir / "custom_matches"
        custom_matches.mkdir(exist_ok=True)

        mock_serial_runner.return_value = None
        basic_alicevision.structureFromMotion(
            inputSfm=custom_sfm,
            featuresFolders=custom_features,
            matchesFolders=custom_matches,
        )

        mock_serial_runner.assert_called_once()

    @patch("tirtha.alicevision.AliceVision._serialRunner")
    def test_06_sfm_transform_with_sfm_output(
        self, mock_serial_runner, basic_alicevision
    ):
        """Test SfM transform using structure from motion output."""
        # Create structure from motion output (simulating stage 5 completion)
        sfm_path = basic_alicevision.cache_dir / "05_structureFromMotion"
        sfm_path.mkdir(parents=True, exist_ok=True)
        (sfm_path / "sfm.abc").touch()
        (sfm_path / "cameras.sfm").touch()

        def create_output(*args, **kwargs):
            # Create SfM transform outputs
            transform_path = basic_alicevision.cache_dir / "06_sfmTransform"
            transform_path.mkdir(parents=True, exist_ok=True)
            (transform_path / "sfmTrans.abc").touch()
            return None

        mock_serial_runner.side_effect = create_output
        basic_alicevision.sfmTransform()

        mock_serial_runner.assert_called_once()
        assert (basic_alicevision.cache_dir / "06_sfmTransform").exists()
        assert (basic_alicevision.cache_dir / "06_sfmTransform/sfmTrans.abc").exists()

    @patch("tirtha.alicevision.AliceVision._serialRunner")
    def test_06_sfm_transform_with_transformation(
        self, mock_serial_runner, basic_alicevision
    ):
        """Test SfM transform with specific transformation parameter."""
        # Create required input files
        sfm_path = basic_alicevision.cache_dir / "05_structureFromMotion"
        sfm_path.mkdir(parents=True, exist_ok=True)
        (sfm_path / "sfm.abc").touch()
        (sfm_path / "cameras.sfm").touch()

        mock_serial_runner.return_value = None
        basic_alicevision.sfmTransform(transformation="center_image")

        mock_serial_runner.assert_called_once()
        # Verify transformation parameter is processed
        args, kwargs = mock_serial_runner.call_args
        cmd_str = args[0]
        assert "from_single_camera" in cmd_str
        assert "center_image" in cmd_str

    @pytest.mark.parametrize(
        "custom_input,custom_output",
        [(True, False), (False, True), (True, True), (False, False)],
    )
    @patch("tirtha.alicevision.AliceVision._serialRunner")
    def test_06_sfm_transform_various_inputs(
        self,
        mock_serial_runner,
        basic_alicevision,
        temp_cache_dir,
        custom_input,
        custom_output,
    ):
        """Test SfM transform with various input combinations."""
        # Create default files (previous stage outputs)
        sfm_path = basic_alicevision.cache_dir / "05_structureFromMotion"
        sfm_path.mkdir(parents=True, exist_ok=True)
        (sfm_path / "sfm.abc").touch()
        (sfm_path / "cameras.sfm").touch()

        # Create custom files if needed
        input_sfm = None
        output_poses = None

        if custom_input:
            input_sfm = temp_cache_dir / "custom_input.sfm"
            input_sfm.touch()

        if custom_output:
            output_poses = temp_cache_dir / "custom_output.sfm"
            output_poses.touch()

        mock_serial_runner.return_value = None
        basic_alicevision.sfmTransform(
            inputSfm=input_sfm, outputViewsAndPoses=output_poses
        )

        mock_serial_runner.assert_called_once()

    @patch("tirtha.alicevision.AliceVision._serialRunner")
    def test_07_sfm_rotate_with_transform_output(
        self, mock_serial_runner, basic_alicevision
    ):
        """Test SfM rotate using SfM transform output."""
        # Create SfM transform output (simulating stage 6 completion)
        sfm_transform_path = basic_alicevision.cache_dir / "06_sfmTransform"
        sfm_transform_path.mkdir(parents=True, exist_ok=True)
        (sfm_transform_path / "sfmTrans.abc").touch()

        # Also need cameras.sfm from structure from motion
        sfm_path = basic_alicevision.cache_dir / "05_structureFromMotion"
        sfm_path.mkdir(parents=True, exist_ok=True)
        (sfm_path / "cameras.sfm").touch()

        def create_output(*args, **kwargs):
            # Create SfM rotate outputs
            rotate_path = basic_alicevision.cache_dir / "07_sfmRotate"
            rotate_path.mkdir(parents=True, exist_ok=True)
            (rotate_path / "sfmRota.abc").touch()
            return None

        mock_serial_runner.side_effect = create_output
        basic_alicevision.sfmRotate()

        mock_serial_runner.assert_called_once()
        assert (basic_alicevision.cache_dir / "07_sfmRotate").exists()
        assert (basic_alicevision.cache_dir / "07_sfmRotate/sfmRota.abc").exists()

    @pytest.mark.parametrize(
        "rotation,orient_mesh",
        [
            ([90.0, 0.0, 0.0], False),
            ([0.0, 90.0, 0.0], True),
            ([45.0, 45.0, 90.0], False),
            ([180.0, 270.0, 0.0], True),
        ],
    )
    @patch("tirtha.alicevision.AliceVision._serialRunner")
    def test_07_sfm_rotate_custom_rotation(
        self, mock_serial_runner, basic_alicevision, rotation, orient_mesh
    ):
        """Test SfM rotate with custom rotation values."""
        # Create required input files (previous stage outputs)
        sfm_transform_path = basic_alicevision.cache_dir / "06_sfmTransform"
        sfm_transform_path.mkdir(parents=True, exist_ok=True)
        (sfm_transform_path / "sfmTrans.abc").touch()

        sfm_path = basic_alicevision.cache_dir / "05_structureFromMotion"
        sfm_path.mkdir(parents=True, exist_ok=True)
        (sfm_path / "cameras.sfm").touch()

        mock_serial_runner.return_value = None
        basic_alicevision.sfmRotate(rotation=rotation, orientMesh=orient_mesh)

        mock_serial_runner.assert_called_once()
        # Verify rotation values are in command only when orientMesh=True
        args, kwargs = mock_serial_runner.call_args
        cmd_str = args[0]
        if orient_mesh:
            assert str(rotation[0]) in cmd_str
            assert str(rotation[1]) in cmd_str
            assert str(rotation[2]) in cmd_str

    def test_07_sfm_rotate_invalid_rotation(self, basic_alicevision):
        """Test SfM rotate with invalid rotation values."""
        # Create required input files
        sfm_transform_path = basic_alicevision.cache_dir / "06_sfmTransform"
        sfm_transform_path.mkdir(parents=True, exist_ok=True)
        (sfm_transform_path / "sfmTrans.abc").touch()

        with pytest.raises(ValueError, match="Rotation must be between 0 and 360"):
            basic_alicevision.sfmRotate(rotation=[400.0, 0.0, 0.0])

    @patch("tirtha.alicevision.AliceVision._parallelRunner")
    def test_08_prepare_dense_scene_with_rotate_output(
        self, mock_parallel_runner, basic_alicevision
    ):
        """Test prepare dense scene using SfM rotate output."""
        # Create SfM rotate output (simulating stage 7 completion)
        sfm_rotate_path = basic_alicevision.cache_dir / "07_sfmRotate"
        sfm_rotate_path.mkdir(parents=True, exist_ok=True)
        (sfm_rotate_path / "sfmRota.abc").touch()

        def create_output(*args, **kwargs):
            # Create prepare dense scene outputs
            dense_path = basic_alicevision.cache_dir / "08_prepareDenseScene"
            dense_path.mkdir(parents=True, exist_ok=True)
            (dense_path / "densePointCloud.abc").touch()
            return None

        mock_parallel_runner.side_effect = create_output
        basic_alicevision.prepareDenseScene()

        mock_parallel_runner.assert_called_once()
        assert (basic_alicevision.cache_dir / "08_prepareDenseScene").exists()
        assert (
            basic_alicevision.cache_dir / "08_prepareDenseScene/densePointCloud.abc"
        ).exists()

    @patch("tirtha.alicevision.AliceVision._parallelRunner")
    def test_08_prepare_dense_scene_custom_input(
        self, mock_parallel_runner, basic_alicevision, temp_cache_dir
    ):
        """Test prepare dense scene with custom input file."""
        custom_input = temp_cache_dir / "custom_input.abc"
        custom_input.touch()

        mock_parallel_runner.return_value = None
        basic_alicevision.prepareDenseScene(inputSfm=custom_input)

        mock_parallel_runner.assert_called_once()

    @patch("tirtha.alicevision.AliceVision._parallelRunner")
    def test_09_depth_map_estimation_with_dense_scene_output(
        self, mock_parallel_runner, basic_alicevision
    ):
        """Test depth map estimation using prepare dense scene output."""
        # Create prepare dense scene output (simulating stage 8 completion)
        dense_scene_path = basic_alicevision.cache_dir / "08_prepareDenseScene"
        dense_scene_path.mkdir(parents=True, exist_ok=True)
        (dense_scene_path / "densePointCloud.abc").touch()

        # Also create the sfmRotate output that depthMapEstimation expects
        sfm_rotate_path = basic_alicevision.cache_dir / "07_sfmRotate"
        sfm_rotate_path.mkdir(parents=True, exist_ok=True)
        (sfm_rotate_path / "sfmRota.abc").touch()

        def create_output(*args, **kwargs):
            # Create depth map estimation outputs
            depth_path = basic_alicevision.cache_dir / "09_depthMapEstimation"
            depth_path.mkdir(parents=True, exist_ok=True)
            return None

        mock_parallel_runner.side_effect = create_output
        basic_alicevision.depthMapEstimation()

        mock_parallel_runner.assert_called_once()
        assert (basic_alicevision.cache_dir / "09_depthMapEstimation").exists()

    @patch("tirtha.alicevision.AliceVision._parallelRunner")
    def test_10_depth_map_filtering_with_estimation_output(
        self, mock_parallel_runner, basic_alicevision
    ):
        """Test depth map filtering using depth map estimation output."""
        # Create depth map estimation output (simulating stage 9 completion)
        depth_estimation_path = basic_alicevision.cache_dir / "09_depthMapEstimation"
        depth_estimation_path.mkdir(parents=True, exist_ok=True)

        # Also need the dense scene file
        dense_scene_path = basic_alicevision.cache_dir / "08_prepareDenseScene"
        dense_scene_path.mkdir(parents=True, exist_ok=True)
        (dense_scene_path / "densePointCloud.abc").touch()

        # Create the sfmRotate output that depthMapFiltering expects
        sfm_rotate_path = basic_alicevision.cache_dir / "07_sfmRotate"
        sfm_rotate_path.mkdir(parents=True, exist_ok=True)
        (sfm_rotate_path / "sfmRota.abc").touch()

        def create_output(*args, **kwargs):
            # Create depth map filtering outputs
            filtering_path = basic_alicevision.cache_dir / "10_depthMapFiltering"
            filtering_path.mkdir(parents=True, exist_ok=True)
            return None

        mock_parallel_runner.side_effect = create_output
        basic_alicevision.depthMapFiltering()

        mock_parallel_runner.assert_called_once()
        assert (basic_alicevision.cache_dir / "10_depthMapFiltering").exists()

    # Note: Removed test_10_depth_map_filtering_custom_params as the parameters don't exist in the actual API
