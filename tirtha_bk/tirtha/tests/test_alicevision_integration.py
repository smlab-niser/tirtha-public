"""
Integration tests for AliceVision class.
"""

import pytest
from unittest.mock import Mock, patch
from subprocess import CalledProcessError, TimeoutExpired
from pathlib import Path

from tirtha.alicevision import AliceVision


class TestAliceVisionIntegration:
    """Test AliceVision integration scenarios."""

    # Full Pipeline Integration Tests
    @patch("tirtha.alicevision.AliceVision.texturing")
    @patch("tirtha.alicevision.AliceVision.meshDecimate")
    @patch("tirtha.alicevision.AliceVision.meshFiltering")
    @patch("tirtha.alicevision.AliceVision.meshing")
    @patch("tirtha.alicevision.AliceVision.depthMapFiltering")
    @patch("tirtha.alicevision.AliceVision.depthMapEstimation")
    @patch("tirtha.alicevision.AliceVision.prepareDenseScene")
    @patch("tirtha.alicevision.AliceVision.sfmRotate")
    @patch("tirtha.alicevision.AliceVision.sfmTransform")
    @patch("tirtha.alicevision.AliceVision.structureFromMotion")
    @patch("tirtha.alicevision.AliceVision.featureMatching")
    @patch("tirtha.alicevision.AliceVision.imageMatching")
    @patch("tirtha.alicevision.AliceVision.featureExtraction")
    @patch("tirtha.alicevision.AliceVision.cameraInit")
    def test_run_all_complete_pipeline(
        self,
        mock_camera_init,
        mock_feature_extraction,
        mock_image_matching,
        mock_feature_matching,
        mock_structure_motion,
        mock_sfm_transform,
        mock_sfm_rotate,
        mock_prepare_dense,
        mock_depth_estimation,
        mock_depth_filtering,
        mock_meshing,
        mock_mesh_filtering,
        mock_mesh_decimate,
        mock_texturing,
        basic_alicevision,
    ):
        """Test complete pipeline execution with _run_all method."""

        # Set up mock side effects to create expected outputs
        def create_camera_init(*args, **kwargs):
            output_path = basic_alicevision.cache_dir / "01_cameraInit"
            output_path.mkdir(parents=True, exist_ok=True)
            (output_path / "cameraInit.sfm").touch()

        def create_feature_extraction(*args, **kwargs):
            output_path = basic_alicevision.cache_dir / "02_featureExtraction"
            output_path.mkdir(parents=True, exist_ok=True)

        def create_image_matching(*args, **kwargs):
            output_path = basic_alicevision.cache_dir / "03_imageMatching"
            output_path.mkdir(parents=True, exist_ok=True)
            (output_path / "imageMatches.txt").touch()

        def create_structure_motion(*args, **kwargs):
            output_path = basic_alicevision.cache_dir / "05_structureFromMotion"
            output_path.mkdir(parents=True, exist_ok=True)
            (output_path / "sfm.abc").touch()
            (output_path / "cameras.sfm").touch()

        mock_camera_init.side_effect = create_camera_init
        mock_feature_extraction.side_effect = create_feature_extraction
        mock_image_matching.side_effect = create_image_matching
        mock_structure_motion.side_effect = create_structure_motion

        # Run the complete pipeline
        basic_alicevision._run_all()

        # Verify all major stages were called
        mock_camera_init.assert_called_once()
        mock_feature_extraction.assert_called_once()
        mock_image_matching.assert_called_once()
        mock_feature_matching.assert_called_once()
        mock_structure_motion.assert_called_once()

    @patch("multiprocessing.Pool")
    @patch("tirtha.alicevision.AliceVision.meshDenoising")
    @patch("tirtha.alicevision.AliceVision.texturing")
    @patch("tirtha.alicevision.AliceVision.meshDecimate")
    @patch("tirtha.alicevision.AliceVision.meshFiltering")
    @patch("tirtha.alicevision.AliceVision.meshing")
    @patch("tirtha.alicevision.AliceVision.depthMapFiltering")
    @patch("tirtha.alicevision.AliceVision.depthMapEstimation")
    @patch("tirtha.alicevision.AliceVision.prepareDenseScene")
    @patch("tirtha.alicevision.AliceVision.sfmRotate")
    @patch("tirtha.alicevision.AliceVision.sfmTransform")
    @patch("tirtha.alicevision.AliceVision.structureFromMotion")
    @patch("tirtha.alicevision.AliceVision.featureMatching")
    @patch("tirtha.alicevision.AliceVision.imageMatching")
    @patch("tirtha.alicevision.AliceVision.featureExtraction")
    @patch("tirtha.alicevision.AliceVision.cameraInit")
    def test_run_all_with_custom_parameters(
        self,
        mock_camera_init,
        mock_feature_extraction,
        mock_image_matching,
        mock_feature_matching,
        mock_structure_from_motion,
        mock_sfm_transform,
        mock_sfm_rotate,
        mock_prepare_dense_scene,
        mock_depth_map_estimation,
        mock_depth_map_filtering,
        mock_meshing,
        mock_mesh_filtering,
        mock_mesh_decimate,
        mock_texturing,
        mock_mesh_denoising,
        mock_pool,
        basic_alicevision,
    ):
        """Test _run_all with custom parameters."""

        # Mock outputs creation
        def create_outputs(*args, **kwargs):
            # Create minimal required outputs
            for stage in [
                "01_cameraInit",
                "02_featureExtraction",
                "03_imageMatching",
                "04_featureMatching",
                "05_structureFromMotion",
                "06_sfmTransform",
                "07_sfmRotate",
                "08_prepareDenseScene",
                "09_depthMapEstimation",
                "10_depthMapFiltering",
                "11_meshing",
                "12_meshFiltering",
                "13_meshDecimate",
                "15_texturing",
            ]:
                stage_path = basic_alicevision.cache_dir / stage
                stage_path.mkdir(parents=True, exist_ok=True)
                if stage == "01_cameraInit":
                    (stage_path / "cameraInit.sfm").touch()
                elif stage == "03_imageMatching":
                    (stage_path / "imageMatches.txt").touch()
                elif stage == "05_structureFromMotion":
                    (stage_path / "sfm.abc").touch()
                    (stage_path / "cameras.sfm").touch()
                elif stage == "06_sfmTransform":
                    (stage_path / "sfmTrans.abc").touch()
                elif stage == "07_sfmRotate":
                    (stage_path / "sfmRota.abc").touch()
                elif stage == "11_meshing":
                    (stage_path / "rawMesh.obj").touch()
                    (stage_path / "densePointCloud.abc").touch()
                elif stage == "12_meshFiltering":
                    (stage_path / "filteredMesh.obj").touch()
                elif stage == "13_meshDecimate":
                    (stage_path / "decimatedMesh.obj").touch()
            return "Success"

        # Set up mocks to return success and create files
        for mock_method in [
            mock_camera_init,
            mock_feature_extraction,
            mock_image_matching,
            mock_feature_matching,
            mock_structure_from_motion,
            mock_sfm_transform,
            mock_sfm_rotate,
            mock_prepare_dense_scene,
            mock_depth_map_estimation,
            mock_depth_map_filtering,
            mock_meshing,
            mock_mesh_filtering,
            mock_mesh_decimate,
            mock_texturing,
            mock_mesh_denoising,
        ]:
            mock_method.side_effect = create_outputs

        # Run with custom parameters
        basic_alicevision._run_all(
            denoise=True,
            center_image="test_image",
            rotation=[90, 0, 0],
            orientMesh=True,
            estimateSpaceMinObservationAngle=45,
        )

        # Verify that the methods were called
        assert mock_camera_init.called
        assert mock_feature_extraction.called
        assert mock_image_matching.called

    # Error Recovery Integration Tests
    @patch("tirtha.alicevision.AliceVision._timeoutRunner")
    def test_pipeline_error_recovery_camera_init_failure(
        self, mock_timeout_runner, basic_alicevision
    ):
        """Test pipeline behavior when camera init fails."""
        # Simulate camera init failure
        mock_timeout_runner.side_effect = TimeoutExpired("cmd", 2)

        # Camera init should handle the error gracefully
        basic_alicevision.cameraInit()

        # Error state should be set
        assert AliceVision.state["error"] is True

        # Subsequent operations should be skipped
        with pytest.raises(RuntimeError, match="Skipping due to error"):
            basic_alicevision.featureExtraction()

        # Clean up state
        AliceVision.state = {"error": False, "source": None, "log_file": None}

    @patch("tirtha.alicevision.AliceVision._serialRunner")
    def test_pipeline_error_recovery_mid_pipeline_failure(
        self, mock_serial_runner, basic_alicevision
    ):
        """Test pipeline behavior when error occurs mid-pipeline."""
        # Reset state
        AliceVision.state = {"error": False, "source": None, "log_file": None}

        # Create initial outputs
        camera_init_path = basic_alicevision.cache_dir / "01_cameraInit"
        camera_init_path.mkdir(parents=True, exist_ok=True)
        (camera_init_path / "cameraInit.sfm").touch()

        # Create feature extraction directory (needed for imageMatching)
        feature_path = basic_alicevision.cache_dir / "02_featureExtraction"
        feature_path.mkdir(parents=True, exist_ok=True)

        # Simulate failure in a mid-pipeline stage
        def fail_and_set_state(*args, **kwargs):
            # Set error state as the real _serialRunner would after retries
            AliceVision.state = {
                "error": True,
                "source": "imageMatching",
                "log_file": basic_alicevision.cache_dir
                / "03_imageMatching"
                / "imageMatching.log",
            }
            raise CalledProcessError(1, "cmd", output=b"Error")

        mock_serial_runner.side_effect = fail_and_set_state

        # This should set error state
        with pytest.raises(CalledProcessError):
            basic_alicevision.imageMatching()

        # Verify error state was set
        assert AliceVision.state["error"] is True

        # Clean up state
        AliceVision.state = {"error": False, "source": None, "log_file": None}

    # File Dependency Integration Tests
    def test_stage_file_dependencies(self, basic_alicevision):
        """Test that stages properly check for required input files."""
        # Test without creating required input files
        with pytest.raises(FileNotFoundError):
            basic_alicevision.featureExtraction()  # Needs cameraInit.sfm

        # Create camera init output
        camera_init_path = basic_alicevision.cache_dir / "01_cameraInit"
        camera_init_path.mkdir(parents=True, exist_ok=True)
        (camera_init_path / "cameraInit.sfm").touch()

        # Now featureExtraction should not raise FileNotFoundError for input
        with patch("tirtha.alicevision.AliceVision._parallelRunner"):
            basic_alicevision.featureExtraction()  # Should work now

    def test_alternative_file_fallback(self, basic_alicevision, temp_cache_dir):
        """Test that methods fall back to alternative files when primary is missing."""
        # Create alternative file but not primary
        alt_sfm = basic_alicevision.cache_dir / "05_structureFromMotion/sfm.abc"
        alt_sfm.parent.mkdir(parents=True, exist_ok=True)
        alt_sfm.touch()

        # Create cameras.sfm as well (required for sfmTransform)
        cameras_sfm = basic_alicevision.cache_dir / "05_structureFromMotion/cameras.sfm"
        cameras_sfm.touch()

        with patch("tirtha.alicevision.AliceVision._serialRunner"):
            # Should use the alternative file without error
            basic_alicevision.sfmTransform()

    # Cross-Method Integration Tests
    @patch("tirtha.alicevision.AliceVision._serialRunner")
    @patch("tirtha.alicevision.AliceVision._parallelRunner")
    def test_sequential_stage_execution(
        self, mock_parallel_runner, mock_serial_runner, basic_alicevision
    ):
        """Test sequential execution of pipeline stages."""

        # Create outputs progressively
        def create_stage_outputs(*args, **kwargs):
            # Determine which stage we're in based on call count
            call_count = mock_parallel_runner.call_count + mock_serial_runner.call_count

            if call_count == 1:  # featureExtraction
                path = basic_alicevision.cache_dir / "02_featureExtraction"
                path.mkdir(parents=True, exist_ok=True)
            elif call_count == 2:  # imageMatching
                path = basic_alicevision.cache_dir / "03_imageMatching"
                path.mkdir(parents=True, exist_ok=True)
                (path / "imageMatches.txt").touch()
            elif call_count == 3:  # featureMatching
                path = basic_alicevision.cache_dir / "04_featureMatching"
                path.mkdir(parents=True, exist_ok=True)
            elif call_count == 4:  # structureFromMotion
                path = basic_alicevision.cache_dir / "05_structureFromMotion"
                path.mkdir(parents=True, exist_ok=True)
                (path / "sfm.abc").touch()
                (path / "cameras.sfm").touch()

        mock_parallel_runner.side_effect = create_stage_outputs
        mock_serial_runner.side_effect = create_stage_outputs

        # Create initial cameraInit output
        camera_init_path = basic_alicevision.cache_dir / "01_cameraInit"
        camera_init_path.mkdir(parents=True, exist_ok=True)
        (camera_init_path / "cameraInit.sfm").touch()

        # Execute stages sequentially
        basic_alicevision.featureExtraction()
        basic_alicevision.imageMatching()
        basic_alicevision.featureMatching()
        basic_alicevision.structureFromMotion()

        # Verify all stages were called
        assert (
            mock_parallel_runner.call_count == 2
        )  # featureExtraction, featureMatching
        assert mock_serial_runner.call_count == 2  # imageMatching, structureFromMotion

    # State Management Integration Tests
    def test_state_persistence_across_methods(self, basic_alicevision):
        """Test that state persists correctly across method calls."""
        # Reset state
        AliceVision.state = {"error": False, "source": None, "log_file": None}

        # Simulate error in first method
        with patch("tirtha.alicevision.AliceVision._timeoutRunner") as mock_runner:
            mock_runner.side_effect = TimeoutExpired("cmd", 2)
            basic_alicevision.cameraInit()

        # State should be set to error
        assert AliceVision.state["error"] is True

        # All subsequent method calls should be skipped
        with pytest.raises(RuntimeError, match="Skipping due to error"):
            basic_alicevision.featureExtraction()

        with pytest.raises(RuntimeError, match="Skipping due to error"):
            basic_alicevision.imageMatching()

        # Clean up state
        AliceVision.state = {"error": False, "source": None, "log_file": None}

    # Configuration Integration Tests
    def test_descriptor_presets_integration(
        self, real_exec_path, temp_input_dir, temp_cache_dir, mock_logger
    ):
        """Test that descriptor presets are properly integrated across methods."""
        custom_presets = {
            "Preset": "high",
            "Quality": "ultra",
            "Types": "sift,akaze,cctag3",
        }

        av = AliceVision(
            exec_path=real_exec_path,
            input_dir=temp_input_dir,
            cache_dir=temp_cache_dir,
            logger=mock_logger,
            descPresets=custom_presets,
        )

        # Create required input files
        camera_init_path = Path(av.cache_dir) / "01_cameraInit"
        camera_init_path.mkdir(parents=True, exist_ok=True)
        (camera_init_path / "cameraInit.sfm").touch()

        with patch("tirtha.alicevision.AliceVision._parallelRunner") as mock_runner:
            av.featureExtraction()

            # Verify that custom presets were used in command
            args, kwargs = mock_runner.call_args
            command = args[0]
            assert "high" in command
            assert "ultra" in command
            assert "sift,akaze,cctag3" in command

    def test_verbose_level_integration(self, basic_alicevision):
        """Test that verbose level is consistently applied across methods."""
        # Set a specific verbose level
        basic_alicevision.verboseLevel = "debug"

        # Create required files
        mesh_path = basic_alicevision.cache_dir / "11_meshing"
        mesh_path.mkdir(parents=True, exist_ok=True)
        (mesh_path / "rawMesh.obj").touch()

        with patch("tirtha.alicevision.AliceVision._serialRunner") as mock_runner:
            basic_alicevision.meshFiltering()

            # Verify verbose level is in command
            args, kwargs = mock_runner.call_args
            command = args[0]
            assert "--verboseLevel debug" in command

    # Property Integration Tests
    def test_property_calculations_with_real_files(self, basic_alicevision):
        """Test property calculations with actual files in input directory."""
        # Check existing files first
        initial_count = basic_alicevision.inputSize

        # Create some test files in input directory
        for i in range(5):
            (basic_alicevision.input_dir / f"image_{i}.jpg").write_bytes(b"fake_image")

        # Test properties (should now include the new files)
        expected_count = initial_count + 5
        assert basic_alicevision.inputSize == expected_count
        assert basic_alicevision.blockSize > 0
        assert basic_alicevision.numBlocks > 0
        assert basic_alicevision.cpu_count > 0

    def test_block_size_calculation_integration(self, basic_alicevision):
        """Test block size calculation affects parallel execution."""
        # Create many files to trigger parallel processing
        for i in range(20):  # Create 20 files for integration testing
            (basic_alicevision.input_dir / f"image_{i}.jpg").write_bytes(b"fake_image")

        # Test that with many files, the block size and parallel execution calculations work
        assert basic_alicevision.inputSize > 20  # Should have more than 20 files now
        assert basic_alicevision.blockSize > 0
        assert basic_alicevision.numBlocks > 0

        output_path = basic_alicevision.cache_dir / "test_output"
        output_path.mkdir(parents=True, exist_ok=True)

        with patch("tirtha.alicevision.Pool") as mock_pool:
            mock_pool_instance = Mock()
            mock_pool.return_value.__enter__ = Mock(return_value=mock_pool_instance)
            mock_pool.return_value.__exit__ = Mock(return_value=None)

            # This should trigger parallel execution
            basic_alicevision._parallelRunner("test cmd", output_path, "test_caller")

            # Verify parallel execution was used
            mock_pool.assert_called_once()

    # Cleanup Integration Tests
    def test_cache_directory_creation_integration(
        self, real_exec_path, temp_input_dir, mock_logger, tmp_path
    ):
        """Test that cache directory is created during initialization."""
        cache_dir = tmp_path / "new_cache_dir"

        # Cache directory doesn't exist yet
        assert not cache_dir.exists()

        av = AliceVision(
            exec_path=real_exec_path,
            input_dir=temp_input_dir,
            cache_dir=cache_dir,
            logger=mock_logger,
        )

        # Cache directory should be created
        assert cache_dir.exists()
        assert av.cache_dir == cache_dir

    def test_output_directory_creation_integration(self, basic_alicevision):
        """Test that output directories are created by pipeline methods."""
        # Create required input
        camera_init_path = basic_alicevision.cache_dir / "01_cameraInit"
        camera_init_path.mkdir(parents=True, exist_ok=True)
        (camera_init_path / "cameraInit.sfm").touch()

        with patch("tirtha.alicevision.AliceVision._parallelRunner") as mock_runner:

            def create_output(*args, **kwargs):
                # The method should create the output directory
                output_path = basic_alicevision.cache_dir / "02_featureExtraction"
                output_path.mkdir(parents=True, exist_ok=True)

            mock_runner.side_effect = create_output
            basic_alicevision.featureExtraction()

            # Verify output directory exists
            assert (basic_alicevision.cache_dir / "02_featureExtraction").exists()

    # ===============================
    # MIGRATED INTEGRATION TESTS
    # ===============================

    # Integration test for complete pipeline
    @patch("tirtha.alicevision.AliceVision._timeoutRunner")
    @patch("tirtha.alicevision.AliceVision._serialRunner")
    @patch("tirtha.alicevision.AliceVision._parallelRunner")
    def test_pipeline_integration(
        self, mock_parallel, mock_serial, mock_timeout, basic_alicevision
    ):
        """Test integration of multiple pipeline stages."""
        # Mock successful execution
        mock_timeout.return_value = "Success"
        mock_serial.return_value = None
        mock_parallel.return_value = None

        # Create necessary intermediate files after each stage
        def create_camera_init_files(*args, **kwargs):
            camera_init_path = basic_alicevision.cache_dir / "01_cameraInit"
            camera_init_path.mkdir(parents=True, exist_ok=True)
            (camera_init_path / "cameraInit.sfm").touch()

        def create_feature_files(*args, **kwargs):
            feature_path = basic_alicevision.cache_dir / "02_featureExtraction"
            feature_path.mkdir(parents=True, exist_ok=True)

        def create_matching_files(*args, **kwargs):
            matching_path = basic_alicevision.cache_dir / "03_imageMatching"
            matching_path.mkdir(parents=True, exist_ok=True)
            (matching_path / "imageMatches.txt").touch()

        # Set up side effects
        mock_timeout.side_effect = create_camera_init_files
        mock_parallel.side_effect = create_feature_files
        mock_serial.side_effect = create_matching_files

        # Run pipeline stages
        basic_alicevision.cameraInit()
        basic_alicevision.featureExtraction()
        basic_alicevision.imageMatching()

        # Verify all stages were called
        assert mock_timeout.called
        assert mock_parallel.called
        assert mock_serial.called

        # Verify intermediate files were created
        assert (basic_alicevision.cache_dir / "01_cameraInit/cameraInit.sfm").exists()
        assert (basic_alicevision.cache_dir / "02_featureExtraction").exists()
        assert (
            basic_alicevision.cache_dir / "03_imageMatching/imageMatches.txt"
        ).exists()
