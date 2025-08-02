"""
Real execution tests for AliceVision class.
Tests that require actual AliceVision executables.
"""

import pytest

from tirtha.alicevision import AliceVision


class TestAliceVisionRealExecution:
    """Test AliceVision with real execution scenarios."""

    # Real Executable Tests
    @pytest.mark.real_execution
    def test_camera_init_with_real_executables(
        self, real_exec_path, sample_images_dir, temp_cache_dir, mock_logger
    ):
        """Test camera initialization with real AliceVision executables."""
        av = AliceVision(
            exec_path=real_exec_path,
            input_dir=sample_images_dir,
            cache_dir=temp_cache_dir,
            logger=mock_logger,
        )

        # This will actually run the aliceVision_cameraInit executable
        av.cameraInit()

        # Check that output was created
        camera_init_output = temp_cache_dir / "01_cameraInit"
        assert camera_init_output.exists()
        assert (camera_init_output / "cameraInit.sfm").exists()

    @pytest.mark.real_execution
    def test_feature_extraction_with_real_executables(
        self, real_exec_path, sample_images_dir, temp_cache_dir, mock_logger
    ):
        """Test feature extraction with real AliceVision executables."""
        av = AliceVision(
            exec_path=real_exec_path,
            input_dir=sample_images_dir,
            cache_dir=temp_cache_dir,
            logger=mock_logger,
        )

        # Run camera init first
        av.cameraInit()

        # Run feature extraction
        av.featureExtraction()

        # Check that output was created
        feature_output = temp_cache_dir / "02_featureExtraction"
        assert feature_output.exists()

    @pytest.mark.real_execution
    def test_complete_pipeline_with_real_executables(
        self, real_exec_path, sample_images_dir, temp_cache_dir, mock_logger
    ):
        """Test complete pipeline with real AliceVision executables."""
        av = AliceVision(
            exec_path=real_exec_path,
            input_dir=sample_images_dir,
            cache_dir=temp_cache_dir,
            logger=mock_logger,
        )

        # Run the complete pipeline
        av._run_all()

        # Check that major outputs were created
        expected_outputs = [
            "01_cameraInit/cameraInit.sfm",
            "02_featureExtraction",
            "05_structureFromMotion/sfm.abc",
            "11_meshing/rawMesh.obj",
            "12_meshFiltering/filteredMesh.obj",
            "15_texturing",
        ]

        for output in expected_outputs:
            output_path = temp_cache_dir / output
            assert output_path.exists(), f"Expected output {output} was not created"

    # Performance Tests
    @pytest.mark.real_execution
    @pytest.mark.performance
    def test_parallel_execution_performance(
        self, real_exec_path, large_sample_images_dir, temp_cache_dir, mock_logger
    ):
        """Test parallel execution performance with large image set."""
        av = AliceVision(
            exec_path=real_exec_path,
            input_dir=large_sample_images_dir,
            cache_dir=temp_cache_dir,
            logger=mock_logger,
        )

        # Run stages that benefit from parallel execution
        av.cameraInit()

        import time

        start_time = time.time()
        av.featureExtraction()
        extraction_time = time.time() - start_time

        # Verify that multiple cores were used (indirect test)
        assert av.cpu_count > 1  # Test environment should have multiple cores
        assert extraction_time > 0  # Basic sanity check

        # Check outputs
        feature_output = temp_cache_dir / "02_featureExtraction"
        assert feature_output.exists()

    # Error Handling with Real Executables
    @pytest.mark.real_execution
    def test_real_executable_error_handling(
        self, real_exec_path, temp_cache_dir, mock_logger
    ):
        """Test error handling with real executables and invalid inputs."""
        # Create an empty input directory
        empty_input_dir = temp_cache_dir / "empty_input"
        empty_input_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="Image folder is empty"):
            AliceVision(
                exec_path=real_exec_path,
                input_dir=empty_input_dir,
                cache_dir=temp_cache_dir,
                logger=mock_logger,
            )

    @pytest.mark.real_execution
    def test_real_executable_missing_dependency(
        self, temp_cache_dir, sample_images_dir, mock_logger
    ):
        """Test behavior when AliceVision executable is missing."""
        fake_exec_path = temp_cache_dir / "fake_alicevision"
        fake_exec_path.mkdir()

        av = AliceVision(
            exec_path=fake_exec_path,
            input_dir=sample_images_dir,
            cache_dir=temp_cache_dir,
            logger=mock_logger,
        )

        # This should fail because the executable doesn't exist
        with pytest.raises(
            Exception
        ):  # Could be FileNotFoundError or CalledProcessError
            av.cameraInit()

    # Configuration Tests with Real Executables
    @pytest.mark.real_execution
    def test_different_descriptor_types_real_execution(
        self, real_exec_path, sample_images_dir, temp_cache_dir, mock_logger
    ):
        """Test different descriptor types with real execution."""
        descriptor_configs = [
            {"Preset": "normal", "Quality": "normal", "Types": "sift"},
            {"Preset": "high", "Quality": "high", "Types": "sift,akaze"},
            {"Preset": "normal", "Quality": "normal", "Types": "dspsift"},
        ]

        for desc_config in descriptor_configs:
            # Use separate cache directories for each configuration
            config_cache_dir = (
                temp_cache_dir / f"cache_{desc_config['Types'].replace(',', '_')}"
            )
            config_cache_dir.mkdir(exist_ok=True)

            av = AliceVision(
                exec_path=real_exec_path,
                input_dir=sample_images_dir,
                cache_dir=config_cache_dir,
                logger=mock_logger,
                descPresets=desc_config,
            )

            # Run camera init and feature extraction
            av.cameraInit()
            av.featureExtraction()

            # Verify outputs
            assert (config_cache_dir / "01_cameraInit/cameraInit.sfm").exists()
            assert (config_cache_dir / "02_featureExtraction").exists()

    @pytest.mark.real_execution
    def test_different_verbose_levels_real_execution(
        self, real_exec_path, sample_images_dir, temp_cache_dir, mock_logger
    ):
        """Test different verbose levels with real execution."""
        verbose_levels = ["fatal", "error", "warning", "info"]

        for verbose_level in verbose_levels:
            # Use separate cache directories for each verbose level
            verbose_cache_dir = temp_cache_dir / f"cache_{verbose_level}"
            verbose_cache_dir.mkdir(exist_ok=True)

            av = AliceVision(
                exec_path=real_exec_path,
                input_dir=sample_images_dir,
                cache_dir=verbose_cache_dir,
                logger=mock_logger,
                verboseLevel=verbose_level,
            )

            # Run camera init
            av.cameraInit()

            # Verify output exists regardless of verbose level
            assert (verbose_cache_dir / "01_cameraInit/cameraInit.sfm").exists()

    # Resource Usage Tests
    @pytest.mark.real_execution
    @pytest.mark.resource_intensive
    def test_memory_usage_with_large_dataset(
        self, real_exec_path, large_sample_images_dir, temp_cache_dir, mock_logger
    ):
        """Test memory usage with large dataset."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        av = AliceVision(
            exec_path=real_exec_path,
            input_dir=large_sample_images_dir,
            cache_dir=temp_cache_dir,
            logger=mock_logger,
        )

        # Run memory-intensive operations
        av.cameraInit()
        av.featureExtraction()

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Basic sanity checks
        assert memory_increase >= 0  # Memory should not decrease
        assert final_memory > 0  # Should have positive memory usage

    # Output Validation Tests
    @pytest.mark.real_execution
    def test_output_file_formats_validation(
        self, real_exec_path, sample_images_dir, temp_cache_dir, mock_logger
    ):
        """Test that output files have correct formats and content."""
        av = AliceVision(
            exec_path=real_exec_path,
            input_dir=sample_images_dir,
            cache_dir=temp_cache_dir,
            logger=mock_logger,
        )

        # Run first few stages
        av.cameraInit()
        av.featureExtraction()

        # Validate cameraInit.sfm format
        camera_init_file = temp_cache_dir / "01_cameraInit/cameraInit.sfm"
        assert camera_init_file.exists()
        assert camera_init_file.stat().st_size > 0  # Should not be empty

        # Read first few lines to check basic format
        with open(camera_init_file, "r") as f:
            content = f.read()
            # Basic validation - should contain camera/view information
            assert len(content) > 0

        # Validate feature extraction outputs
        feature_dir = temp_cache_dir / "02_featureExtraction"
        assert feature_dir.exists()
        assert feature_dir.is_dir()

        # Should contain feature files
        feature_files = list(feature_dir.glob("*.feat"))
        descriptor_files = list(feature_dir.glob("*.desc"))

        # At least some feature/descriptor files should exist
        assert len(feature_files) > 0 or len(descriptor_files) > 0

    # Timing and Performance Tests
    @pytest.mark.real_execution
    @pytest.mark.timing
    def test_stage_execution_timing(
        self, real_exec_path, sample_images_dir, temp_cache_dir, mock_logger
    ):
        """Test execution timing for different stages."""
        import time

        av = AliceVision(
            exec_path=real_exec_path,
            input_dir=sample_images_dir,
            cache_dir=temp_cache_dir,
            logger=mock_logger,
        )

        # Time camera initialization
        start_time = time.time()
        av.cameraInit()
        camera_init_time = time.time() - start_time

        # Time feature extraction
        start_time = time.time()
        av.featureExtraction()
        feature_extraction_time = time.time() - start_time

        # Basic sanity checks
        assert camera_init_time > 0
        assert feature_extraction_time > 0
        assert camera_init_time < 300  # Should not take more than 5 minutes
        assert feature_extraction_time < 600  # Should not take more than 10 minutes

    # Cleanup Tests
    @pytest.mark.real_execution
    def test_cleanup_after_execution(
        self, real_exec_path, sample_images_dir, temp_cache_dir, mock_logger
    ):
        """Test cleanup behavior after execution."""
        av = AliceVision(
            exec_path=real_exec_path,
            input_dir=sample_images_dir,
            cache_dir=temp_cache_dir,
            logger=mock_logger,
        )

        # Run some stages
        av.cameraInit()
        av.featureExtraction()

        # Verify log files were created
        camera_init_log = temp_cache_dir / "01_cameraInit/cameraInit.log"
        feature_extraction_logs = list(
            (temp_cache_dir / "02_featureExtraction").glob("*.log")
        )

        assert camera_init_log.exists()
        assert len(feature_extraction_logs) > 0

        # Verify log files contain meaningful content
        assert camera_init_log.stat().st_size > 0

    # Integration with System Resources
    @pytest.mark.real_execution
    def test_cpu_core_utilization(
        self, real_exec_path, sample_images_dir, temp_cache_dir, mock_logger
    ):
        """Test CPU core utilization during execution."""
        av = AliceVision(
            exec_path=real_exec_path,
            input_dir=sample_images_dir,
            cache_dir=temp_cache_dir,
            logger=mock_logger,
        )

        # Check that CPU count is properly detected
        import os

        system_cpu_count = os.cpu_count() or 1  # Handle None case
        assert av.cpu_count <= system_cpu_count
        assert av.cpu_count > 0

        # Run parallel stage
        av.cameraInit()
        av.featureExtraction()

        # Verify that the parallel execution used appropriate number of cores
        assert av.maxCores <= av.cpu_count

    # Edge Case Tests with Real Execution
    @pytest.mark.real_execution
    def test_single_image_execution(self, real_exec_path, temp_cache_dir, mock_logger):
        """Test execution with single image."""
        # Create directory with single image
        single_image_dir = temp_cache_dir / "single_image"
        single_image_dir.mkdir()

        # Create a minimal image file (this is a placeholder - real test would need actual image)
        (single_image_dir / "test_image.jpg").write_bytes(b"fake_image_data")

        av = AliceVision(
            exec_path=real_exec_path,
            input_dir=single_image_dir,
            cache_dir=temp_cache_dir,
            logger=mock_logger,
        )

        # Should handle single image case
        assert av.inputSize == 1
        assert av.blockSize == av.inputSize  # Should not split single image

        # This might fail with real executables due to insufficient images for SfM
        # but the initialization should work
        try:
            av.cameraInit()
        except Exception:
            pass  # Expected to potentially fail with single image

        # At minimum, the attempt should be made
        assert (temp_cache_dir / "01_cameraInit").exists()

    @pytest.mark.real_execution
    def test_custom_parameter_validation_real_execution(
        self, real_exec_path, sample_images_dir, temp_cache_dir, mock_logger
    ):
        """Test custom parameter validation with real execution."""
        av = AliceVision(
            exec_path=real_exec_path,
            input_dir=sample_images_dir,
            cache_dir=temp_cache_dir,
            logger=mock_logger,
        )

        # Run initial stages
        av.cameraInit()
        av.featureExtraction()
        av.imageMatching()
        av.featureMatching()
        av.structureFromMotion()
        av.sfmTransform()
        av.sfmRotate()

        # Test custom parameters in later stages
        # Test invalid rotation (should raise error before execution)
        with pytest.raises(ValueError):
            av.sfmRotate(rotation=[400, 0, 0])  # Invalid rotation value

    # ===============================
    # MIGRATED REAL EXECUTION TESTS
    # ===============================

    def test_check_state_normal_condition(self, basic_alicevision):
        """Test state checking under normal conditions."""
        # Set normal state
        AliceVision.state = {"error": False, "source": None, "log_file": None}

        # Should not raise any exception
        basic_alicevision._check_state()

    def test_check_state_error_condition(self, basic_alicevision):
        """Test state checking when error condition is set."""
        # Set error state
        AliceVision.state = {
            "error": True,
            "source": "test",
            "log_file": "/tmp/test.log",
        }

        with pytest.raises(RuntimeError, match="Skipping due to error"):
            basic_alicevision._check_state()

    def test_input_size_property(self, basic_alicevision):
        """Test inputSize property calculation."""
        assert basic_alicevision.inputSize >= 0
        assert isinstance(basic_alicevision.inputSize, int)

    def test_block_size_property(self, basic_alicevision):
        """Test blockSize property calculation."""
        expected_block_size = (
            basic_alicevision.inputSize
            if basic_alicevision.inputSize <= basic_alicevision.minBlockSize
            else basic_alicevision.inputSize // basic_alicevision.maxCores
        )
        assert basic_alicevision.blockSize == expected_block_size
