"""
Core functionality tests for AliceVision class.
"""

import pytest
import tempfile
import stat
from pathlib import Path
from unittest.mock import Mock, patch
from subprocess import CalledProcessError

from tirtha.alicevision import AliceVision


class TestAliceVisionCore:
    """Test core AliceVision functionality."""

    def test_initialization_with_real_executables(
        self, real_exec_path, temp_input_dir, temp_cache_dir, mock_logger
    ):
        """Test AliceVision initialization with real executables."""
        av = AliceVision(
            exec_path=real_exec_path,
            input_dir=temp_input_dir,
            cache_dir=temp_cache_dir,
            logger=mock_logger,
        )

        assert av.exec_path == real_exec_path
        assert av.input_dir == temp_input_dir
        assert av.cache_dir == temp_cache_dir
        assert av.logger == mock_logger

    def test_initialization_with_custom_settings(
        self, real_exec_path, temp_input_dir, temp_cache_dir, mock_logger
    ):
        """Test AliceVision initialization with custom settings."""
        av = AliceVision(
            exec_path=real_exec_path,
            input_dir=temp_input_dir,
            cache_dir=temp_cache_dir,
            logger=mock_logger,
            verboseLevel="info",
        )

        assert av.verboseLevel == "info"
        # These are set automatically in __post_init__
        assert hasattr(av, "minBlockSize")
        assert hasattr(av, "maxCores")

    @pytest.mark.parametrize("verbose_level", ["fatal", "error", "warning", "info"])
    def test_initialization_verbose_levels(
        self, real_exec_path, temp_input_dir, temp_cache_dir, mock_logger, verbose_level
    ):
        """Test AliceVision initialization with different verbose levels."""
        av = AliceVision(
            exec_path=real_exec_path,
            input_dir=temp_input_dir,
            cache_dir=temp_cache_dir,
            logger=mock_logger,
            verboseLevel=verbose_level,
        )
        assert av.verboseLevel == verbose_level

    @pytest.mark.parametrize(
        "invalid_type", ["invalid_type", "bad_descriptor", "unknown"]
    )
    def test_initialization_with_invalid_descriptors(
        self, real_exec_path, temp_input_dir, temp_cache_dir, mock_logger, invalid_type
    ):
        """Test AliceVision initialization with invalid descriptor types."""
        desc_presets = {"Preset": "normal", "Quality": "normal", "Types": invalid_type}

        with pytest.raises(ValueError, match="Invalid describerType"):
            AliceVision(
                exec_path=real_exec_path,
                input_dir=temp_input_dir,
                cache_dir=temp_cache_dir,
                logger=mock_logger,
                descPresets=desc_presets,
            )

    def test_initialization_with_empty_input_dir(
        self, real_exec_path, temp_cache_dir, mock_logger
    ):
        """Test initialization with empty input directory raises error."""
        with tempfile.TemporaryDirectory() as empty_dir:
            with pytest.raises(FileNotFoundError, match="Image folder is empty"):
                AliceVision(
                    exec_path=real_exec_path,
                    input_dir=Path(empty_dir),
                    cache_dir=temp_cache_dir,
                    logger=mock_logger,
                )

    def test_initialization_with_nonexistent_input_dir(
        self, real_exec_path, temp_cache_dir, mock_logger
    ):
        """Test initialization with non-existent input directory raises error."""
        with pytest.raises(FileNotFoundError, match="Image folder not found"):
            AliceVision(
                exec_path=real_exec_path,
                input_dir=Path("/nonexistent/path"),
                cache_dir=temp_cache_dir,
                logger=mock_logger,
            )

    @pytest.mark.parametrize(
        "verbose_level", ["trace", "debug", "info", "warning", "error", "fatal"]
    )
    def test_init_verbose_levels(
        self, real_exec_path, temp_input_dir, temp_cache_dir, mock_logger, verbose_level
    ):
        """Test AliceVision initialization with different verbose levels."""
        av = AliceVision(
            exec_path=real_exec_path,
            input_dir=temp_input_dir,
            cache_dir=temp_cache_dir,
            logger=mock_logger,
            verboseLevel=verbose_level,
        )
        assert av.verboseLevel == verbose_level

    @pytest.mark.parametrize(
        "preset,quality,types",
        [
            ("low", "low", "sift"),
            ("medium", "normal", "dspsift"),
            ("high", "high", "akaze"),
            ("ultra", "ultra", "sift,akaze"),
            ("normal", "medium", "cctag3,sift_float"),
        ],
    )
    def test_init_desc_presets(
        self,
        real_exec_path,
        temp_input_dir,
        temp_cache_dir,
        mock_logger,
        preset,
        quality,
        types,
    ):
        """Test AliceVision initialization with different descriptor presets."""
        desc_presets = {"Preset": preset, "Quality": quality, "Types": types}
        av = AliceVision(
            exec_path=real_exec_path,
            input_dir=temp_input_dir,
            cache_dir=temp_cache_dir,
            logger=mock_logger,
            descPresets=desc_presets,
        )
        assert av.descPresets == desc_presets

    def test_init_nonexistent_input_dir(
        self, real_exec_path, temp_cache_dir, mock_logger
    ):
        """Test initialization with non-existent input directory."""
        with pytest.raises(FileNotFoundError, match="Image folder not found"):
            AliceVision(
                exec_path=real_exec_path,
                input_dir="/nonexistent/path",
                cache_dir=temp_cache_dir,
                logger=mock_logger,
            )

    def test_init_empty_input_dir(self, real_exec_path, temp_cache_dir, mock_logger):
        """Test initialization with empty input directory."""
        with tempfile.TemporaryDirectory() as empty_dir:
            with pytest.raises(FileNotFoundError, match="Image folder is empty"):
                AliceVision(
                    exec_path=real_exec_path,
                    input_dir=empty_dir,
                    cache_dir=temp_cache_dir,
                    logger=mock_logger,
                )

    @pytest.mark.parametrize("invalid_preset", ["invalid", "super", "extreme"])
    def test_init_invalid_preset(
        self,
        real_exec_path,
        temp_input_dir,
        temp_cache_dir,
        mock_logger,
        invalid_preset,
    ):
        """Test initialization with invalid descriptor preset."""
        desc_presets = {"Preset": invalid_preset, "Quality": "normal", "Types": "sift"}
        with pytest.raises(
            ValueError, match="Invalid describerPreset or describerQuality"
        ):
            AliceVision(
                exec_path=real_exec_path,
                input_dir=temp_input_dir,
                cache_dir=temp_cache_dir,
                logger=mock_logger,
                descPresets=desc_presets,
            )

    @pytest.mark.parametrize(
        "invalid_type", ["invalid_type", "bad_descriptor", "unknown"]
    )
    def test_init_invalid_descriptor_type(
        self, real_exec_path, temp_input_dir, temp_cache_dir, mock_logger, invalid_type
    ):
        """Test initialization with invalid descriptor type."""
        desc_presets = {"Preset": "normal", "Quality": "normal", "Types": invalid_type}
        with pytest.raises(ValueError, match="Invalid describerType"):
            AliceVision(
                exec_path=real_exec_path,
                input_dir=temp_input_dir,
                cache_dir=temp_cache_dir,
                logger=mock_logger,
                descPresets=desc_presets,
            )

    # Test properties
    def test_input_size_property(self, basic_alicevision):
        """Test inputSize property calculation."""
        expected_size = len(list(basic_alicevision.input_dir.glob("*")))
        assert basic_alicevision.inputSize == expected_size

    def test_cpu_count_property(self, basic_alicevision):
        """Test cpu_count property returns actual CPU count."""
        import os

        actual_cpu_count = os.cpu_count() or 1
        assert basic_alicevision.cpu_count == actual_cpu_count

    def test_block_size_property(self, basic_alicevision):
        """Test blockSize property calculation."""
        expected_block_size = (
            basic_alicevision.inputSize
            if basic_alicevision.inputSize <= basic_alicevision.minBlockSize
            else basic_alicevision.inputSize // basic_alicevision.maxCores
        )
        assert basic_alicevision.blockSize == expected_block_size

    def test_num_blocks_property(self, basic_alicevision):
        """Test numBlocks property calculation."""
        expected_num_blocks = (
            basic_alicevision.inputSize // basic_alicevision.blockSize
        ) + 1
        assert basic_alicevision.numBlocks == expected_num_blocks

    # Test utility methods
    @pytest.mark.parametrize(
        "value,rng,should_pass",
        [
            (50, [0, 100], True),
            (1, [0, 100], True),
            (99, [0, 100], True),
            (0, [0, 100], False),  # boundary value
            (100, [0, 100], False),  # boundary value
            (150, [0, 100], False),  # out of range
            (-10, [0, 100], False),  # out of range
        ],
    )
    def test_check_value(self, basic_alicevision, value, rng, should_pass):
        """Test _check_value method with various values and ranges."""
        cmd = "base_command"

        if should_pass:
            result_cmd = basic_alicevision._check_value(cmd, "testParam", value, rng)
            assert f"--testParam {value}" in result_cmd
        else:
            with pytest.raises(ValueError):
                basic_alicevision._check_value(cmd, "testParam", value, rng)

    def test_check_input_existing_file(self, basic_alicevision, temp_input_dir):
        """Test _check_input with existing file."""
        test_file = temp_input_dir / "test_file.txt"
        test_file.touch()

        cmd = "base_command"
        result_cmd, result_file = basic_alicevision._check_input(cmd, test_file)

        assert str(test_file) in result_cmd
        assert result_file == test_file

    def test_check_input_nonexistent_file(self, basic_alicevision):
        """Test _check_input with non-existent file."""
        cmd = "base_command"
        with pytest.raises(FileNotFoundError):
            basic_alicevision._check_input(cmd, "/nonexistent/file.txt")

    def test_check_input_with_alternative(self, basic_alicevision, temp_input_dir):
        """Test _check_input with alternative file when primary doesn't exist."""
        alt_file = temp_input_dir / "alt_file.txt"
        alt_file.touch()

        cmd = "base_command"
        result_cmd, result_file = basic_alicevision._check_input(
            cmd, None, alt=alt_file
        )

        assert str(alt_file) in result_cmd
        assert result_file == alt_file

    @pytest.mark.parametrize("add_all", [True, False])
    def test_add_desc_presets(self, basic_alicevision, add_all):
        """Test _add_desc_presets method."""
        cmd = "base_command"
        result_cmd = basic_alicevision._add_desc_presets(cmd, addAll=add_all)

        assert "-d" in result_cmd
        if add_all:
            assert "-p" in result_cmd  # Fixed: code uses -p not --describerPreset
            assert "--describerQuality" in result_cmd

    # Test state management
    def test_check_state_normal_condition(self, basic_alicevision):
        """Test _check_state under normal conditions."""
        # Reset state to normal
        AliceVision.state = {"error": False, "source": None, "log_file": None}

        # Should not raise any exception
        basic_alicevision._check_state()

    def test_check_state_error_condition(self, basic_alicevision):
        """Test _check_state when error state is set."""
        # Set error state
        AliceVision.state = {
            "error": True,
            "source": "test_error",
            "log_file": "/tmp/test.log",
        }

        with pytest.raises(RuntimeError, match="Skipping due to error"):
            basic_alicevision._check_state()

        # Clean up state for other tests
        AliceVision.state = {"error": False, "source": None, "log_file": None}

    def test_class_state_persistence(self, basic_alicevision, real_exec_path):
        """Test that AliceVision class state persists across instances."""
        # Clean up any existing state first
        AliceVision.state = {"error": False, "source": None, "log_file": None}

        # Set error state
        AliceVision.state = {
            "error": True,
            "source": "test",
            "log_file": "/tmp/test.log",
        }

        # Test that state persists and _check_state raises RuntimeError
        with pytest.raises(RuntimeError, match="Skipping due to error"):
            basic_alicevision._check_state()

        # Clean up state for other tests
        AliceVision.state = {"error": False, "source": None, "log_file": None}

    # Test timeout runner
    @patch("tirtha.alicevision.Popen")
    def test_timeout_runner_success(self, mock_popen, basic_alicevision):
        """Test _timeoutRunner with successful command execution."""
        mock_process = Mock()
        mock_process.communicate.return_value = (b"output", b"error")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        result = basic_alicevision._timeoutRunner(["test", "command"], timeout=5)
        assert result == "output"
        mock_popen.assert_called_once()

    @patch("tirtha.alicevision.Popen")
    def test_timeout_runner_failure(self, mock_popen, basic_alicevision):
        """Test _timeoutRunner with failed command execution."""
        mock_process = Mock()
        mock_process.communicate.return_value = (b"output", b"error")
        mock_process.returncode = 1
        mock_popen.return_value = mock_process

        with pytest.raises(CalledProcessError):
            basic_alicevision._timeoutRunner(["test", "command"], timeout=5)

        mock_popen.assert_called_once()

    @patch("tirtha.alicevision.Popen")
    def test_timeout_runner_with_working_directory(self, mock_popen, basic_alicevision):
        """Test _timeoutRunner method."""
        mock_process = Mock()
        mock_process.communicate.return_value = (b"output", b"error")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        result = basic_alicevision._timeoutRunner(["test", "command"], timeout=5)

        assert result == "output"
        mock_popen.assert_called_once()

    # Test real executables exist and are accessible
    def test_real_executable_paths(self, real_exec_path):
        """Test that real AliceVision executables exist and are accessible."""
        # Check key executables exist
        key_executables = [
            "aliceVision_cameraInit",
            "aliceVision_featureExtraction",
            "aliceVision_imageMatching",
            "aliceVision_featureMatching",
            "aliceVision_meshing",
            "aliceVision_texturing",
        ]

        for exe in key_executables:
            exe_path = real_exec_path / exe
            assert exe_path.exists(), f"Executable {exe} not found at {exe_path}"
            assert exe_path.is_file(), f"{exe} is not a file"
            # Check if executable (on Unix systems)
            assert exe_path.stat().st_mode & stat.S_IEXEC, f"{exe} is not executable"

        # Check required database files
        assert (real_exec_path / "cameraSensors.db").exists()
        assert (real_exec_path / "vlfeat_K80L3.SIFT.tree").exists()

    # Test error conditions
    def test_check_input_missing_alternative(self, basic_alicevision):
        """Test _check_input when both primary and alternative files are missing."""
        cmd = "base_command"
        with pytest.raises(FileNotFoundError):
            basic_alicevision._check_input(cmd, None, alt="/nonexistent/alt.txt")

    @patch("tirtha.alicevision.AliceVision._check_state")
    def test_methods_check_state_called(self, mock_check_state, basic_alicevision):
        """Test that pipeline methods call _check_state."""
        # Create minimal required files for different methods
        camera_init_path = basic_alicevision.cache_dir / "01_cameraInit"
        camera_init_path.mkdir(parents=True, exist_ok=True)
        (camera_init_path / "cameraInit.sfm").touch()

        feature_path = basic_alicevision.cache_dir / "02_featureExtraction"
        feature_path.mkdir(parents=True, exist_ok=True)

        with patch("tirtha.alicevision.AliceVision._parallelRunner"):
            basic_alicevision.featureExtraction()
            mock_check_state.assert_called()
