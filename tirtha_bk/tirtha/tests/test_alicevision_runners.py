"""
Runner methods tests for AliceVision class.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from subprocess import CalledProcessError, TimeoutExpired

from tirtha.alicevision import AliceVision


class TestAliceVisionRunners:
    """Test AliceVision runner methods."""

    # Serial Runner Tests
    @patch("tirtha.alicevision.check_output")
    def test_serial_runner_success(self, mock_check_output):
        """Test successful execution of serial runner."""
        mock_logger = Mock()
        AliceVision.logger = mock_logger  # Set class logger directly

        mock_check_output.return_value = b"Success output"

        log_file = Path("/tmp/test.log")
        AliceVision._serialRunner("test command", log_file)

        mock_check_output.assert_called_once()
        # Check the actual log messages that are called
        mock_logger.info.assert_any_call(
            f"Starting command execution. Log file: {log_file.resolve()}."
        )
        mock_logger.info.assert_any_call(
            f"Finished command execution after 0 retries. Log file: {log_file.resolve()}."
        )

    @patch("tirtha.alicevision.check_output")
    @patch("tirtha.alicevision.sleep")
    def test_serial_runner_retry_on_error(self, mock_sleep, mock_check_output):
        """Test serial runner retries on CalledProcessError."""
        mock_logger = Mock()
        AliceVision.logger = mock_logger  # Set class logger directly

        # Fail first two times, succeed on third
        mock_check_output.side_effect = [
            CalledProcessError(1, "cmd", output=b"Error 1"),
            CalledProcessError(1, "cmd", output=b"Error 2"),
            b"Success output",
        ]

        log_file = Path("/tmp/test.log")
        AliceVision._serialRunner("test command", log_file)

        assert mock_check_output.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("tirtha.alicevision.check_output")
    def test_serial_runner_max_retries_exceeded(self, mock_check_output):
        """Test serial runner when max retries are exceeded."""
        mock_logger = Mock()
        AliceVision.logger = mock_logger  # Set class logger directly

        # Always fail
        error = CalledProcessError(1, "cmd", output=b"Persistent error")
        mock_check_output.side_effect = error

        log_file = Path("/tmp/test.log")

        with pytest.raises(CalledProcessError):
            AliceVision._serialRunner("test command", log_file)

        # Should try MAX_RETRIES times
        assert mock_check_output.call_count == 3  # MAX_RETRIES
        # Should set error state
        assert AliceVision.state["error"] is True
        assert AliceVision.state["source"] is not None

    # Parallel Runner Tests
    @patch("tirtha.alicevision.Pool")
    def test_parallel_runner_single_command(self, mock_pool, basic_alicevision):
        """Test parallel runner with single command (blockSize >= inputSize)."""
        # Create a clean input directory for this test
        import tempfile

        with tempfile.TemporaryDirectory() as clean_input_dir:
            # Create specific number of files to control inputSize
            clean_path = Path(clean_input_dir)
            for i in range(4):
                (clean_path / f"test_{i}.jpg").touch()

            # Create a new instance with the clean directory
            test_av = AliceVision(
                exec_path=basic_alicevision.exec_path,
                input_dir=clean_path,
                cache_dir=basic_alicevision.cache_dir,
                logger=basic_alicevision.logger,
            )

            # Force specific cpu_count for predictable testing
            test_av.cpu_count = 2

            # Set up conditions for single command
            test_av.minBlockSize = 8  # inputSize < minBlockSize

            output_path = Path(test_av.cache_dir) / "test_output"
            output_path.mkdir(parents=True, exist_ok=True)

            # Mock the pool
            mock_pool_instance = Mock()
            mock_pool.return_value.__enter__ = Mock(return_value=mock_pool_instance)
            mock_pool.return_value.__exit__ = Mock(return_value=None)

            test_av._parallelRunner("test cmd", output_path, "test_caller")

            # Should create a pool with the forced maxCores value
            mock_pool.assert_called_once_with(2)

    # ===============================
    # MIGRATED TIMEOUT TESTS
    # ===============================

    # Test timeout runner
    @patch("tirtha.alicevision.Popen")
    def test_timeout_runner_success(self, mock_popen, basic_alicevision):
        """Test timeout runner with successful execution."""
        mock_process = Mock()
        mock_process.communicate.return_value = (b"success output", b"")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        result = basic_alicevision._timeoutRunner(["test", "command"], timeout=5)

        assert result == "success output"

    @patch("tirtha.alicevision.Popen")
    def test_timeout_runner_timeout(self, mock_popen, basic_alicevision):
        """Test timeout runner when command times out."""
        mock_process = Mock()
        mock_process.communicate.side_effect = TimeoutExpired("cmd", 5)
        mock_popen.return_value = mock_process

        with pytest.raises(TimeoutExpired):
            basic_alicevision._timeoutRunner(["test", "command"], timeout=5)

    @patch("tirtha.alicevision.Popen")
    def test_timeout_runner_process_error(self, mock_popen, basic_alicevision):
        """Test timeout runner when process returns error code."""
        mock_process = Mock()
        mock_process.communicate.return_value = (b"", b"error output")
        mock_process.returncode = 1
        mock_popen.return_value = mock_process

        with pytest.raises(CalledProcessError):
            basic_alicevision._timeoutRunner(["test", "command"], timeout=5)

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

    @patch("tirtha.alicevision.Pool")
    @patch("tirtha.alicevision.AliceVision._serialRunner")
    def test_parallel_runner_multiple_commands(
        self, mock_serial_runner, mock_pool, basic_alicevision
    ):
        """Test parallel runner with multiple commands (parallel execution)."""
        # Create many files to trigger parallel execution
        for i in range(20):  # Create 20 files for parallel testing
            (basic_alicevision.input_dir / f"test_{i}.jpg").touch()

        # Ensure maxCores is not zero to avoid division by zero
        basic_alicevision.maxCores = max(basic_alicevision.maxCores, 4)

        # Set up conditions for parallel execution
        basic_alicevision.minBlockSize = 4  # inputSize > minBlockSize

        output_path = basic_alicevision.cache_dir / "test_output"
        output_path.mkdir(parents=True, exist_ok=True)

        # Mock the pool
        mock_pool_instance = Mock()
        mock_pool.return_value.__enter__ = Mock(return_value=mock_pool_instance)
        mock_pool.return_value.__exit__ = Mock(return_value=None)

        basic_alicevision._parallelRunner("test cmd", output_path, "test_caller")

        # Should use Pool for parallel execution
        mock_pool.assert_called_once_with(basic_alicevision.cpu_count)
        mock_pool_instance.starmap.assert_called_once()

    # Timeout Runner Tests
    @patch("tirtha.alicevision.Popen")
    def test_timeout_runner_success(self, mock_popen, basic_alicevision):
        """Test successful execution of timeout runner."""
        mock_process = Mock()
        mock_process.communicate.return_value = (b"Success output", b"")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        result = basic_alicevision._timeoutRunner(["test", "command"], 10)

        assert result == "Success output"
        mock_process.communicate.assert_called_once_with(timeout=10)

    @patch("tirtha.alicevision.Popen")
    def test_timeout_runner_timeout_expired(self, mock_popen, basic_alicevision):
        """Test timeout runner when timeout expires."""
        mock_process = Mock()
        mock_process.communicate.side_effect = TimeoutExpired("cmd", 10)
        mock_process.kill.return_value = None
        mock_popen.return_value = mock_process

        with pytest.raises(TimeoutExpired):
            basic_alicevision._timeoutRunner(["test", "command"], 10)

        mock_process.kill.assert_called_once()

    @patch("tirtha.alicevision.Popen")
    def test_timeout_runner_process_error(self, mock_popen, basic_alicevision):
        """Test timeout runner when process fails."""
        mock_process = Mock()
        mock_process.communicate.return_value = (b"", b"Error output")
        mock_process.returncode = 1
        mock_popen.return_value = mock_process

        with pytest.raises(CalledProcessError):
            basic_alicevision._timeoutRunner(["test", "command"], 10)

    # Check Input Tests
    def test_check_input_existing_file(self, basic_alicevision, temp_cache_dir):
        """Test _check_input with existing file."""
        test_file = temp_cache_dir / "test_input.txt"
        test_file.touch()

        cmd = "base_command"
        result_cmd, result_input = basic_alicevision._check_input(cmd, test_file)

        assert str(test_file) in result_cmd
        assert result_input == test_file

    def test_check_input_nonexistent_file_with_alternative(
        self, basic_alicevision, temp_cache_dir
    ):
        """Test _check_input with None primary file and existing alternative."""
        alt_file = temp_cache_dir / "alternative.txt"
        alt_file.touch()

        cmd = "base_command"
        result_cmd, result_input = basic_alicevision._check_input(
            cmd, None, alt=alt_file
        )

        assert str(alt_file) in result_cmd
        assert result_input == alt_file

    def test_check_input_nonexistent_file_no_alternative(
        self, basic_alicevision, temp_cache_dir
    ):
        """Test _check_input with non-existent file and no alternative."""
        nonexistent_file = temp_cache_dir / "nonexistent.txt"

        cmd = "base_command"

        with pytest.raises(FileNotFoundError):
            basic_alicevision._check_input(cmd, nonexistent_file)

    def test_check_input_custom_argument(self, basic_alicevision, temp_cache_dir):
        """Test _check_input with custom argument flag."""
        test_file = temp_cache_dir / "test_input.txt"
        test_file.touch()

        cmd = "base_command"
        result_cmd, result_input = basic_alicevision._check_input(
            cmd, test_file, arg="--custom-input"
        )

        assert "--custom-input" in result_cmd
        assert str(test_file) in result_cmd

    # Check Value Tests
    def test_check_value_valid_range(self, basic_alicevision):
        """Test _check_value with value in valid range."""
        cmd = "base_command"
        result_cmd = basic_alicevision._check_value(cmd, "param", 5, [0, 10])

        assert "--param 5" in result_cmd

    def test_check_value_out_of_range(self, basic_alicevision):
        """Test _check_value with value out of range."""
        cmd = "base_command"

        with pytest.raises(ValueError, match="Value must be between 0 and 10"):
            basic_alicevision._check_value(cmd, "param", 15, [0, 10])

    def test_check_value_edge_cases(self, basic_alicevision):
        """Test _check_value with edge case values (boundary values should fail)."""
        cmd = "base_command"

        # Test minimum boundary value (should fail - boundary excluded)
        with pytest.raises(ValueError, match="Value must be between 0 and 10"):
            basic_alicevision._check_value(cmd, "param", 0, [0, 10])

        # Test maximum boundary value (should fail - boundary excluded)
        with pytest.raises(ValueError, match="Value must be between 0 and 10"):
            basic_alicevision._check_value(cmd, "param", 10, [0, 10])

        # Test valid values (just inside boundaries)
        result_cmd = basic_alicevision._check_value(cmd, "param", 1, [0, 10])
        assert "--param 1" in result_cmd

        result_cmd = basic_alicevision._check_value(cmd, "param", 9, [0, 10])
        assert "--param 9" in result_cmd

    # Add Descriptor Presets Tests
    def test_add_desc_presets_default(self, basic_alicevision):
        """Test _add_desc_presets with default parameters (addAll=False)."""
        cmd = "base_command"
        result_cmd = basic_alicevision._add_desc_presets(cmd)

        # Should only include descriptor types when addAll=False (default)
        assert "-d" in result_cmd
        assert "-p" not in result_cmd
        assert "--describerQuality" not in result_cmd

    def test_add_desc_presets_add_all(self, basic_alicevision):
        """Test _add_desc_presets with addAll=True."""
        cmd = "base_command"
        result_cmd = basic_alicevision._add_desc_presets(cmd, addAll=True)

        # Should include all preset parameters when addAll=True
        assert "-d" in result_cmd
        assert "-p" in result_cmd  # Code uses -p not --describerPreset
        assert "--describerQuality" in result_cmd

    def test_add_desc_presets_custom_values(
        self, real_exec_path, temp_input_dir, temp_cache_dir, mock_logger
    ):
        """Test _add_desc_presets with custom descriptor presets."""
        custom_presets = {"Preset": "high", "Quality": "ultra", "Types": "sift,akaze"}

        av = AliceVision(
            exec_path=real_exec_path,
            input_dir=temp_input_dir,
            cache_dir=temp_cache_dir,
            logger=mock_logger,
            descPresets=custom_presets,
        )

        cmd = "base_command"
        result_cmd = av._add_desc_presets(cmd, addAll=True)

        assert "high" in result_cmd
        assert "ultra" in result_cmd
        assert "sift,akaze" in result_cmd

    # State Check Tests
    def test_check_state_normal(self, basic_alicevision):
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

    # Runner Integration Tests
    @patch("tirtha.alicevision.AliceVision._serialRunner")
    def test_runner_integration_with_real_method(
        self, mock_serial_runner, basic_alicevision
    ):
        """Test runner integration with a real AliceVision method."""
        # Create required input files
        mesh_path = basic_alicevision.cache_dir / "11_meshing"
        mesh_path.mkdir(parents=True, exist_ok=True)
        (mesh_path / "rawMesh.obj").touch()

        # Test that meshFiltering calls _serialRunner
        basic_alicevision.meshFiltering()

        mock_serial_runner.assert_called_once()
        # Verify log file path is passed correctly
        args, kwargs = mock_serial_runner.call_args
        assert len(args) == 2  # cmd and log_file
        assert "meshFiltering.log" in str(args[1])

    @patch("tirtha.alicevision.AliceVision._parallelRunner")
    def test_parallel_runner_integration(self, mock_parallel_runner, basic_alicevision):
        """Test parallel runner integration with a real AliceVision method."""
        # Create required input files
        camera_init_path = basic_alicevision.cache_dir / "01_cameraInit"
        camera_init_path.mkdir(parents=True, exist_ok=True)
        (camera_init_path / "cameraInit.sfm").touch()

        # Test that featureExtraction calls _parallelRunner
        basic_alicevision.featureExtraction()

        mock_parallel_runner.assert_called_once()
        # Verify output path and caller are passed correctly
        args, kwargs = mock_parallel_runner.call_args
        assert len(args) == 3  # cmd, log_path, caller
        assert "02_featureExtraction" in str(args[1])
        assert args[2] == "featureExtraction"

    # Error Handling in Runners
    @patch("tirtha.alicevision.Logger")
    @patch("tirtha.alicevision.check_output")
    def test_serial_runner_sets_error_state_on_failure(
        self, mock_check_output, mock_logger_class
    ):
        """Test that serial runner sets error state on persistent failure."""
        # Reset state
        AliceVision.state = {"error": False, "source": None, "log_file": None}

        mock_logger = Mock()
        mock_logger_class.return_value = mock_logger

        # Always fail
        error = CalledProcessError(1, "cmd", output=b"Persistent error")
        mock_check_output.side_effect = error

        log_file = Path("/tmp/test.log")

        with pytest.raises(CalledProcessError):
            AliceVision._serialRunner("test command", log_file)

        # Verify error state was set
        assert AliceVision.state["error"] is True
        assert AliceVision.state["source"] == mock_logger.name
        assert str(log_file.resolve()) in str(AliceVision.state["log_file"])

    def test_check_state_class_level_persistence(self, basic_alicevision):
        """Test that state persists at class level across instances."""
        # Create new instance first (this resets state in __post_init__)
        new_av = AliceVision(
            exec_path=basic_alicevision.exec_path,
            input_dir=basic_alicevision.input_dir,
            cache_dir=basic_alicevision.cache_dir,
            logger=basic_alicevision.logger,
        )

        # Now set error state after both instances exist
        AliceVision.state = {
            "error": True,
            "source": "test_error",
            "log_file": "/tmp/test.log",
        }

        # Both instances should see the error state
        with pytest.raises(RuntimeError, match="Skipping due to error"):
            basic_alicevision._check_state()

        with pytest.raises(RuntimeError, match="Skipping due to error"):
            new_av._check_state()

        # Clean up state
        AliceVision.state = {"error": False, "source": None, "log_file": None}

    # Command Building Tests
    def test_command_building_verbosity(self, basic_alicevision):
        """Test that commands include verbosity level."""
        # Create required files for a test method
        mesh_path = basic_alicevision.cache_dir / "11_meshing"
        mesh_path.mkdir(parents=True, exist_ok=True)
        (mesh_path / "rawMesh.obj").touch()

        with patch("tirtha.alicevision.AliceVision._serialRunner") as mock_runner:
            basic_alicevision.meshFiltering()

            # Get the command that was called
            args, kwargs = mock_runner.call_args
            command = args[0]

            assert f"--verboseLevel {basic_alicevision.verboseLevel}" in command

    def test_command_building_executable_path(self, basic_alicevision):
        """Test that commands include correct executable path."""
        mesh_path = basic_alicevision.cache_dir / "11_meshing"
        mesh_path.mkdir(parents=True, exist_ok=True)
        (mesh_path / "rawMesh.obj").touch()

        with patch("tirtha.alicevision.AliceVision._serialRunner") as mock_runner:
            basic_alicevision.meshFiltering()

            # Get the command that was called
            args, kwargs = mock_runner.call_args
            command = args[0]

            # Should include the executable path
            assert str(basic_alicevision.exec_path) in command
            assert "aliceVision_meshFiltering" in command
