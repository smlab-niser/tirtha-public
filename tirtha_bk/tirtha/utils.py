"""
Utility classes & functions

"""
from pathlib import Path
from logging import Logger, FileHandler, Formatter, DEBUG
from typing import Union


class Logger(Logger):
    """
    Logger class for logging to file

    """
    def __init__(self, name: str, log_path: Union[str, Path], level: str = DEBUG):
        super().__init__(name, level)

        # Set up logging
        self.setLevel(level)
        self._log_file = Path(log_path) / f"{name}.log"
        self._log_file.touch(exist_ok=True)

        fh = FileHandler(self._log_file)
        fh.setLevel(level)
        formatter = Formatter(
            '%(asctime)s | %(levelname)s: %(message)s',
            datefmt="%Y-%m-%d %H:%M:%S %Z"
        )
        fh.setFormatter(formatter)
        self.addHandler(fh)
