"""
Utility classes & functions

"""

import psutil
import subprocess as sp

from pathlib import Path
from typing import Union
from logging import DEBUG, Formatter, Logger
from logging.handlers import RotatingFileHandler


class Logger(Logger):
    """
    Logger class for logging to file

    """

    def __init__(
        self, name: str, log_path: Union[str, Path], level: str = DEBUG
    ) -> None:
        super().__init__(name, level)

        # Set up logging
        self.setLevel(level)
        self._log_file = Path(log_path) / f"{name}.log"
        self._log_file.touch(exist_ok=True)

        fh = RotatingFileHandler(
            self._log_file,
            maxBytes=10**8,  # 100 MB
            backupCount=5,  # Keep 5 backups
            encoding="utf-8",
        )
        fh.setLevel(level)
        formatter = Formatter(
            "%(asctime)s | %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S %Z"
        )
        fh.setFormatter(formatter)
        self.addHandler(fh)


def _sysinfo() -> dict:
    """
    Get system info (CPU, RAM, GPU)

    """
    # CPU + RAM
    cpu_util = psutil.cpu_percent(interval=0.5, percpu=True)
    ram = psutil.virtual_memory()
    ram = {
        "total": f"{ram.total / 2**30:.2f} GB",
        "available": f"{ram.available / 2**30:.2f} GB",
        "used": f"{ram.used / 2**30:.2f} GB",
        "free": f"{ram.free / 2**30:.2f} GB",
    }
    c_details = {
        "cpu_util": cpu_util,
        "mem": ram,
    }

    # GPU Details
    cmd = "nvidia-smi --query-gpu=driver_version,pstate,temperature.gpu,utilization.gpu,memory.free,memory.used,memory.total --format=csv"
    res = sp.check_output(cmd.split()).decode("ascii").split("\n")[:-1]
    labels = res[0].split(", ")
    vals = res[1].split(", ")
    vdetails = dict(zip(labels, vals))

    res = {
        "cpu": c_details,
        "gpu": vdetails,
    }

    return res
