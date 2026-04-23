"""joblib serialization helpers."""
import logging
from pathlib import Path
from typing import Any, Union

import joblib

logger = logging.getLogger(__name__)


def load_joblib(file_path: Union[str, Path]) -> Any:
    """Load data from a joblib file."""
    data = joblib.load(file_path)
    logger.info(f"Loaded data from: {file_path}")
    return data


def save_joblib(data: Any, file_path: Union[str, Path], compress: int = 3) -> None:
    """Save data via joblib, creating parent directories if needed.

    Parameters
    ----------
    data : Any
    file_path : str or Path
    compress : int
        joblib compression level 0-9 (default 3).
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(data, file_path, compress=compress)
    logger.info(f"Saved data to: {file_path}")
