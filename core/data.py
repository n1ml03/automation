"""
Data Module - CSV and JSON data read/write management
"""

import csv
import json
import os
from datetime import datetime
from typing import Any, List, Dict, Optional
from collections import Counter
from .utils import get_logger, ensure_directory

logger = get_logger(__name__)


# ==================== DATA LOADING ====================

def _validate_file(file_path: str) -> None:
    """Validate file exists, raise FileNotFoundError if not."""
    if not os.path.exists(file_path):
        logger.error(f" File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")


def load_csv(file_path: str, encoding: str = 'utf-8') -> List[Dict[str, Any]]:
    """Read data from CSV file."""
    _validate_file(file_path)
    try:
        with open(file_path, encoding=encoding) as f:
            data = list(csv.DictReader(f))
        logger.info(f" Loaded {len(data)} rows from CSV: {file_path}")
        return data
    except Exception as e:
        logger.error(f" Error loading CSV: {e}")
        return []


def load_json(file_path: str, encoding: str = 'utf-8') -> List[Dict[str, Any]]:
    """Read data from JSON file, normalize to list."""
    _validate_file(file_path)
    try:
        with open(file_path, encoding=encoding) as f:
            data = json.load(f)
        
        result = data if isinstance(data, list) else [data] if isinstance(data, dict) else []
        if not isinstance(data, (list, dict)):
            logger.warning(f" Unexpected JSON type: {type(data)}, returning empty list")
        
        logger.info(f" Loaded {len(result)} items from JSON: {file_path}")
        return result
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f" Error loading JSON: {e}")
        return []


def load_data(file_path: str, encoding: str = 'utf-8') -> List[Dict[str, Any]]:
    """Auto-detect and load data from CSV or JSON file."""
    _validate_file(file_path)
    ext = os.path.splitext(file_path)[1].lower()
    loaders = {'.csv': load_csv, '.json': load_json}
    
    if ext not in loaders:
        raise ValueError(f"Unsupported file format: {ext}. Use .csv or .json")
    return loaders[ext](file_path, encoding)


# ==================== DATA WRITING ====================

def write_csv(file_path: str, data: List[Dict[str, Any]],
              encoding: str = 'utf-8', mode: str = 'w') -> bool:
    """Write list of dictionaries to CSV file."""
    if not data:
        logger.warning(" No data to write")
        return False

    try:
        if directory := os.path.dirname(file_path):
            ensure_directory(directory)

        with open(file_path, mode, newline='', encoding=encoding) as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            if mode == 'w' or (mode == 'a' and not os.path.exists(file_path)):
                writer.writeheader()
            writer.writerows(data)

        logger.info(f" Wrote {len(data)} rows to CSV: {file_path}")
        return True
    except Exception as e:
        logger.error(f" Error writing CSV: {e}")
        return False


# ==================== RESULT WRITER CLASS ====================

class ResultWriter:
    """Utility class for writing test/automation results to CSV with buffering."""

    RESULT_OK, RESULT_NG, RESULT_SKIP, RESULT_ERROR = 'OK', 'NG', 'SKIP', 'ERROR'

    def __init__(self, output_path: str, auto_write: bool = False):
        """Initialize ResultWriter with output path and auto-write option."""
        self.output_path = output_path
        self.auto_write = auto_write
        self.results: List[Dict[str, Any]] = []

        if directory := os.path.dirname(output_path):
            ensure_directory(directory)
        logger.info(f"ResultWriter initialized: {output_path}")

    def add_result(self, test_case: Dict[str, Any], result: str,
                  error_message: Optional[str] = None,
                  extra_fields: Optional[Dict[str, Any]] = None) -> None:
        """Add a test case result with timestamp and optional error/extra fields."""
        row_data = {**test_case, 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                    'result': result}
        if error_message:
            row_data['error_message'] = error_message
        if extra_fields:
            row_data.update(extra_fields)
        
        self.results.append(row_data)
        if self.auto_write:
            self.write()

    def write(self, clear_after_write: bool = False) -> bool:
        """Write all results to CSV file, optionally clear buffer after."""
        if not self.results:
            logger.warning(" No results to write")
            return False

        success = write_csv(self.output_path, self.results)
        if success and clear_after_write:
            self.clear()
        return success

    def clear(self) -> None:
        """Clear results buffer."""
        self.results.clear()
        logger.debug(" Results buffer cleared")

    def get_summary(self) -> Dict[str, int]:
        """Get results summary with count of each result type."""
        return dict(Counter(row.get('result', self.RESULT_ERROR) for row in self.results))

    def print_summary(self) -> None:
        """Print results summary to logger."""
        summary = self.get_summary()
        total = sum(summary.values())

        logger.info("=" * 60)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("=" * 60)

        if total == 0:
            logger.info("  No results recorded")
        else:
            for result_type, count in sorted(summary.items()):
                if count > 0:
                    logger.info(f"  {result_type:<10}: {count:>4} ({count/total*100:>5.1f}%)")

        logger.info("-" * 60)
        logger.info(f"  {'TOTAL':<10}: {total:>4}")
        logger.info("=" * 60)

    @property
    def count(self) -> int:
        """Current number of results."""
        return len(self.results)

    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return not self.results