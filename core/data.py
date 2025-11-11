"""
Data Module - CSV and JSON data read/write management
"""

import csv
import json
import os
from datetime import datetime
from typing import Any, List, Dict, Optional
from .utils import get_logger, ensure_directory

logger = get_logger(__name__)


# ==================== DATA LOADING ====================

def load_csv(file_path: str, encoding: str = 'utf-8') -> List[Dict[str, Any]]:
    """
    Read data from CSV file.

    Args:
        file_path (str): CSV file path
        encoding (str): File encoding (default: utf-8)

    Returns:
        List[Dict[str, Any]]: List of dictionaries

    Raises:
        FileNotFoundError: If file does not exist
    """
    if not os.path.exists(file_path):
        logger.error(f" File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        with open(file_path, mode='r', encoding=encoding) as f:
            data = list(csv.DictReader(f))
        logger.info(f" Loaded {len(data)} rows from CSV: {file_path}")
        return data
    except Exception as e:
        logger.error(f" Error loading CSV: {e}")
        return []


def load_json(file_path: str, encoding: str = 'utf-8') -> List[Dict[str, Any]]:
    """
    Read data from JSON file.

    Args:
        file_path (str): JSON file path
        encoding (str): File encoding (default: utf-8)

    Returns:
        List[Dict[str, Any]]: List of dictionaries

    Raises:
        FileNotFoundError: If file does not exist
    """
    if not os.path.exists(file_path):
        logger.error(f" File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        with open(file_path, mode='r', encoding=encoding) as f:
            data = json.load(f)

        # Normalize to list
        if isinstance(data, list):
            result = data
        elif isinstance(data, dict):
            result = [data]
        else:
            logger.warning(f" Unexpected JSON type: {type(data)}, returning empty list")
            result = []

        logger.info(f" Loaded {len(result)} items from JSON: {file_path}")
        return result

    except json.JSONDecodeError as e:
        logger.error(f" JSON decode error: {e}")
        return []
    except Exception as e:
        logger.error(f" Error loading JSON: {e}")
        return []


def load_data(file_path: str, encoding: str = 'utf-8') -> List[Dict[str, Any]]:
    """
    Auto-detect and load data from CSV or JSON file.
    
    Args:
        file_path (str): File path (CSV or JSON)
        encoding (str): File encoding (default: utf-8)
    
    Returns:
        List[Dict[str, Any]]: List of dictionaries
    
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file format is not supported
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.csv':
        return load_csv(file_path, encoding)
    elif ext == '.json':
        return load_json(file_path, encoding)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .csv or .json")


# ==================== DATA WRITING ====================

def write_csv(file_path: str, data: List[Dict[str, Any]],
              encoding: str = 'utf-8', mode: str = 'w') -> bool:
    """
    Write list of dictionaries to CSV file.

    Args:
        file_path (str): CSV file path
        data (List[Dict[str, Any]]): Data to write
        encoding (str): File encoding
        mode (str): File write mode ('w' = overwrite, 'a' = append)

    Returns:
        bool: True if successful
    """
    if not data:
        logger.warning(" No data to write")
        return False

    try:
        # Ensure directory exists
        directory = os.path.dirname(file_path)
        if directory:
            ensure_directory(directory)

        # Write data
        with open(file_path, mode, newline='', encoding=encoding) as f:
            fieldnames = data[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)

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
    """
    Utility class for writing test/automation results to CSV.
    Supports buffering for performance optimization.
    """

    # Result statuses
    RESULT_OK = 'OK'
    RESULT_NG = 'NG'
    RESULT_SKIP = 'SKIP'
    RESULT_ERROR = 'ERROR'

    def __init__(self, output_path: str, auto_write: bool = False):
        """
        Initialize ResultWriter.

        Args:
            output_path (str): CSV output file path
            auto_write (bool): Auto-write after each add_result call
        """
        self.output_path = output_path
        self.auto_write = auto_write
        self.results: List[Dict[str, Any]] = []

        # Ensure output directory exists
        directory = os.path.dirname(output_path)
        if directory:
            ensure_directory(directory)

        logger.info(f"📝 ResultWriter initialized: {output_path}")

    def add_result(self, test_case: Dict[str, Any], result: str,
                  error_message: Optional[str] = None,
                  extra_fields: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a test case result.

        Args:
            test_case (Dict[str, Any]): Dictionary containing test case info
            result (str): Result (OK, NG, SKIP, ERROR)
            error_message (Optional[str]): Error message if any
            extra_fields (Optional[Dict[str, Any]]): Additional fields
        """
        row_data = test_case.copy()

        # Add standard fields
        row_data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        row_data['result'] = result

        if error_message:
            row_data['error_message'] = error_message

        # Add extra fields if provided
        if extra_fields:
            row_data.update(extra_fields)

        self.results.append(row_data)

        # Auto write if enabled
        if self.auto_write:
            self.write()

    def write(self, clear_after_write: bool = False) -> bool:
        """
        Write all results to CSV file.

        Args:
            clear_after_write (bool): Clear buffer after writing

        Returns:
            bool: True if successful
        """
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
        """
        Get results summary.

        Returns:
            Dict[str, int]: Dictionary containing count of each result type
        """
        summary = {
            self.RESULT_OK: 0,
            self.RESULT_NG: 0,
            self.RESULT_SKIP: 0,
            self.RESULT_ERROR: 0,
        }

        for row in self.results:
            result = row.get('result', self.RESULT_ERROR)
            if result in summary:
                summary[result] += 1
            else:
                # Unknown result type
                if result not in summary:
                    summary[result] = 0
                summary[result] += 1

        return summary

    def print_summary(self) -> None:
        """Print results summary to logger."""
        summary = self.get_summary()
        total = sum(summary.values())

        logger.info("=" * 60)
        logger.info("📊 TEST RESULTS SUMMARY")
        logger.info("=" * 60)

        if total == 0:
            logger.info("  No results recorded")
        else:
            for result_type, count in sorted(summary.items()):
                if count > 0:
                    percentage = (count / total * 100) if total > 0 else 0
                    logger.info(f"  {result_type:<10}: {count:>4} ({percentage:>5.1f}%)")

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
        return len(self.results) == 0