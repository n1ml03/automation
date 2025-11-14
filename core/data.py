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
    """
    Write list of dictionaries to CSV file.
    
    Note: For incremental writes, always use mode='w' to rewrite entire file
    with all accumulated results. This ensures data consistency.
    """
    if not data:
        logger.warning(" No data to write")
        return False

    try:
        if directory := os.path.dirname(file_path):
            ensure_directory(directory)

        # Always write in 'w' mode to ensure complete file rewrite
        # This is safe because ResultWriter maintains all results in memory
        with open(file_path, 'w', newline='', encoding=encoding) as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

        logger.debug(f" Wrote {len(data)} rows to CSV: {file_path}")
        return True
    except Exception as e:
        logger.error(f" Error writing CSV: {e}")
        return False


# ==================== RESULT WRITER CLASS ====================

class ResultWriter:
    """Utility class for writing test/automation results to CSV with buffering and resume support."""

    RESULT_OK, RESULT_NG, RESULT_SKIP, RESULT_ERROR = 'OK', 'NG', 'SKIP', 'ERROR'

    def __init__(self, output_path: str, auto_write: bool = True, resume: bool = True, batch_size: int = 100):
        """
        Initialize ResultWriter with output path and auto-write option.
        
        Args:
            output_path: Path to output CSV file
            auto_write: Automatically write after each add_result (default: True for incremental saving)
            resume: Load existing results if file exists (default: True for resume support)
            batch_size: Number of results to accumulate before writing (default: 100 for memory efficiency)
        """
        self.output_path = output_path
        self.auto_write = auto_write
        self.batch_size = batch_size
        self.results: List[Dict[str, Any]] = []
        self.completed_test_ids: set = set()
        self._pending_writes = 0

        if directory := os.path.dirname(output_path):
            ensure_directory(directory)
        
        # Load existing results for resume support
        if resume and os.path.exists(output_path):
            self._load_existing_results()
        
        logger.info(f"ResultWriter initialized: {output_path} ({len(self.results)} existing results, batch_size={batch_size})")

    def _load_existing_results(self) -> None:
        """Load existing results from CSV file for resume support."""
        try:
            existing_data = load_csv(self.output_path)
            self.results = existing_data
            # Track completed test IDs
            for row in existing_data:
                test_id = row.get('test_case_id')
                if test_id:
                    self.completed_test_ids.add(str(test_id))
            logger.info(f"Loaded {len(self.results)} existing results, {len(self.completed_test_ids)} completed test cases")
        except Exception as e:
            logger.warning(f"Could not load existing results: {e}")

    def is_completed(self, test_case: Dict[str, Any]) -> bool:
        """Check if a test case has already been completed."""
        test_id = test_case.get('test_case_id')
        return str(test_id) in self.completed_test_ids if test_id else False

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
        
        # Track completed test ID
        test_id = test_case.get('test_case_id')
        if test_id:
            self.completed_test_ids.add(str(test_id))
        
        self.results.append(row_data)
        
        # Batch writes for memory efficiency
        if self.auto_write:
            self._pending_writes += 1
            # Write when batch size reached, or immediately for first result
            if self._pending_writes >= self.batch_size:
                self.write()
                self._pending_writes = 0
            elif self._pending_writes == 1:
                # Write first result immediately for responsiveness
                self.write()

    def write(self, clear_after_write: bool = False) -> bool:
        """Write all results to CSV file, optionally clear buffer after."""
        if not self.results:
            logger.warning(" No results to write")
            return False

        success = write_csv(self.output_path, self.results)
        if success and clear_after_write:
            self.clear()
        elif success:
            logger.debug(f"Incremental save: {len(self.results)} results written to {self.output_path}")
        return success
    
    def flush(self) -> bool:
        """Force write all results immediately (alias for write())."""
        return self.write()

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