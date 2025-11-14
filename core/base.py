"""
Base Automation Module - Common functionality for all automations.

This module provides shared methods used by Festival, Gacha, and Hopping automations.
"""

import os
import cv2
from typing import Dict, List, Optional, Any, Callable
from airtest.core.api import Template, exists, sleep

from .agent import Agent
from .utils import get_logger, ensure_directory


logger = get_logger(__name__)


class CancellationError(Exception):
    """Exception raised when automation is cancelled."""
    pass


class BaseAutomation:
    """Base class for all automation modules with common functionality."""

    def __init__(self, agent: Agent, config: Dict[str, Any], roi_config_dict: Dict[str, Dict[str, Any]], cancel_event=None):
        """
        Initialize base automation.

        Args:
            agent: Agent instance for device interaction
            config: Configuration dictionary
            roi_config_dict: ROI configuration dictionary for this automation type
            cancel_event: Optional threading.Event for cancellation checking
        """
        self.agent = agent
        self.roi_config_dict = roi_config_dict
        self.cancel_event = cancel_event

        # Extract common config
        self.templates_path = config['templates_path']
        self.snapshot_dir = config['snapshot_dir']
        self.results_dir = config['results_dir']
        self.wait_after_touch = config['wait_after_touch']

        # Ensure directories exist
        ensure_directory(self.snapshot_dir)
        ensure_directory(self.results_dir)
    
    def is_cancelled(self) -> bool:
        """Check if cancellation was requested."""
        return self.cancel_event is not None and self.cancel_event.is_set()
    
    def check_cancelled(self, context: str = ""):
        """Check cancellation and raise CancellationError if cancelled."""
        if self.is_cancelled():
            msg = f"Cancellation requested{f' during {context}' if context else ''}"
            logger.info(msg)
            raise CancellationError(msg)
    
    def retry_with_cancellation(self, func: Callable, max_retries: int, retry_delay: float = 1.0,
                                 step_name: str = "", *args, **kwargs):
        """
        Retry a function with cancellation checking.
        
        Args:
            func: Function to retry (should return truthy on success)
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries
            step_name: Name of step for logging
            *args, **kwargs: Arguments to pass to func
            
        Returns:
            Result from func if successful
            
        Raises:
            CancellationError: If cancellation is requested
        """
        retry_count = 0
        while retry_count < max_retries:
            self.check_cancelled(step_name)
            result = func(*args, **kwargs)
            if result:
                return result
            retry_count += 1
            if retry_count < max_retries:
                logger.warning(f"{step_name}: Attempt {retry_count}/{max_retries} failed, retrying...")
                sleep(retry_delay)
        logger.error(f"{step_name}: Failed after {max_retries} attempts")
        return None

    def touch_template(self, template_name: str, optional: bool = False) -> bool:
        """
        Touch template image.

        Args:
            template_name: Template filename
            optional: If True, return success even if template not found

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.check_cancelled(f"touch_template({template_name})")
            
            template_path = os.path.join(self.templates_path, template_name)
            if not os.path.exists(template_path):
                return optional

            if self.agent.device is None:
                logger.error("Device not connected")
                return False

            template = Template(template_path)
            pos = exists(template)

            if pos:
                self.check_cancelled(f"touch_template({template_name})")
                self.agent.safe_touch(pos)
                logger.info(f"✓ {template_name}")
                sleep(self.wait_after_touch)
                return True

            return optional if optional else False

        except CancellationError:
            return False
        except Exception as e:
            logger.error(f"✗ {template_name}: {e}") if not optional else None
            return optional

    def get_screenshot(self, screenshot: Optional[Any] = None) -> Optional[Any]:
        """
        Get screenshot (use cached or take new).
        Helper function to avoid duplicate screenshot logic.

        Args:
            screenshot: Cached screenshot or None to take new

        Returns:
            Optional[Any]: Screenshot or None if failed
        """
        if screenshot is not None:
            return screenshot
        
        screenshot = self.agent.snapshot()
        if screenshot is None:
            logger.error("Cannot get screenshot")
        return screenshot

    def get_roi_config(self, roi_name: str) -> Optional[Dict[str, Any]]:
        """
        Get ROI configuration.
        Helper function to avoid duplicate ROI config logic.

        Args:
            roi_name: ROI name

        Returns:
            Optional[Dict[str, Any]]: ROI config or None if not found
        """
        if roi_name not in self.roi_config_dict:
            logger.error(f"ROI '{roi_name}' not found")
            return None
        return self.roi_config_dict[roi_name]

    def crop_roi(self, screenshot: Any, roi_name: str) -> Optional[Any]:
        """
        Crop ROI from screenshot.
        Helper function to avoid duplicate crop logic.

        Args:
            screenshot: Screenshot to crop from
            roi_name: ROI name

        Returns:
            Optional[Any]: Cropped ROI image or None if failed
        """
        roi_config = self.get_roi_config(roi_name)
        if roi_config is None:
            return None

        coords = roi_config['coords']  # [x1, x2, y1, y2]
        x1, x2, y1, y2 = coords
        
        roi_image = screenshot[y1:y2, x1:x2]
        if roi_image is None or roi_image.size == 0:
            logger.warning(f"✗ ROI '{roi_name}': Invalid crop")
            return None
        
        return roi_image

    def snapshot_and_save(self, folder_name: str, filename: str) -> Optional[Any]:
        """
        Take screenshot and save to folder.

        Args:
            folder_name: Folder name under snapshot_dir
            filename: Filename to save

        Returns:
            Optional[Any]: Screenshot array or None if failed
        """
        try:
            screenshot = self.agent.snapshot()
            if screenshot is None:
                return None

            folder_path = os.path.join(self.snapshot_dir, folder_name)
            ensure_directory(folder_path)
            file_path = os.path.join(folder_path, filename)
            cv2.imwrite(file_path, screenshot)
            logger.info(f"✓ Saved: {filename}")
            return screenshot

        except Exception as e:
            logger.error(f"✗ Snapshot: {e}")
            return None

    def ocr_roi(self, roi_name: str, screenshot: Optional[Any] = None) -> str:
        """
        OCR specific ROI region using agent.ocr().

        Args:
            roi_name: ROI name
            screenshot: Screenshot for OCR, None = take new

        Returns:
            str: OCR text from ROI region (cleaned)
        """
        try:
            # Get ROI config using helper
            roi_config = self.get_roi_config(roi_name)
            if roi_config is None:
                return ""

            coords = roi_config['coords']  # [x1, x2, y1, y2]
            x1, x2, y1, y2 = coords
            region = (x1, y1, x2, y2)

            # Use agent's methods directly
            if screenshot is None:
                # Use agent.ocr() with region - more efficient
                ocr_result = self.agent.ocr(region)
            else:
                # Crop ROI using helper
                roi_image = self.crop_roi(screenshot, roi_name)
                if roi_image is None:
                    return ""
                
                if self.agent.ocr_engine is None:
                    logger.error("OCR engine not initialized")
                    return ""
                
                ocr_result = self.agent.ocr_engine.recognize(roi_image)

            if ocr_result is None:
                logger.warning(f"✗ ROI '{roi_name}': OCR failed")
                return ""

            text = ocr_result.get('text', '').strip()
            text = self._clean_ocr_text(text)

            logger.debug(f"ROI '{roi_name}': '{text}'")
            return text

        except Exception as e:
            logger.error(f"✗ OCR ROI '{roi_name}': {e}")
            return ""

    def _clean_ocr_text(self, text: str) -> str:
        """
        Clean OCR text (remove special chars, normalize).

        Args:
            text: Text to clean

        Returns:
            str: Cleaned text
        """
        if not text:
            return ""

        # Remove extra whitespace
        text = ' '.join(text.split())

        # Remove newlines
        text = text.replace('\n', ' ').replace('\r', '')

        return text.strip()

    def scan_screen_roi(self, screenshot: Optional[Any] = None,
                       roi_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Scan screen according to defined ROI regions.

        Args:
            screenshot: Screenshot to scan, None = take new
            roi_names: List of ROI names to scan, None = scan all

        Returns:
            Dict[str, Any]: Dictionary with ROI name as key, OCR text as value
        """
        try:
            # Get screenshot using helper
            screenshot = self.get_screenshot(screenshot)
            if screenshot is None:
                return {}

            # Determine ROI list to scan
            if roi_names is None:
                roi_names = list(self.roi_config_dict.keys())

            # OCR each ROI
            results = {}
            for roi_name in roi_names:
                try:
                    text = self.ocr_roi(roi_name, screenshot)
                    results[roi_name] = text
                    logger.debug(f"✓ {roi_name}: '{text}'")
                except Exception as e:
                    logger.warning(f"✗ {roi_name}: {e}")
                    results[roi_name] = ""

            logger.info(f"Scanned {len(results)} ROIs")
            return results

        except Exception as e:
            logger.error(f"✗ Scan screen ROI: {e}")
            return {}

    def snapshot_and_ocr(self) -> List[Dict[str, Any]]:
        """
        Take screenshot and OCR to get text + coordinates using agent.ocr().

        Returns:
            List[Dict[str, Any]]: List of detected text with positions
        """
        try:
            # Use agent.ocr() directly - more efficient
            ocr_result = self.agent.ocr()
            if ocr_result is None:
                logger.error("OCR failed")
                return []

            lines = ocr_result.get('lines', [])

            results = []
            for line in lines:
                text = line.get('text', '').strip()
                bbox = line.get('bounding_rect', {})
                if text and bbox:
                    center_x = (bbox.get('x1', 0) + bbox.get('x3', 0)) / 2
                    center_y = (bbox.get('y1', 0) + bbox.get('y3', 0)) / 2
                    results.append({'text': text, 'center': (center_x, center_y)})

            logger.info(f"OCR: {len(results)} texts")
            return results

        except Exception as e:
            logger.error(f"✗ OCR: {e}")
            return []

    def find_text(self, ocr_results: List[Dict[str, Any]], search_text: str) -> Optional[Dict[str, Any]]:
        """
        Find text in OCR results.

        Args:
            ocr_results: List of OCR results
            search_text: Text to search for

        Returns:
            Optional[Dict[str, Any]]: Found text info or None
        """
        search_lower = search_text.lower().strip()
        for result in ocr_results:
            if search_lower in result['text'].lower():
                return result
        return None

    def ocr_roi_with_lines(self, roi_name: str) -> List[Dict[str, Any]]:
        """
        OCR specific ROI region and return individual text lines with coordinates.
        
        This method is useful when you need to find and interact with specific text
        within a defined region, avoiding false positives from other screen areas.

        Args:
            roi_name: ROI name from roi_config_dict

        Returns:
            List[Dict[str, Any]]: List of detected text with absolute screen positions
                [{'text': str, 'center': (x, y)}, ...]
        """
        try:
            # Get ROI config using helper
            roi_config = self.get_roi_config(roi_name)
            if roi_config is None:
                return []

            coords = roi_config['coords']  # [x1, x2, y1, y2]
            x1, x2, y1, y2 = coords
            region = (x1, y1, x2, y2)

            # OCR the ROI region
            ocr_result = self.agent.ocr(region)
            if ocr_result is None:
                logger.warning(f"✗ ROI '{roi_name}': OCR failed")
                return []

            # Parse individual lines (similar to snapshot_and_ocr)
            lines = ocr_result.get('lines', [])
            results = []

            for line in lines:
                text = line.get('text', '').strip()
                bbox = line.get('bounding_rect', {})
                if text and bbox:
                    # Coordinates are relative to region, convert to absolute screen coordinates
                    center_x = (bbox.get('x1', 0) + bbox.get('x3', 0)) / 2 + x1
                    center_y = (bbox.get('y1', 0) + bbox.get('y3', 0)) / 2 + y1
                    results.append({'text': text, 'center': (center_x, center_y)})

            logger.debug(f"✓ ROI '{roi_name}': Found {len(results)} texts")
            return results

        except Exception as e:
            logger.error(f"✗ OCR ROI with lines '{roi_name}': {e}")
            return []

    def find_and_touch_in_roi(self, roi_name: str, search_text: str) -> bool:
        """
        Find text in specific ROI region and touch it.
        
        Searches for text only within the defined ROI area, reducing false 
        positives and improving accuracy by focusing on the relevant screen region.

        Args:
            roi_name: ROI name from roi_config_dict to search within
            search_text: Text to search and touch

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.check_cancelled(f"find_and_touch_in_roi({roi_name}, {search_text})")

            logger.info(f"Find & touch '{search_text}' in ROI '{roi_name}'")

            # OCR the ROI to get list of texts with coordinates
            ocr_results = self.ocr_roi_with_lines(roi_name)

            if not ocr_results:
                logger.warning(f"✗ No text found in ROI '{roi_name}'")
                return False

            self.check_cancelled(f"find_and_touch_in_roi({roi_name}, {search_text})")

            # Find the text in results
            text_info = self.find_text(ocr_results, search_text)

            if text_info:
                self.check_cancelled(f"find_and_touch_in_roi({roi_name}, {search_text})")
                logger.info(f"✓ Found '{search_text}' in ROI '{roi_name}' at {text_info['center']}")
                success = self.agent.safe_touch(text_info['center'])
                if success:
                    sleep(self.wait_after_touch)
                return success

            logger.warning(f"✗ Text '{search_text}' not found in ROI '{roi_name}'")
            return False

        except CancellationError:
            return False
        except Exception as e:
            logger.error(f"✗ Find and touch in ROI '{roi_name}': {e}")
            return False
