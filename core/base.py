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


class BaseAutomation:
    """Base class for all automation modules with common functionality."""

    def __init__(self, agent: Agent, config: Dict[str, Any], roi_config_dict: Dict[str, Dict[str, Any]]):
        """
        Initialize base automation.

        Args:
            agent: Agent instance for device interaction
            config: Configuration dictionary
            roi_config_dict: ROI configuration dictionary for this automation type
        """
        self.agent = agent
        self.roi_config_dict = roi_config_dict

        # Extract common config
        self.templates_path = config['templates_path']
        self.snapshot_dir = config['snapshot_dir']
        self.results_dir = config['results_dir']
        self.wait_after_touch = config['wait_after_touch']

        # Ensure directories exist
        ensure_directory(self.snapshot_dir)
        ensure_directory(self.results_dir)

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
            template_path = os.path.join(self.templates_path, template_name)
            if not os.path.exists(template_path):
                return optional

            if self.agent.device is None:
                logger.error("Device not connected")
                return False

            template = Template(template_path)
            pos = exists(template)

            if pos:
                self.agent.safe_touch(pos)
                logger.info(f"✓ {template_name}")
                sleep(self.wait_after_touch)
                return True

            return optional if optional else False

        except Exception as e:
            logger.error(f"✗ {template_name}: {e}") if not optional else None
            return optional

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

    def ocr_roi(self, roi_name: str, screenshot: Optional[Any] = None,
                roi_config_getter: Optional[Callable] = None) -> str:
        """
        OCR specific ROI region using agent.ocr().

        Args:
            roi_name: ROI name
            screenshot: Screenshot for OCR, None = take new
            roi_config_getter: Function to get ROI config (optional)

        Returns:
            str: OCR text from ROI region (cleaned)
        """
        try:
            # Get ROI config - use getter if provided, otherwise use dict
            if roi_config_getter:
                roi_config = roi_config_getter(roi_name)
            elif roi_name in self.roi_config_dict:
                roi_config = self.roi_config_dict[roi_name]
            else:
                logger.error(f"ROI '{roi_name}' not found")
                return ""

            coords = roi_config['coords']  # [x1, x2, y1, y2]
            x1, x2, y1, y2 = coords
            region = (x1, y1, x2, y2)

            # Use agent's methods directly
            if screenshot is None:
                # Use agent.ocr() with region - more efficient
                ocr_result = self.agent.ocr(region)
            else:
                # Crop from existing screenshot and use agent's OCR engine
                roi_image = screenshot[y1:y2, x1:x2]
                if roi_image is None or roi_image.size == 0:
                    logger.warning(f"✗ ROI '{roi_name}': Invalid crop")
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
            # Get screenshot if not provided
            if screenshot is None:
                screenshot = self.agent.snapshot()
                if screenshot is None:
                    logger.error("Cannot get screenshot")
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

    def find_and_touch(self, search_text: str) -> bool:
        """
        Find text with OCR and touch.

        Args:
            search_text: Text to search and touch

        Returns:
            bool: True if successful
        """
        logger.info(f"Find & touch: {search_text}")
        ocr_results = self.snapshot_and_ocr()
        text_info = self.find_text(ocr_results, search_text)

        if text_info:
            logger.info(f"✓ Found: {search_text}")
            success = self.agent.safe_touch(text_info['center'])
            if success:
                sleep(self.wait_after_touch)
            return success

        logger.warning(f"✗ Not found: {search_text}")
        return False
