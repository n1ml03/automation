"""
Agent Module - Device interaction and OCR with enhanced performance.
"""
import time
import copy
import cv2
import numpy as np
import oneocr
from typing import Optional, Tuple, List, Union
from airtest.core.api import connect_device, touch
from airtest.core.error import AirtestError
from .utils import get_logger

# ==================== OCR ENGINE ENHANCEMENT ====================

class EnhancedOcrEngine(oneocr.OcrEngine):
    """Enhanced OCR engine with direct NumPy array processing for better performance."""
    
    def recognize(self, image_array: np.ndarray) -> dict:
        """Process image from NumPy array efficiently without encode/decode overhead."""
        if any(x < 50 or x > 10000 for x in image_array.shape[:2]):
            result = copy.deepcopy(self.empty_result)
            result['error'] = 'Unsupported image size'
            return result

        channels = image_array.shape[2] if len(image_array.shape) == 3 else 1

        if channels == 1:
            img_bgra = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGRA)
        elif channels == 3:
            img_bgra = cv2.cvtColor(image_array, cv2.COLOR_BGR2BGRA)
        else:
            img_bgra = image_array

        return self._process_image(
            cols=img_bgra.shape[1],
            rows=img_bgra.shape[0],
            step=img_bgra.shape[1] * 4,
            data=img_bgra.ctypes.data
        )

# ==================== AGENT CLASS ====================

class Agent:
    """Agent for device interaction via Airtest and OCR processing."""
    
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_DELAY = 1.0
    DEFAULT_TOUCH_TIMES = 1

    def __init__(self, device_url: str = "Windows:///?title_re=DOAX VenusVacation.*", 
                 enable_retry: bool = True, auto_connect: bool = True):
        """Initialize Agent with device connection and OCR engine."""
        self.logger = get_logger(__name__)
        self.device = None
        self.ocr_engine = None
        self._device_verified = False
        self.device_url = device_url

        try:
            self.ocr_engine = EnhancedOcrEngine()
            self.logger.info("Enhanced OCR engine initialized")

            if auto_connect:
                if enable_retry:
                    if not self.connect_device_with_retry(device_url):
                        raise RuntimeError(f"Cannot connect to device: {device_url}")
                else:
                    self.device = connect_device(device_url)
                    self.logger.info(f"Connected to device: {device_url}")
                    if not self._verify_device():
                        self.logger.warning("Device connected but verification failed")

        except Exception as e:
            self.logger.error(f"Agent initialization failed: {e}")
            raise RuntimeError(f"Agent initialization failed: {e}") from e

    def connect_device_with_retry(self, device_url: str = "Windows:///?title_re=DOAX",
                                  max_retries: int = DEFAULT_MAX_RETRIES,
                                  retry_delay: float = DEFAULT_RETRY_DELAY) -> bool:
        """Connect to device with automatic retry on failure."""
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Connecting to device (attempt {attempt + 1}/{max_retries})...")
                self.device = connect_device(device_url)

                if self._verify_device():
                    self.logger.info("Device connected and verified")
                    return True

                if attempt < max_retries - 1:
                    self.logger.warning("Verification failed, retrying...")
                    time.sleep(retry_delay)

            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Connection failed: {e}, retrying...")
                    time.sleep(retry_delay)
                else:
                    self.logger.error(f"Connection failed: {e}")

        self.logger.error(f"Failed to connect after {max_retries} attempts")
        return False

    def _verify_device(self) -> bool:
        """Verify device is working properly."""
        try:
            if self.device and hasattr(self.device, 'snapshot'):
                self._device_verified = True
                return True
            return False
        except (AirtestError, Exception):
            return False

    def is_device_connected(self) -> bool:
        """Check if device is connected and functional."""
        if not self.device:
            self._device_verified = False
            return False
        return self._device_verified or self._verify_device()

    def snapshot(self) -> Optional[np.ndarray]:
        """Capture full screen screenshot."""
        if not self.is_device_connected():
            self.logger.error("Device not connected")
            return None
            
        try:
            return self.device.snapshot()
        except Exception as e:
            self.logger.error(f"Snapshot failed: {e}")
            self._device_verified = False
            return None

    def snapshot_region(self, region: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Capture screenshot of specific region (x1, y1, x2, y2)."""
        x1, y1, x2, y2 = region
        if x1 >= x2 or y1 >= y2:
            self.logger.error(f"Invalid region: {region}")
            return None
        
        full_screenshot = self.snapshot()
        if full_screenshot is None:
            return None
        
        return full_screenshot[y1:y2, x1:x2]

    def ocr(self, region: Optional[Tuple[int, int, int, int]] = None) -> Optional[dict]:
        """Perform OCR on full screen or specified region."""
        if not self.ocr_engine:
            self.logger.error("OCR engine not initialized")
            return None

        image_array = self.snapshot_region(region) if region else self.snapshot()
        if image_array is None:
            self.logger.error("Failed to get screenshot for OCR")
            return None

        try:
            return self.ocr_engine.recognize(image_array)
        except Exception as e:
            self.logger.error(f"OCR recognition failed: {e}")
            return None

    def safe_touch(self, pos: Union[Tuple[float, float], List[float]],
                   times: int = DEFAULT_TOUCH_TIMES) -> bool:
        """Safely touch screen at position with error handling."""
        if not self.is_device_connected():
            self.logger.error("Device not connected")
            return False

        if isinstance(pos, list):
            pos = tuple[float, ...](pos)

        if not (isinstance(pos, tuple) and len(pos) == 2):
            self.logger.error(f"Invalid coordinates: {pos}")
            return False

        try:
            touch(pos, times=times)
            return True
        except Exception as e:
            self.logger.error(f"Touch failed at {pos}: {e}")
            self._device_verified = False
            return False