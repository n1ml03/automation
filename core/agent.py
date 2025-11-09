"""
Agent Module - Tương tác với device và OCR
"""

import oneocr
import time
import numpy as np
from typing import Optional, Tuple, List, Union
from airtest.core.api import connect_device, touch
from airtest.core.error import AirtestError
from .utils import get_logger


class Agent:
    """
    Agent tương tác với device thông qua Airtest và xử lý OCR.
    """

    # ==================== CONSTANTS ====================
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_DELAY = 1.0
    DEFAULT_TOUCH_DURATION = 0.1

    def __init__(self, device_url: str = "Windows:///?title_re=DOAX", enable_retry: bool = True):
        """
        Khởi tạo Agent.

        Args:
            device_url (str): URL của device (Windows:///, Android:///, iOS:///)
            enable_retry (bool): Có retry khi kết nối thất bại không

        Raises:
            RuntimeError: Khi không thể khởi tạo OCR hoặc kết nối device
        """
        self.logger = get_logger(__name__)
        self.device = None
        self.ocr_engine = None
        self._device_verified = False  # Cache để tránh kiểm tra device nhiều lần

        try:
            # Khởi tạo OCR engine
            self.ocr_engine = oneocr.OcrEngine()
            self.logger.info("✅ OCR engine initialized")

            # Kết nối device
            if enable_retry:
                if not self.connect_device_with_retry(device_url):
                    raise RuntimeError(f"Cannot connect to device: {device_url}")
            else:
                self.device = connect_device(device_url)
                self.logger.info(f"✅ Connected to device: {device_url}")
                # Verify device ngay sau khi kết nối
                if not self._verify_device():
                    self.logger.warning("⚠️ Device connected but verification failed")

        except Exception as e:
            self.logger.error(f"❌ Agent initialization failed: {e}")
            raise RuntimeError(f"Agent initialization failed: {e}")

    # ==================== DEVICE CONNECTION ====================

    def connect_device_with_retry(self, device_url: str = "Windows:///?title_re=DOAX",
                                  max_retries: int = DEFAULT_MAX_RETRIES,
                                  retry_delay: float = DEFAULT_RETRY_DELAY) -> bool:
        """
        Kết nối device với retry logic.

        Args:
            device_url (str): URL device
            max_retries (int): Số lần retry tối đa
            retry_delay (float): Thời gian chờ giữa các lần retry (giây)

        Returns:
            bool: True nếu kết nối thành công
        """
        for attempt in range(max_retries):
            try:
                self.logger.info(f"🔄 Connecting to device (attempt {attempt + 1}/{max_retries})...")
                self.device = connect_device(device_url)
                
                # Verify device ngay sau khi kết nối
                if self._verify_device():
                    self.logger.info("✅ Device connected and verified")
                    return True
                    
                # Retry nếu verification fail
                if attempt < max_retries - 1:
                    self.logger.warning(f"⚠️ Verification failed, retrying...")
                    time.sleep(retry_delay)
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"⚠️ Connection failed: {e}, retrying...")
                    time.sleep(retry_delay)
                else:
                    self.logger.error(f"❌ Connection failed: {e}")

        self.logger.error(f"❌ Failed to connect after {max_retries} attempts")
        return False

    def _verify_device(self) -> bool:
        """
        Verify device internal - kiểm tra device có hoạt động không.
        
        Returns:
            bool: True nếu device hoạt động tốt
        """
        try:
            # Kiểm tra device object tồn tại và có các method cần thiết
            if self.device is None or not hasattr(self.device, 'snapshot'):
                return False

            # Kiểm tra UUID - nếu device không connected, UUID sẽ None/empty
            if not getattr(self.device, 'uuid', None):
                return False

            # Thử snapshot để đảm bảo device hoạt động (window chưa đóng)
            test_screenshot = self.device.snapshot()
            if test_screenshot is None or not hasattr(test_screenshot, 'shape'):
                return False

            # Device hoạt động tốt, cache kết quả
            self._device_verified = True
            return True

        except (AirtestError, Exception):
            # Bất kỳ lỗi nào cũng nghĩa là device không hoạt động
            return False

    def is_device_connected(self) -> bool:
        """
        Kiểm tra device có kết nối và hoạt động được không.

        Returns:
            bool: True nếu device đã kết nối và hoạt động được
        """
        if self.device is None:
            self._device_verified = False
            return False

        # Sử dụng cache nếu đã verify thành công trước đó
        if self._device_verified:
            return True

        # Chưa verify hoặc cache đã reset, verify lại
        return self._verify_device()

    # ==================== SCREENSHOT & OCR ====================

    def snapshot(self) -> Optional[np.ndarray]:
        """
        Chụp màn hình hiện tại.

        Returns:
            Optional[np.ndarray]: Ảnh BGR hoặc None nếu thất bại
        """
        if not self.is_device_connected():
            self.logger.error("❌ Device not connected")
            return None
            
        try:
            return self.device.snapshot()
        except Exception as e:
            self.logger.error(f"❌ Snapshot failed: {e}")
            self._device_verified = False  # Reset cache khi có lỗi
            return None

    def snapshot_region(self, region: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Chụp một vùng cụ thể trên màn hình.

        Args:
            region (Tuple[int, int, int, int]): Vùng (x1, y1, x2, y2)

        Returns:
            Optional[np.ndarray]: Ảnh vùng hoặc None nếu thất bại
        """
        full_screenshot = self.snapshot()
        if full_screenshot is None:
            return None

        x1, y1, x2, y2 = region
        
        # Validate region
        if x1 >= x2 or y1 >= y2:
            self.logger.error(f"❌ Invalid region: ({x1}, {y1}, {x2}, {y2})")
            return None
        
        return full_screenshot[y1:y2, x1:x2]

    # ==================== TOUCH & INPUT ====================

    def safe_touch(self, pos: Union[Tuple[float, float], List[float]],
                   duration: float = DEFAULT_TOUCH_DURATION) -> bool:
        """
        Touch an toàn với error handling.

        Args:
            pos (Tuple[float, float]): Vị trí touch (x, y)
            duration (float): Thời gian nhấn giữ (giây)

        Returns:
            bool: True nếu thành công
        """
        if not self.is_device_connected():
            self.logger.error("❌ Device not connected")
            return False

        # Convert list to tuple và validate
        if isinstance(pos, list):
            if len(pos) != 2:
                self.logger.error(f"❌ Invalid coordinates: must have 2 elements")
                return False
            pos = (float(pos[0]), float(pos[1]))
        elif len(pos) != 2:
            self.logger.error(f"❌ Invalid coordinates: must have 2 elements")
            return False

        try:
            touch(pos, duration=duration)
            return True
        except Exception as e:
            self.logger.error(f"❌ Touch failed at {pos}: {e}")
            self._device_verified = False  # Reset cache khi có lỗi
            return False

