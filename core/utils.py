"""
Shared utility functions for the Auto C-Peach system.
"""

import os
import logging

logger = logging.getLogger(__name__)


# ==================== FILE & PATH UTILITIES ====================

def ensure_directory(path: str) -> None:
    """
    Đảm bảo thư mục tồn tại, tạo nếu chưa có.
    
    Args:
        path (str): Đường dẫn thư mục
    """
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"Đã tạo thư mục: {path}")


def get_logger(name: str) -> logging.Logger:
    """
    Lấy logger instance theo tên module.
    
    Args:
        name (str): Tên module (thường dùng __name__)
    
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)

