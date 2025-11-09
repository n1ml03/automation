"""
Configuration Module - Quản lý cấu hình toàn hệ thống

Centralized configuration cho tất cả automation modules:
- Festival Automation
- Gacha Automation  
- Hopping Automation
- Detector (YOLO, Template Matching)
"""

from typing import Dict, Any
from .utils import get_logger

logger = get_logger(__name__)


# ==================== ROI CONFIGURATION ====================
# Region of Interest cho các vùng trích xuất dữ liệu
# Format: [x1, x2, y1, y2] - tọa độ pixel trên màn hình

FESTIVALS_ROI_CONFIG: Dict[str, Dict[str, Any]] = {
    "フェス名": {
        "coords": [784, 1296, 247, 759],
        "description": "Vùng festival name/rank"
    },
    "フェスランク": {
        "coords": [392, 904, 41, 86],
        "description": "Vùng festival rank level"
    },
    "勝利点数": {
        "coords": [1012, 1240, 41, 86],
        "description": "Vùng victory points"
    },
    "推奨ランク": {
        "coords": [1050, 1255, 108, 163],
        "description": "Vùng recommended rank"
    },
    "Sランクボーダー": {
        "coords": [1050, 1255, 170, 225],
        "description": "Vùng S rank border points"
    },
    "初回クリア報酬": {
        "coords": [1050, 1255, 233, 330],
        "description": "Vùng first clear reward"
    },
    "Sランク報酬": {
        "coords": [1050, 1255, 343, 440],
        "description": "Vùng S rank reward"
    },
    "獲得ザックマネー": {
        "coords": [392, 904, 20, 60],
        "description": "Vùng zack money earned"
    },
    "獲得アイテム": {
        "coords": [392, 904, 70, 170],
        "description": "Vùng items earned"
    },
    "獲得EXP-Ace": {
        "coords": [950, 1040, 115, 155],
        "description": "Vùng EXP for Ace units"
    },
    "獲得EXP-NonAce": {
        "coords": [950, 1040, 215, 255],
        "description": "Vùng EXP for non-Ace units"
    },
    "エース": {
        "coords": [1085, 1175, 85, 175],
        "description": "Vùng Venus memory for Ace"
    },
    "非エース": {
        "coords": [1085, 1175, 185, 275],
        "description": "Vùng Venus memory for non-Ace"
    }
}


# ==================== GACHA ROI CONFIGURATION ====================
# ROI cho Gacha automation
GACHA_ROI_CONFIG: Dict[str, Dict[str, Any]] = {
    "rarity": {
        "coords": [600, 800, 200, 250],
        "description": "Vùng rarity của character (SSR, SR, R)"
    },
    "character": {
        "coords": [400, 900, 300, 400],
        "description": "Vùng tên character"
    },
    "pull_count": {
        "coords": [100, 300, 50, 100],
        "description": "Vùng số pull counter"
    },
}

# ==================== HOPPING ROI CONFIGURATION ====================
# ROI cho Hopping automation
HOPPING_ROI_CONFIG: Dict[str, Dict[str, Any]] = {
    "world_name": {
        "coords": [50, 400, 20, 80],
        "description": "Vùng tên world hiện tại"
    },
    "world_level": {
        "coords": [450, 650, 20, 80],
        "description": "Vùng level của world"
    },
    "hop_cooldown": {
        "coords": [800, 1000, 500, 550],
        "description": "Vùng thời gian cooldown hop"
    },
}


# ==================== UTILITY FUNCTIONS ====================

def get_festivals_roi_config(festivals_roi_name: str) -> Dict[str, Any]:
    """
    Lấy cấu hình FESTIVALS ROI theo tên.

    Args:
        festivals_roi_name (str): Tên FESTIVALS ROI

    Returns:
        Dict[str, Any]: Cấu hình FESTIVALS ROI với keys 'coords' và 'description'

    Raises:
        KeyError: Nếu festivals_roi_name không tồn tại
    """
    if festivals_roi_name not in FESTIVALS_ROI_CONFIG:
        available = list(FESTIVALS_ROI_CONFIG.keys())
        raise KeyError(
            f"FESTIVALS ROI '{festivals_roi_name}' not found. Available ROIs: {available}"
        )
    return FESTIVALS_ROI_CONFIG[festivals_roi_name]


def get_gacha_roi_config(gacha_roi_name: str) -> Dict[str, Any]:
    """
    Lấy cấu hình Gacha ROI theo tên.

    Args:
        gacha_roi_name (str): Tên Gacha ROI

    Returns:
        Dict[str, Any]: Cấu hình Gacha ROI với keys 'coords' và 'description'

    Raises:
        KeyError: Nếu gacha_roi_name không tồn tại
    """
    if gacha_roi_name not in GACHA_ROI_CONFIG:
        available = list(GACHA_ROI_CONFIG.keys())
        raise KeyError(
            f"GACHA ROI '{gacha_roi_name}' not found. Available ROIs: {available}"
        )
    return GACHA_ROI_CONFIG[gacha_roi_name]


def get_hopping_roi_config(hopping_roi_name: str) -> Dict[str, Any]:
    """
    Lấy cấu hình Hopping ROI theo tên.

    Args:
        hopping_roi_name (str): Tên Hopping ROI

    Returns:
        Dict[str, Any]: Cấu hình Hopping ROI với keys 'coords' và 'description'

    Raises:
        KeyError: Nếu hopping_roi_name không tồn tại
    """
    if hopping_roi_name not in HOPPING_ROI_CONFIG:
        available = list(HOPPING_ROI_CONFIG.keys())
        raise KeyError(
            f"HOPPING ROI '{hopping_roi_name}' not found. Available ROIs: {available}"
        )
    return HOPPING_ROI_CONFIG[hopping_roi_name]


# ==================== DEFAULT PATHS ====================
# Paths mặc định cho toàn hệ thống

DEFAULT_PATHS: Dict[str, str] = {
    'templates': './templates',
    'results': './result',
    'snapshots': './result/snapshots',
    'logs': './logs',
}


# ==================== FESTIVAL CONFIGURATION ====================
# Cấu hình cho Festival Automation

FESTIVAL_CONFIG: Dict[str, Any] = {
    # Paths
    'templates_path': './templates',
    'snapshot_dir': './result/festival/snapshots',
    'results_dir': './result/festival/results',
    
    # Timing
    'wait_after_touch': 1.0,
    
    # Detector settings
    'use_detector': True,  # Có sử dụng detector không
    'detector_type': 'template',  # 'yolo', 'template', 'auto'
    
    # YOLO config
    'yolo_config': {
        'model_path': 'yolo11n.pt',
        'confidence': 0.25,
        'device': 'cpu'  # 'cpu', 'cuda', 'mps', 'auto'
    },
    
    # Template matching config
    'template_config': {
        'templates_dir': './templates',
        'threshold': 0.85,
        'method': 'TM_CCOEFF_NORMED'
    },
    
    # ROI groups
    'pre_battle_rois': [
        'フェス名', 'フェスランク', '勝利点数', '推奨ランク',
        'Sランクボーダー', '初回クリア報酬', 'Sランク報酬'
    ],
    'post_battle_rois': [
        '獲得ザックマネー', '獲得アイテム',
        '獲得EXP-Ace', '獲得EXP-NonAce', 'エース', '非エース'
    ],
}


# ==================== GACHA CONFIGURATION ====================
# Cấu hình cho Gacha Automation

GACHA_CONFIG: Dict[str, Any] = {
    # Paths
    'templates_path': './templates',
    'snapshot_dir': './result/gacha/snapshots',
    'results_dir': './result/gacha/results',
    
    # Timing
    'wait_after_touch': 1.0,
    'wait_after_pull': 2.0,
    
    # Pull settings
    'max_pulls': 10,
    'pull_type': 'single',  # 'single' hoặc 'multi'
    
    # Detector settings
    'use_detector': False,  # Gacha thường không cần detector
    'detector_type': 'auto',
}


# ==================== HOPPING CONFIGURATION ====================
# Cấu hình cho Hopping Automation

HOPPING_CONFIG: Dict[str, Any] = {
    # Paths
    'templates_path': './templates',
    'snapshot_dir': './result/hopping/snapshots',
    'results_dir': './result/hopping/results',
    
    # Timing
    'wait_after_touch': 1.0,
    'loading_wait': 5.0,  # Thời gian chờ loading khi hop
    'cooldown_wait': 3.0,  # Thời gian chờ cooldown
    
    # Hop settings
    'max_hops': 10,
    'retry_on_fail': True,
    'max_retries': 3,
    
    # Detector settings
    'use_detector': False,
    'detector_type': 'auto',
}


# ==================== DETECTOR CONFIGURATION ====================
# Cấu hình chung cho Detector (YOLO, Template Matching)

DETECTOR_CONFIG: Dict[str, Any] = {
    # YOLO settings
    'yolo': {
        'model_path': 'yolo11n.pt',  # Có thể dùng: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
        'confidence': 0.25,  # Ngưỡng confidence (0.0-1.0)
        'iou': 0.45,  # IoU threshold cho NMS
        'imgsz': 640,  # Kích thước input image
        'device': 'cpu',  # 'cpu', 'cuda', 'mps', 'auto'
    },
    
    # Template Matching settings
    'template': {
        'templates_dir': './templates',
        'threshold': 0.85,  # Ngưỡng matching (0.0-1.0)
        'method': 'TM_CCOEFF_NORMED',  # Phương pháp matching
        'min_distance': 10,  # Khoảng cách tối thiểu để loại bỏ duplicate
    },
    
    # Quantity extraction settings (cho YOLO)
    'quantity_extraction': {
        'offset_x': 30,  # Offset X từ góc phải bbox
        'offset_y': 0,  # Offset Y từ góc dưới bbox
        'roi_width': 80,  # Chiều rộng vùng OCR
        'roi_height': 30,  # Chiều cao vùng OCR
    },
}


# ==================== UTILITY FUNCTIONS ====================

def get_festival_config() -> Dict[str, Any]:
    """
    Lấy cấu hình Festival Automation.
    
    Returns:
        Dict[str, Any]: Festival configuration
    """
    return FESTIVAL_CONFIG.copy()


def get_gacha_config() -> Dict[str, Any]:
    """
    Lấy cấu hình Gacha Automation.
    
    Returns:
        Dict[str, Any]: Gacha configuration
    """
    return GACHA_CONFIG.copy()


def get_hopping_config() -> Dict[str, Any]:
    """
    Lấy cấu hình Hopping Automation.
    
    Returns:
        Dict[str, Any]: Hopping configuration
    """
    return HOPPING_CONFIG.copy()


def get_detector_config(detector_type: str = 'yolo') -> Dict[str, Any]:
    """
    Lấy cấu hình Detector.
    
    Args:
        detector_type (str): Loại detector ('yolo' hoặc 'template')
        
    Returns:
        Dict[str, Any]: Detector configuration
    """
    if detector_type not in ['yolo', 'template']:
        logger.warning(f"Unknown detector type: {detector_type}, using 'yolo'")
        detector_type = 'yolo'
    
    config = DETECTOR_CONFIG[detector_type].copy()
    if detector_type == 'yolo':
        config['quantity_extraction'] = DETECTOR_CONFIG['quantity_extraction'].copy()
    
    return config


def merge_config(base_config: Dict[str, Any], custom_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge custom config vào base config (deep merge).
    
    Args:
        base_config: Base configuration
        custom_config: Custom configuration to override
        
    Returns:
        Dict[str, Any]: Merged configuration
    """
    result = base_config.copy()
    
    for key, value in custom_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Deep merge for nested dicts
            result[key] = merge_config(result[key], value)
        else:
            # Override value
            result[key] = value
    
    return result


logger.info("✅ Configuration module loaded successfully")
