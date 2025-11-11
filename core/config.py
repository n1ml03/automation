"""
Configuration Module - System-wide configuration management

Centralized configuration for all automation modules:
- Festival Automation
- Gacha Automation
- Hopping Automation
- Detector (YOLO, Template Matching)
"""

from typing import Dict, Any
from .utils import get_logger

logger = get_logger(__name__)


# ==================== ROI CONFIGURATION ====================
# Region of Interest for data extraction areas
# Format: [x1, x2, y1, y2] - pixel coordinates on screen

FESTIVALS_ROI_CONFIG: Dict[str, Dict[str, Any]] = {
    "フェス名": {
        "coords": [784, 1296, 247, 759],
        "description": "Festival name/rank area"
    },
    "フェスランク": {
        "coords": [392, 904, 41, 86],
        "description": "Festival rank level area"
    },
    "勝利点数": {
        "coords": [1012, 1240, 41, 86],
        "description": "Victory points area"
    },
    "推奨ランク": {
        "coords": [1050, 1255, 108, 163],
        "description": "Recommended rank area"
    },
    "Sランクボーダー": {
        "coords": [1050, 1255, 170, 225],
        "description": "S rank border points area"
    },
    "初回クリア報酬": {
        "coords": [1050, 1255, 233, 330],
        "description": "First clear reward area"
    },
    "Sランク報酬": {
        "coords": [1050, 1255, 343, 440],
        "description": "S rank reward area"
    },
    "獲得ザックマネー": {
        "coords": [392, 904, 20, 60],
        "description": "Zack money earned area"
    },
    "獲得アイテム": {
        "coords": [392, 904, 70, 170],
        "description": "Items earned area"
    },
    "獲得EXP-Ace": {
        "coords": [950, 1040, 115, 155],
        "description": "EXP for Ace units area"
    },
    "獲得EXP-NonAce": {
        "coords": [950, 1040, 215, 255],
        "description": "EXP for non-Ace units area"
    },
    "エース": {
        "coords": [1085, 1175, 85, 175],
        "description": "Venus memory for Ace area"
    },
    "非エース": {
        "coords": [1085, 1175, 185, 275],
        "description": "Venus memory for non-Ace area"
    }
}


# ==================== GACHA ROI CONFIGURATION ====================
# ROI for Gacha automation
GACHA_ROI_CONFIG: Dict[str, Dict[str, Any]] = {
    "rarity": {
        "coords": [600, 800, 200, 250],
        "description": "Character rarity area (SSR, SR, R)"
    },
    "character": {
        "coords": [400, 900, 300, 400],
        "description": "Character name area"
    },
    "pull_count": {
        "coords": [100, 300, 50, 100],
        "description": "Pull counter area"
    },
}

# ==================== HOPPING ROI CONFIGURATION ====================
# ROI for Hopping automation
HOPPING_ROI_CONFIG: Dict[str, Dict[str, Any]] = {
    "world_name": {
        "coords": [50, 400, 20, 80],
        "description": "Current world name area"
    },
    "world_level": {
        "coords": [450, 650, 20, 80],
        "description": "World level area"
    },
    "hop_cooldown": {
        "coords": [800, 1000, 500, 550],
        "description": "Hop cooldown time area"
    },
}


# ==================== UTILITY FUNCTIONS ====================

def get_festivals_roi_config(festivals_roi_name: str) -> Dict[str, Any]:
    """
    Get FESTIVALS ROI configuration by name.

    Args:
        festivals_roi_name (str): FESTIVALS ROI name

    Returns:
        Dict[str, Any]: FESTIVALS ROI configuration with 'coords' and 'description' keys

    Raises:
        KeyError: If festivals_roi_name does not exist
    """
    if festivals_roi_name not in FESTIVALS_ROI_CONFIG:
        available = list(FESTIVALS_ROI_CONFIG.keys())
        raise KeyError(
            f"FESTIVALS ROI '{festivals_roi_name}' not found. Available ROIs: {available}"
        )
    return FESTIVALS_ROI_CONFIG[festivals_roi_name]


def get_gacha_roi_config(gacha_roi_name: str) -> Dict[str, Any]:
    """
    Get Gacha ROI configuration by name.

    Args:
        gacha_roi_name (str): Gacha ROI name

    Returns:
        Dict[str, Any]: Gacha ROI configuration with 'coords' and 'description' keys

    Raises:
        KeyError: If gacha_roi_name does not exist
    """
    if gacha_roi_name not in GACHA_ROI_CONFIG:
        available = list(GACHA_ROI_CONFIG.keys())
        raise KeyError(
            f"GACHA ROI '{gacha_roi_name}' not found. Available ROIs: {available}"
        )
    return GACHA_ROI_CONFIG[gacha_roi_name]


def get_hopping_roi_config(hopping_roi_name: str) -> Dict[str, Any]:
    """
    Get Hopping ROI configuration by name.

    Args:
        hopping_roi_name (str): Hopping ROI name

    Returns:
        Dict[str, Any]: Hopping ROI configuration with 'coords' and 'description' keys

    Raises:
        KeyError: If hopping_roi_name does not exist
    """
    if hopping_roi_name not in HOPPING_ROI_CONFIG:
        available = list(HOPPING_ROI_CONFIG.keys())
        raise KeyError(
            f"HOPPING ROI '{hopping_roi_name}' not found. Available ROIs: {available}"
        )
    return HOPPING_ROI_CONFIG[hopping_roi_name]


# ==================== DEFAULT PATHS ====================
# Default paths for the entire system

DEFAULT_PATHS: Dict[str, str] = {
    'templates': './templates',
    'results': './result',
    'snapshots': './result/snapshots',
    'logs': './logs',
}


# ==================== FESTIVAL CONFIGURATION ====================
# Configuration for Festival Automation

FESTIVAL_CONFIG: Dict[str, Any] = {
    # Paths
    'templates_path': DEFAULT_PATHS['templates'],
    'snapshot_dir': f"{DEFAULT_PATHS['results']}/festival/snapshots",
    'results_dir': f"{DEFAULT_PATHS['results']}/festival/results",
    
    # Timing
    'wait_after_touch': 1.0,
    
    # Detector settings
    'use_detector': True,  # Whether to use detector
    'detector_type': 'template',  # 'yolo', 'template', 'auto'
    
    # YOLO config
    'yolo_config': {
        'model_path': 'yolo11n.pt',
        'confidence': 0.25,
        'device': 'cpu'  # 'cpu', 'cuda', 'mps', 'auto'
    },
    
    # Template matching config
    'template_config': {
        'templates_dir': DEFAULT_PATHS['templates'],
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
# Configuration for Gacha Automation

GACHA_CONFIG: Dict[str, Any] = {
    # Paths
    'templates_path': DEFAULT_PATHS['templates'],
    'snapshot_dir': f"{DEFAULT_PATHS['results']}/gacha/snapshots",
    'results_dir': f"{DEFAULT_PATHS['results']}/gacha/results",
    
    # Timing
    'wait_after_touch': 1.0,
    'wait_after_pull': 2.0,
    
    # Pull settings
    'max_pulls': 10,
    'pull_type': 'single',  # 'single' or 'multi'
    
    # Detector settings
    'use_detector': False,  # Gacha usually doesn't need detector
    'detector_type': 'auto',
}


# ==================== HOPPING CONFIGURATION ====================
# Configuration for Hopping Automation

HOPPING_CONFIG: Dict[str, Any] = {
    # Paths
    'templates_path': DEFAULT_PATHS['templates'],
    'snapshot_dir': f"{DEFAULT_PATHS['results']}/hopping/snapshots",
    'results_dir': f"{DEFAULT_PATHS['results']}/hopping/results",
    
    # Timing
    'wait_after_touch': 1.0,
    'loading_wait': 5.0,  # Loading wait time when hopping
    'cooldown_wait': 3.0,  # Cooldown wait time
    
    # Hop settings
    'max_hops': 10,
    'retry_on_fail': True,
    'max_retries': 3,
    
    # Detector settings
    'use_detector': False,
    'detector_type': 'auto',
}


# ==================== DETECTOR CONFIGURATION ====================
# Common configuration for Detector (YOLO, Template Matching)

DETECTOR_CONFIG: Dict[str, Any] = {
    # YOLO settings
    'yolo': {
        'model_path': 'yolo11n.pt',  # Available: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
        'confidence': 0.25,  # Confidence threshold (0.0-1.0)
        'iou': 0.45,  # IoU threshold for NMS
        'imgsz': 640,  # Input image size
        'device': 'cpu',  # 'cpu', 'cuda', 'mps', 'auto'
    },
    
    # Template Matching settings
    'template': {
        'templates_dir': DEFAULT_PATHS['templates'],
        'threshold': 0.85,  # Matching threshold (0.0-1.0)
        'method': 'TM_CCOEFF_NORMED',  # Matching method
        'min_distance': 10,  # Minimum distance to remove duplicates
    },
    
    # Quantity extraction settings (for YOLO)
    'quantity_extraction': {
        'offset_x': 30,  # Offset X from right corner of bbox
        'offset_y': 0,  # Offset Y from bottom corner of bbox
        'roi_width': 80,  # OCR region width
        'roi_height': 30,  # OCR region height
    },
}


# ==================== UTILITY FUNCTIONS ====================

def get_festival_config() -> Dict[str, Any]:
    """
    Get Festival Automation configuration.

    Returns:
        Dict[str, Any]: Festival configuration
    """
    return FESTIVAL_CONFIG.copy()


def get_gacha_config() -> Dict[str, Any]:
    """
    Get Gacha Automation configuration.

    Returns:
        Dict[str, Any]: Gacha configuration
    """
    return GACHA_CONFIG.copy()


def get_hopping_config() -> Dict[str, Any]:
    """
    Get Hopping Automation configuration.

    Returns:
        Dict[str, Any]: Hopping configuration
    """
    return HOPPING_CONFIG.copy()


def get_detector_config(detector_type: str = 'yolo') -> Dict[str, Any]:
    """
    Get Detector configuration.

    Args:
        detector_type (str): Detector type ('yolo' or 'template')

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
    Merge custom config into base config (deep merge).

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


logger.info("Configuration module loaded successfully")
