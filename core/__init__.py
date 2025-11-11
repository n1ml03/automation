"""
Core module - Device interaction, OCR, detection and automation logic
"""

from .agent import Agent, EnhancedOcrEngine
from .utils import get_logger, ensure_directory
from .data import load_csv, load_json, load_data, ResultWriter
from .base import BaseAutomation
from .config import (
    FESTIVAL_CONFIG,
    GACHA_CONFIG,
    HOPPING_CONFIG,
    get_festival_config,
    get_gacha_config,
    get_hopping_config,
    merge_config
)

__all__ = [
    'Agent',
    'EnhancedOcrEngine',
    'BaseAutomation',
    'get_logger',
    'ensure_directory',
    'load_csv',
    'load_json',
    'load_data',
    'ResultWriter',
    'FESTIVAL_CONFIG',
    'GACHA_CONFIG',
    'HOPPING_CONFIG',
    'get_festival_config',
    'get_gacha_config',
    'get_hopping_config',
    'merge_config',
]

