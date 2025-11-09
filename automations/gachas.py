"""
Gacha Automation

Flow cho Gacha:
1. touch(Template("tpl_gacha.png"))
2. touch(Template("tpl_single_pull.png")) hoặc touch(Template("tpl_multi_pull.png"))
3. snapshot -> lưu folder (gacha_01_before_pull.png)
4. touch(Template("tpl_confirm.png"))
5. touch(Template("tpl_skip.png")) (nếu có animation)
6. snapshot -> lưu folder (gacha_02_after_pull.png)
7. ROI scan -> kiểm tra kết quả gacha
8. touch(Template("tpl_ok.png")) để đóng
9. Lặp lại theo số lượng pulls
"""

import os
import time
import cv2
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from airtest.core.api import Template
from core.agent import Agent
from core.utils import get_logger, ensure_directory
from core.data import ResultWriter, load_json, load_csv
from core.config import (
    GACHA_ROI_CONFIG, get_gacha_roi_config,
    GACHA_CONFIG, get_gacha_config, merge_config
)

logger = get_logger(__name__)


class GachaAutomation:
    """Tự động hóa Gacha pulls."""

    def __init__(self, agent: Agent, config: Optional[Dict[str, Any]] = None):
        self.agent = agent
        
        # Merge config: base config từ GACHA_CONFIG + custom config
        base_config = get_gacha_config()
        cfg = merge_config(base_config, config) if config else base_config

        # Paths
        self.templates_path = cfg.get('templates_path')
        self.snapshot_dir = cfg.get('snapshot_dir')
        self.results_dir = cfg.get('results_dir')
        
        # Timing
        self.wait_after_touch = cfg.get('wait_after_touch')
        self.wait_after_pull = cfg.get('wait_after_pull', 2.0)
        
        # Pull settings
        self.max_pulls = cfg.get('max_pulls', 10)
        self.pull_type = cfg.get('pull_type', 'single')

        # Ensure directories exist
        ensure_directory(self.snapshot_dir)
        ensure_directory(self.results_dir)

        logger.info("GachaAutomation initialized")

    def touch_template(self, template_name: str, optional: bool = False) -> bool:
        """Touch vào template image."""
        try:
            template_path = os.path.join(self.templates_path, template_name)
            if not os.path.exists(template_path):
                return optional

            if self.agent.device is None:
                logger.error("Device not connected")
                return False

            template = Template(template_path)
            pos = self.agent.device.exists(template)

            if pos:
                self.agent.device.touch(pos)
                logger.info(f"✓ {template_name}")
                time.sleep(self.wait_after_touch)
                return True

            return optional if optional else False

        except Exception as e:
            logger.error(f"✗ {template_name}: {e}") if not optional else None
            return optional

    def snapshot_and_save(self, folder_name: str, filename: str) -> Optional[Any]:
        """Chụp màn hình và lưu vào folder."""
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
        OCR một vùng ROI cụ thể cho Gacha.

        Args:
            roi_name: Tên ROI trong GACHA_ROI_CONFIG
            screenshot: Screenshot để OCR, None = chụp mới

        Returns:
            str: Text được OCR từ vùng ROI (đã clean)
        """
        try:
            # Lấy config ROI
            roi_config = get_gacha_roi_config(roi_name)
            coords = roi_config['coords']  # [x1, x2, y1, y2]

            # Convert sang format (x1, y1, x2, y2) cho snapshot_region
            x1, x2, y1, y2 = coords
            region = (x1, y1, x2, y2)

            # Chụp hoặc crop vùng ROI
            if screenshot is None:
                roi_image = self.agent.snapshot_region(region)
            else:
                # Crop từ screenshot có sẵn
                roi_image = screenshot[y1:y2, x1:x2]

            if roi_image is None:
                logger.warning(f"✗ ROI '{roi_name}': Cannot get image")
                return ""

            # OCR vùng ROI
            if self.agent.ocr_engine is None:
                logger.error("OCR engine not initialized")
                return ""

            ocr_result = self.agent.ocr_engine.recognize_cv2(roi_image)
            text = ocr_result.get('text', '').strip()

            # Clean text
            text = self._clean_ocr_text(text)

            logger.debug(f"ROI '{roi_name}': '{text}'")
            return text

        except Exception as e:
            logger.error(f"✗ OCR ROI '{roi_name}': {e}")
            return ""

    def _clean_ocr_text(self, text: str) -> str:
        """
        Clean OCR text (loại bỏ ký tự lạ, normalize).

        Args:
            text: Text cần clean

        Returns:
            str: Text đã clean
        """
        if not text:
            return ""

        # Remove extra whitespace
        text = ' '.join(text.split())

        # Remove newlines
        text = text.replace('\n', ' ').replace('\r', '')

        return text.strip()

    def scan_gacha_result(self, screenshot: Optional[Any] = None,
                         roi_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Scan kết quả gacha theo các vùng ROI đã định nghĩa.

        Args:
            screenshot: Screenshot để scan, None = chụp mới
            roi_names: Danh sách tên ROI cần scan, None = scan tất cả

        Returns:
            Dict[str, Any]: Dictionary với key là tên ROI, value là text OCR được
        """
        try:
            # Lấy screenshot nếu chưa có
            if screenshot is None:
                screenshot = self.agent.snapshot()
                if screenshot is None:
                    logger.error("Cannot get screenshot")
                    return {}

            # Xác định danh sách ROI cần scan
            if roi_names is None:
                roi_names = list(GACHA_ROI_CONFIG.keys())

            # OCR từng ROI
            results = {}
            for roi_name in roi_names:
                try:
                    text = self.ocr_roi(roi_name, screenshot)
                    results[roi_name] = text
                    logger.debug(f"✓ {roi_name}: '{text}'")
                except Exception as e:
                    logger.warning(f"✗ {roi_name}: {e}")
                    results[roi_name] = ""

            logger.info(f"Scanned {len(results)} Gacha ROIs")
            return results

        except Exception as e:
            logger.error(f"✗ Scan gacha result: {e}")
            return {}

    def run_gacha_pull(self, pull_data: Dict[str, Any], pull_idx: int,
                      pull_type: str = "single") -> Dict[str, Any]:
        """Chạy một gacha pull."""
        logger.info(f"\n{'='*50}\nPULL {pull_idx}: {pull_type.upper()}\n{'='*50}")

        folder_name = f"gacha_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result = {
            'pull_idx': pull_idx,
            'pull_type': pull_type,
            'rarity': 'Unknown',
            'character': 'Unknown',
            'success': False
        }

        try:
            # Step 1: Touch Gacha
            logger.info("Step 1: Touch Gacha")
            if not self.touch_template("tpl_gacha.png"):
                return result

            # Step 2: Choose pull type
            logger.info(f"Step 2: Choose {pull_type} pull")
            template_name = "tpl_single_pull.png" if pull_type == "single" else "tpl_multi_pull.png"
            if not self.touch_template(template_name):
                return result

            # Step 3: Snapshot before pull
            logger.info("Step 3: Snapshot before pull")
            screenshot_before = self.snapshot_and_save(folder_name, f"{pull_idx:02d}_before.png")
            if screenshot_before is None:
                return result

            # Step 4: Confirm pull
            logger.info("Step 4: Confirm pull")
            if not self.touch_template("tpl_confirm.png"):
                return result

            # Step 5: Skip animation if exists
            logger.info("Step 5: Skip animation")
            self.touch_template("tpl_skip.png", optional=True)
            time.sleep(2.0)  # Wait for result

            # Step 6: Snapshot result
            logger.info("Step 6: Snapshot result")
            screenshot_result = self.snapshot_and_save(folder_name, f"{pull_idx:02d}_result.png")
            if screenshot_result is None:
                return result

            # Step 7: Scan result
            logger.info("Step 7: Scan result")
            scan_results = self.scan_gacha_result(screenshot_result)

            # Extract rarity and character info
            rarity = scan_results.get('rarity', '')
            character = scan_results.get('character', '')

            result.update({
                'rarity': rarity,
                'character': character,
                'scan_data': scan_results,
                'success': True
            })

            logger.info(f"Pull result: {rarity} - {character}")

            # Step 8: Close result
            logger.info("Step 8: Close result")
            self.touch_template("tpl_ok.png", optional=True)

            logger.info(f"{'='*50}\n✓ COMPLETED: Pull {pull_idx}\n{'='*50}")
            return result

        except Exception as e:
            logger.error(f"✗ Pull {pull_idx}: {e}")
            return result

    def run_multiple_pulls(self, num_pulls: int, pull_type: str = "single",
                          output_path: Optional[str] = None) -> bool:
        """Chạy nhiều gacha pulls."""
        try:
            # Setup output
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(self.results_dir, f"gacha_results_{timestamp}.csv")

            result_writer = ResultWriter(output_path)
            logger.info(f"Gacha pulls: {num_pulls} {pull_type} | Output: {output_path}")

            all_results = []

            # Process each pull
            for idx in range(1, num_pulls + 1):
                pull_result = self.run_gacha_pull({}, idx, pull_type)
                all_results.append(pull_result)

                # Add to result writer
                test_case = {
                    'pull_number': idx,
                    'pull_type': pull_type,
                    'rarity': pull_result.get('rarity', ''),
                    'character': pull_result.get('character', ''),
                }
                result_writer.add_result(test_case,
                                       ResultWriter.RESULT_OK if pull_result['success'] else ResultWriter.RESULT_NG,
                                       error_message=None if pull_result['success'] else "Pull failed")

                time.sleep(1.0)

            # Save results
            result_writer.write()
            result_writer.print_summary()

            # Summary statistics
            successful_pulls = [r for r in all_results if r['success']]
            rarity_counts = {}
            for r in successful_pulls:
                rarity = r.get('rarity', 'Unknown')
                rarity_counts[rarity] = rarity_counts.get(rarity, 0) + 1

            logger.info(f"\nGacha Summary:")
            logger.info(f"Total pulls: {num_pulls}")
            logger.info(f"Successful: {len(successful_pulls)}")
            logger.info(f"Rarity distribution: {rarity_counts}")

            return True

        except Exception as e:
            logger.error(f"✗ Multiple pulls: {e}")
            return False

    def run(self, config: Dict[str, Any]) -> bool:
        """Entry point chính cho Gacha automation."""
        logger.info("="*60 + "\nGACHA AUTOMATION START\n" + "="*60)

        if not self.agent.is_device_connected():
            logger.error("✗ Device not connected")
            return False

        num_pulls = config.get('num_pulls', 1)
        pull_type = config.get('pull_type', 'single')

        success = self.run_multiple_pulls(num_pulls, pull_type)
        logger.info("="*60 + f"\n{'✓ COMPLETED' if success else '✗ FAILED'}\n" + "="*60)
        return success


if __name__ == '__main__':
    from core.agent import Agent

    agent = Agent()
    config = {
        'templates_path': './templates',
        'snapshot_dir': './result/gacha/snapshots',
        'results_dir': './result/gacha/results',
        'wait_after_touch': 1.0,
    }

    gacha = GachaAutomation(agent, config)
    gacha_config = {
        'num_pulls': 10,
        'pull_type': 'single'
    }
    gacha.run(gacha_config)
