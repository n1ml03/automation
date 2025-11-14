"""
Gacha Automation

Gacha flow:
1. touch(Template("tpl_gacha.png"))
2. touch(Template("tpl_single_pull.png")) or touch(Template("tpl_multi_pull.png"))
3. snapshot -> save to folder (gacha_01_before_pull.png)
4. touch(Template("tpl_confirm.png"))
5. touch(Template("tpl_skip.png")) (if animation exists)
6. snapshot -> save to folder (gacha_02_after_pull.png)
7. ROI scan -> check gacha results
8. touch(Template("tpl_ok.png")) to close
9. Repeat according to pull count
"""

from typing import Dict, Optional, Any
from datetime import datetime
from airtest.core.api import sleep

from core.base import BaseAutomation, CancellationError
from core.agent import Agent
from core.utils import get_logger
from core.data import ResultWriter, load_data
from core.config import (
    GACHA_ROI_CONFIG, get_gacha_config, merge_config
)
from core.detector import OCRTextProcessor

logger = get_logger(__name__)


class GachaAutomation(BaseAutomation):
    """Automate Gacha pulls."""

    def __init__(self, agent: Agent, config: Optional[Dict[str, Any]] = None, cancel_event=None):
        # Merge config: base config from GACHA_CONFIG + custom config
        base_config = get_gacha_config()
        cfg = merge_config(base_config, config) if config else base_config

        # Initialize base class with agent, config, ROI config, and cancellation event
        super().__init__(agent, cfg, GACHA_ROI_CONFIG, cancel_event=cancel_event)

        # Gacha-specific timing
        self.wait_after_pull = cfg['wait_after_pull']

        # Pull settings
        self.max_pulls = cfg['max_pulls']
        self.pull_type = cfg['pull_type']

        logger.info("GachaAutomation initialized")

    def process_gacha_result(self, scan_results: Dict[str, str]) -> Dict[str, Any]:
        """
        Process gacha scan results with OCR text processing.
        
        Args:
            scan_results: Raw OCR scan results from ROI
            
        Returns:
            Dict with processed rarity, character, and confidence info
        """
        result = {
            'rarity': 'Unknown',
            'character': 'Unknown',
            'rarity_confidence': 0.0,
            'character_confidence': 0.0,
            'raw_rarity': '',
            'raw_character': ''
        }
        
        try:
            # Process rarity (normalize text)
            raw_rarity = scan_results.get('rarity', '')
            if raw_rarity:
                result['raw_rarity'] = raw_rarity
                # Normalize rarity text
                normalized_rarity = OCRTextProcessor.normalize_text_for_comparison(raw_rarity)
                
                # Match against known rarities
                known_rarities = ['ssr', 'sr', 'r', 'n', '★★★★★', '★★★★', '★★★', '★★', '★']
                best_match = None
                best_similarity = 0.0
                
                for known in known_rarities:
                    normalized_known = OCRTextProcessor.normalize_text_for_comparison(known)
                    if normalized_rarity == normalized_known:
                        best_match = known.upper()
                        best_similarity = 1.0
                        break
                    elif normalized_known in normalized_rarity or normalized_rarity in normalized_known:
                        # Partial match
                        similarity = min(len(normalized_known), len(normalized_rarity)) / max(len(normalized_known), len(normalized_rarity))
                        if similarity > best_similarity:
                            best_match = known.upper()
                            best_similarity = similarity
                
                if best_match:
                    result['rarity'] = best_match
                    result['rarity_confidence'] = best_similarity
                else:
                    result['rarity'] = raw_rarity.upper()
                    result['rarity_confidence'] = 0.5
            
            # Process character name (clean OCR artifacts)
            raw_character = scan_results.get('character', '')
            if raw_character:
                result['raw_character'] = raw_character
                # Clean common OCR artifacts
                cleaned_character = raw_character.strip()
                # Remove extra spaces
                cleaned_character = ' '.join(cleaned_character.split())
                result['character'] = cleaned_character
                result['character_confidence'] = 0.8 if cleaned_character else 0.0
            
            logger.debug(f"Processed gacha result: {result['rarity']} - {result['character']} "
                        f"(rarity conf: {result['rarity_confidence']:.2f}, char conf: {result['character_confidence']:.2f})")
            
        except Exception as e:
            logger.error(f"Error processing gacha result: {e}")
        
        return result

    def validate_gacha_result(self, processed_result: Dict[str, Any],
                             expected_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate gacha result against expected data (if provided).
        
        Args:
            processed_result: Processed gacha result from process_gacha_result()
            expected_data: Optional expected data for validation
            
        Returns:
            Dict with validation status and details
        """
        validation = {
            'status': 'pass',
            'rarity_match': True,
            'character_match': True,
            'message': 'No validation data provided'
        }
        
        if not expected_data:
            return validation
        
        try:
            # Validate rarity
            if 'expected_rarity' in expected_data:
                expected_rarity = str(expected_data['expected_rarity']).upper()
                actual_rarity = processed_result['rarity'].upper()
                
                rarity_match = OCRTextProcessor.compare_with_template(
                    actual_rarity, expected_rarity, threshold=0.7
                )
                validation['rarity_match'] = rarity_match
                
                if not rarity_match:
                    validation['status'] = 'fail'
                    validation['message'] = f"Rarity mismatch: got {actual_rarity}, expected {expected_rarity}"
            
            # Validate character
            if 'expected_character' in expected_data:
                expected_char = str(expected_data['expected_character'])
                actual_char = processed_result['character']
                
                char_match = OCRTextProcessor.compare_with_template(
                    actual_char, expected_char, threshold=0.6
                )
                validation['character_match'] = char_match
                
                if not char_match:
                    validation['status'] = 'fail'
                    if validation['message'] == 'No validation data provided':
                        validation['message'] = f"Character mismatch: got {actual_char}, expected {expected_char}"
                    else:
                        validation['message'] += f"; Character mismatch: got {actual_char}, expected {expected_char}"
            
            if validation['status'] == 'pass':
                validation['message'] = 'All validations passed'
                
        except Exception as e:
            validation['status'] = 'error'
            validation['message'] = f"Validation error: {str(e)}"
        
        return validation

    def run_gacha_stage(self, pull_data: Dict[str, Any], pull_idx: int,
                      pull_type: str = "single") -> Dict[str, Any]:
        """Run gacha stage."""
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
            sleep(0.5)  # Wait for result

            # Step 6: Snapshot result
            logger.info("Step 6: Snapshot result")
            screenshot_result = self.snapshot_and_save(folder_name, f"{pull_idx:02d}_result.png")
            if screenshot_result is None:
                return result

            # Step 7: Scan result
            logger.info("Step 7: Scan result")
            scan_results = self.scan_screen_roi(screenshot_result)

            # Process results with OCR text processing
            processed_result = self.process_gacha_result(scan_results)
            
            # Validate against expected data (if provided in pull_data)
            validation_result = self.validate_gacha_result(processed_result, pull_data)

            result.update({
                'rarity': processed_result['rarity'],
                'character': processed_result['character'],
                'rarity_confidence': processed_result['rarity_confidence'],
                'character_confidence': processed_result['character_confidence'],
                'raw_rarity': processed_result['raw_rarity'],
                'raw_character': processed_result['raw_character'],
                'validation': validation_result,
                'scan_data': scan_results,
                'success': True
            })

            logger.info(f"Pull result: {processed_result['rarity']} - {processed_result['character']} "
                       f"(rarity conf: {processed_result['rarity_confidence']:.2f})")
            
            if validation_result['status'] != 'pass' and validation_result['message'] != 'No validation data provided':
                logger.warning(f"Validation: {validation_result['message']}")

            # Step 8: Close result
            logger.info("Step 8: Close result")
            self.touch_template("tpl_ok.png", optional=True)

            logger.info(f"{'='*50}\n✓ COMPLETED: Pull {pull_idx}\n{'='*50}")
            return result

        except Exception as e:
            logger.error(f"✗ Pull {pull_idx}: {e}")
            return result

    def run_all_pulls(self, data_path: Optional[str] = None, num_pulls: Optional[int] = None,
                      pull_type: str = "single", output_path: Optional[str] = None) -> bool:
        """
        Run all pulls.

        Args:
            data_path: Path to CSV/JSON file with test data (mode 1)
            num_pulls: Number of pulls to run (mode 2, if no data_path)
            pull_type: Pull type ('single' or 'multi', mode 2 only)
            output_path: Output result path (None = auto-generate)

        Returns:
            bool: True if successful
        """
        try:
            # Mode 1: Load from data file
            if data_path:
                test_data = load_data(data_path)
                if not test_data:
                    logger.error("✗ No data loaded")
                    return False

                # Setup output
                if output_path is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = f"{self.results_dir}/gacha_batch_{timestamp}.csv"

                result_writer = ResultWriter(output_path)
                logger.info(f"Batch sessions: {len(test_data)} | Output: {output_path}")

                all_success = True

                # Process each session from data
                for idx, session_data in enumerate(test_data, 1):
                    session_id = session_data.get('session_id', idx)
                    session_num_pulls = int(session_data.get('num_pulls', 1))
                    session_pull_type = session_data.get('pull_type', 'single')

                    # Override timing configs if provided
                    if 'wait_after_pull' in session_data:
                        self.wait_after_pull = float(session_data['wait_after_pull'])
                    if 'wait_after_touch' in session_data:
                        self.wait_after_touch = float(session_data['wait_after_touch'])

                    logger.info(f"\n{'='*60}\nSESSION {idx}/{len(test_data)}: {session_num_pulls} {session_pull_type} pull(s)\n{'='*60}")

                    # Run pulls for this session
                    session_start = datetime.now()
                    successful_pulls = 0
                    rarity_counts = {}

                    for pull_idx in range(1, session_num_pulls + 1):
                        try:
                            self.check_cancelled(f"pull {pull_idx}")
                        except CancellationError:
                            logger.info(f"Cancellation requested, stopping at pull {pull_idx}")
                            break
                        pull_result = self.run_gacha_stage({}, pull_idx, session_pull_type)
                        if pull_result['success']:
                            successful_pulls += 1
                            rarity = pull_result.get('rarity', 'Unknown')
                            rarity_counts[rarity] = rarity_counts.get(rarity, 0) + 1
                        sleep(0.5)

                    session_end = datetime.now()
                    session_duration = (session_end - session_start).total_seconds()
                    session_success = successful_pulls == session_num_pulls

                    if not session_success:
                        all_success = False

                    # Add session summary
                    result_writer.add_result(
                        test_case={
                            'session_id': session_id,
                            'pull_type': session_pull_type,
                            'num_pulls': session_num_pulls,
                            'successful_pulls': successful_pulls,
                            'failed_pulls': session_num_pulls - successful_pulls,
                            'success_rate': f"{(successful_pulls/session_num_pulls*100):.1f}%",
                            'duration_seconds': f"{session_duration:.1f}",
                            'rarity_distribution': str(rarity_counts),
                        },
                        result=ResultWriter.RESULT_OK if session_success else ResultWriter.RESULT_NG,
                        error_message=None if session_success else f"Only {successful_pulls}/{session_num_pulls} pulls succeeded"
                    )

                    logger.info(f"Session {idx} completed: {successful_pulls}/{session_num_pulls} successful")
                    logger.info(f"Rarity distribution: {rarity_counts}")
                    sleep(0.5)

                # Save results
                result_writer.write()
                result_writer.print_summary()
                return all_success

            # Mode 2: Direct num_pulls
            elif num_pulls:
                # Setup output
                if output_path is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = f"{self.results_dir}/gacha_results_{timestamp}.csv"

                result_writer = ResultWriter(output_path)
                logger.info(f"Gacha pulls: {num_pulls} {pull_type} | Output: {output_path}")

                all_results = []

                # Process each pull
                for idx in range(1, num_pulls + 1):
                    try:
                        self.check_cancelled(f"pull {idx}")
                    except CancellationError:
                        logger.info(f"Cancellation requested, stopping at pull {idx}")
                        break
                    pull_result = self.run_gacha_stage({}, idx, pull_type)
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

                    sleep(0.5)

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

            else:
                logger.error("✗ Either 'data_path' or 'num_pulls' must be provided")
                return False

        except Exception as e:
            logger.error(f"✗ Run all pulls: {e}")
            return False

    def run(self, config: Optional[Dict[str, Any]] = None, data_path: Optional[str] = None) -> bool:
        """
        Main entry point for Gacha automation.

        Supports 2 modes:
        1. Config mode: Pass config dict with num_pulls, pull_type
        2. Data mode: Pass data_path to CSV/JSON file

        Args:
            config: Configuration dict (mode 1)
            data_path: Path to CSV/JSON data file (mode 2)

        Returns:
            bool: True if successful

        Example usage:
            # Mode 1: Direct config
            gacha.run(config={'num_pulls': 10, 'pull_type': 'single'})

            # Mode 2: Load from file
            gacha.run(data_path='./data/gacha_tests.csv')
        """
        logger.info("="*60 + "\nGACHA AUTOMATION START\n" + "="*60)

        try:
            self.check_cancelled("before starting")
        except CancellationError:
            return False

        if not self.agent.is_device_connected():
            logger.error("✗ Device not connected")
            return False

        # Mode 2: Load from data file
        if data_path:
            success = self.run_all_pulls(data_path=data_path)
        # Mode 1: Direct config
        elif config:
            num_pulls = config.get('num_pulls', 1)
            pull_type = config.get('pull_type', 'single')
            success = self.run_all_pulls(num_pulls=num_pulls, pull_type=pull_type)
        else:
            logger.error("✗ Either 'config' or 'data_path' must be provided")
            return False

        logger.info("="*60 + f"\n{'✓ COMPLETED' if success else '✗ FAILED'}\n" + "="*60)
        return success
