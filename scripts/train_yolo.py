"""YOLO model training utility for item detection."""

import cv2
import yaml
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional
from core.utils import get_logger
from ultralytics import YOLO  # type: ignore

logger = get_logger(__name__)


class YOLOTrainer:
    """Prepare data and train YOLO model for item detection."""

    def __init__(self, templates_dir: str, output_dir: str = "yolo_dataset"):
        self.templates_dir = Path(templates_dir)
        self.output_dir = Path(output_dir)
        self.classes = []
        
        if not self.templates_dir.exists():
            raise FileNotFoundError(f"Templates directory not found: {templates_dir}")

    def _create_dirs(self) -> None:
        """Create dataset directory structure."""
        for split in ['train', 'val', 'test']:
            for subdir in ['images', 'labels']:
                (self.output_dir / split / subdir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Dataset structure created at {self.output_dir}")
    
    def _load_classes(self) -> List[str]:
        """Load class names from templates directory."""
        self.classes = sorted(set(
            f.stem for f in self.templates_dir.glob("*") 
            if f.suffix.lower() in ['.png', '.jpg', '.jpeg']
        ))
        logger.info(f"Found {len(self.classes)} classes: {self.classes}")
        return self.classes
    
    def _load_templates(self) -> dict:
        """Load template images into memory."""
        templates = {}
        for class_name in self.classes:
            files = list(self.templates_dir.glob(f"{class_name}.*"))
            if files:
                img = cv2.imread(str(files[0]), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    templates[class_name] = img
        return templates

    def _paste_template(self, background: np.ndarray, template: np.ndarray, 
                       x: int, y: int) -> None:
        """Paste template onto background with alpha blending if available."""
        th, tw = template.shape[:2]
        if template.shape[2] == 4:  # Has alpha channel
            alpha = template[:, :, 3:4] / 255.0
            background[y:y+th, x:x+tw] = (
                alpha * template[:, :, :3] + (1 - alpha) * background[y:y+th, x:x+tw]
            )
        else:
            background[y:y+th, x:x+tw] = template[:, :, :3]

    def create_synthetic_data(self, background_images: List[str], 
                            num_samples: int = 100, 
                            max_items_per_image: int = 5) -> None:
        """Create synthetic training data by pasting templates onto backgrounds."""
        if not self.classes:
            self._load_classes()
        
        logger.info(f"Creating {num_samples} synthetic images...")
        templates = self._load_templates()
        
        for i in range(num_samples):
            bg_path = np.random.choice(background_images)
            background = cv2.imread(bg_path)
            
            if background is None:
                logger.warning(f"Cannot read background: {bg_path}")
                continue
            
            h, w = background.shape[:2]
            annotations = []
            num_items = np.random.randint(1, max_items_per_image + 1)
            
            for _ in range(num_items):
                class_name = np.random.choice(self.classes)
                class_id = self.classes.index(class_name)
                template = templates.get(class_name)
                
                if template is None:
                    continue
                
                th, tw = template.shape[:2]
                x = np.random.randint(0, max(1, w - tw))
                y = np.random.randint(0, max(1, h - th))
                
                self._paste_template(background, template, x, y)
                
                # YOLO format: class_id x_center y_center width height (normalized)
                x_center = (x + tw / 2) / w
                y_center = (y + th / 2) / h
                width_norm = tw / w
                height_norm = th / h
                
                annotations.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}"
                )
            
            # Save image and annotation
            split = 'train' if i < num_samples * 0.8 else 'val'
            img_path = self.output_dir / split / 'images' / f"synthetic_{i:04d}.jpg"
            label_path = self.output_dir / split / 'labels' / f"synthetic_{i:04d}.txt"
            
            cv2.imwrite(str(img_path), background)
            label_path.write_text('\n'.join(annotations))
        
        logger.info(f"Created {num_samples} synthetic images")
    
    def create_data_yaml(self, dataset_path: Optional[str] = None) -> str:
        """Create data.yaml config file for YOLO training."""
        if not self.classes:
            self._load_classes()
        
        data_yaml = {
            'path': dataset_path or str(self.output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.classes),
            'names': self.classes
        }
        
        yaml_path = self.output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        logger.info(f"Created data.yaml at {yaml_path}")
        return str(yaml_path)
    
    def train_model(self, model_size: str = 'n', epochs: int = 100, 
                   imgsz: int = 640, batch: int = 16, 
                   data_yaml: Optional[str] = None, device: str = 'cpu') -> str:
        """Train YOLO model."""
        if data_yaml is None:
            data_yaml = self.create_data_yaml()
        
        logger.info(f"Training YOLO11{model_size} model...")
        logger.info(f"Epochs: {epochs}, Image size: {imgsz}, Batch: {batch}, Device: {device}")
        
        model = YOLO(f'yolo11{model_size}.pt')
        
        model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=str(self.output_dir),
            name='train',
            patience=50,
            save=True,
            plots=True,
            device=device
        )
        
        best_model_path = self.output_dir / 'train' / 'weights' / 'best.pt'
        logger.info(f"Training complete! Model saved at: {best_model_path}")
        
        return str(best_model_path)
    
    def convert_labelimg_annotations(self, labelimg_dir: str) -> None:
        """Convert LabelImg XML annotations to YOLO format."""
        logger.info("Converting LabelImg annotations to YOLO format...")
        
        for xml_file in Path(labelimg_dir).glob("*.xml"):
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                size = root.find('size')
                if size is None:
                    continue
                    
                width_elem = size.find('width')
                height_elem = size.find('height')
                if width_elem is None or width_elem.text is None or height_elem is None or height_elem.text is None:
                    continue
                    
                img_width = int(width_elem.text)
                img_height = int(height_elem.text)
                
                annotations = []
                
                for obj in root.findall('object'):
                    name_elem = obj.find('name')
                    if name_elem is None or name_elem.text is None:
                        continue
                        
                    class_name = name_elem.text
                    
                    if class_name not in self.classes:
                        self.classes.append(class_name)
                    
                    class_id = self.classes.index(class_name)
                    
                    bbox = obj.find('bndbox')
                    if bbox is None:
                        continue
                        
                    xmin_elem = bbox.find('xmin')
                    ymin_elem = bbox.find('ymin')
                    xmax_elem = bbox.find('xmax')
                    ymax_elem = bbox.find('ymax')
                    
                    if (xmin_elem is None or xmin_elem.text is None or
                        ymin_elem is None or ymin_elem.text is None or
                        xmax_elem is None or xmax_elem.text is None or
                        ymax_elem is None or ymax_elem.text is None):
                        continue
                    
                    xmin = int(xmin_elem.text)
                    ymin = int(ymin_elem.text)
                    xmax = int(xmax_elem.text)
                    ymax = int(ymax_elem.text)
                    
                    # Convert to YOLO format
                    x_center = ((xmin + xmax) / 2) / img_width
                    y_center = ((ymin + ymax) / 2) / img_height
                    width = (xmax - xmin) / img_width
                    height = (ymax - ymin) / img_height
                    
                    annotations.append(
                        f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                    )
                
                if annotations:
                    xml_file.with_suffix('.txt').write_text('\n'.join(annotations))
                
            except Exception as e:
                logger.error(f"Error converting {xml_file}: {e}")
        
        logger.info("Annotation conversion complete")

    def prepare_dataset(self, backgrounds: Optional[List[str]] = None, 
                       num_samples: int = 100) -> None:
        """Prepare complete dataset."""
        logger.info("=== Preparing Dataset ===")
        self._create_dirs()
        self._load_classes()
        
        if backgrounds:
            self.create_synthetic_data(backgrounds, num_samples)
        else:
            logger.warning("No backgrounds provided. Add images/labels manually.")
        
        self.create_data_yaml()


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train YOLO model for item detection')
    parser.add_argument('--templates-dir', type=str, default='templates',
                       help='Directory containing template images')
    parser.add_argument('--output-dir', type=str, default='yolo_dataset',
                       help='Output directory for dataset')
    parser.add_argument('--mode', type=str, choices=['prepare', 'train', 'full'],
                       default='full', help='Operation mode')
    parser.add_argument('--backgrounds', type=str, nargs='+',
                       help='Background image paths for synthetic data')
    parser.add_argument('--num-samples', type=int, default=100,
                       help='Number of synthetic samples')
    parser.add_argument('--model-size', type=str, default='n',
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLO model size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda', 'mps'],
                       help='Training device')
    
    args = parser.parse_args()
    
    trainer = YOLOTrainer(args.templates_dir, args.output_dir)
    
    if args.mode in ['prepare', 'full']:
        trainer.prepare_dataset(args.backgrounds, args.num_samples)
    
    if args.mode in ['train', 'full']:
        logger.info("=== Training Model ===")
        model_path = trainer.train_model(
            model_size=args.model_size,
            epochs=args.epochs,
            batch=args.batch,
            device=args.device
        )
        logger.info(f"Model trained and saved at: {model_path}")
        logger.info(f"Update YOLO_CONFIG['model_path'] = '{model_path}' in config.py")


if __name__ == "__main__":
    main()
