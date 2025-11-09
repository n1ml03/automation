"""
Script tiện ích để train YOLO model từ các template images và annotations.
Hỗ trợ tạo dataset, augmentation, và training YOLO model.
"""

import os
import cv2
import yaml
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from utils import get_logger

logger = get_logger(__name__)

# Import YOLO
from ultralytics import YOLO


class YOLOTrainer:
    """Class để prepare data và train YOLO model cho item detection."""
    
    def __init__(self, templates_dir: str, output_dir: str = "yolo_dataset"):
        """
        Khởi tạo YOLOTrainer.
        
        Args:
            templates_dir (str): Thư mục chứa template images
            output_dir (str): Thư mục output cho dataset YOLO
        """
        self.templates_dir = Path(templates_dir)
        self.output_dir = Path(output_dir)
        self.classes = []
        
        if not self.templates_dir.exists():
            raise FileNotFoundError(f"Templates directory không tồn tại: {templates_dir}")
    
    def create_dataset_structure(self) -> None:
        """Tạo cấu trúc thư mục cho YOLO dataset."""
        logger.info("Tạo cấu trúc thư mục YOLO dataset...")
        
        # Tạo thư mục chính
        for split in ['train', 'val', 'test']:
            for subdir in ['images', 'labels']:
                path = self.output_dir / split / subdir
                path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Đã tạo cấu trúc dataset tại {self.output_dir}")
    
    def load_class_names(self) -> List[str]:
        """
        Load danh sách class names từ templates directory.
        
        Returns:
            List[str]: Danh sách tên classes
        """
        classes = []
        for file in sorted(self.templates_dir.glob("*")):
            if file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                class_name = file.stem
                if class_name not in classes:
                    classes.append(class_name)
        
        self.classes = classes
        logger.info(f"Tìm thấy {len(classes)} classes: {classes}")
        return classes
    
    def create_synthetic_data(self, 
                             background_images: List[str],
                             num_samples: int = 100,
                             max_items_per_image: int = 5) -> None:
        """
        Tạo synthetic training data bằng cách paste template lên background.
        
        Args:
            background_images (List[str]): Danh sách đường dẫn ảnh background
            num_samples (int): Số lượng ảnh synthetic cần tạo
            max_items_per_image (int): Số lượng items tối đa trên mỗi ảnh
        """
        if not self.classes:
            self.load_class_names()
        
        logger.info(f"Tạo {num_samples} synthetic images...")
        
        # Load templates
        templates = {}
        for class_name in self.classes:
            template_files = list(self.templates_dir.glob(f"{class_name}.*"))
            if template_files:
                template_img = cv2.imread(str(template_files[0]), cv2.IMREAD_UNCHANGED)
                templates[class_name] = template_img
        
        # Tạo synthetic data
        for i in range(num_samples):
            # Chọn random background
            bg_path = np.random.choice(background_images)
            background = cv2.imread(bg_path)
            
            if background is None:
                logger.warning(f"Không thể đọc background: {bg_path}")
                continue
            
            h, w = background.shape[:2]
            annotations = []
            
            # Paste random số lượng items
            num_items = np.random.randint(1, max_items_per_image + 1)
            
            for _ in range(num_items):
                # Chọn random class
                class_name = np.random.choice(self.classes)
                class_id = self.classes.index(class_name)
                template = templates[class_name]
                
                if template is None:
                    continue
                
                th, tw = template.shape[:2]
                
                # Random vị trí paste
                x = np.random.randint(0, max(1, w - tw))
                y = np.random.randint(0, max(1, h - th))
                
                # Paste template lên background
                if template.shape[2] == 4:  # Có alpha channel
                    alpha = template[:, :, 3] / 255.0
                    for c in range(3):
                        background[y:y+th, x:x+tw, c] = (
                            alpha * template[:, :, c] + 
                            (1 - alpha) * background[y:y+th, x:x+tw, c]
                        )
                else:
                    background[y:y+th, x:x+tw] = template
                
                # Tạo annotation YOLO format: class_id x_center y_center width height (normalized)
                x_center = (x + tw / 2) / w
                y_center = (y + th / 2) / h
                width_norm = tw / w
                height_norm = th / h
                
                annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}")
            
            # Lưu image và annotation
            split = 'train' if i < num_samples * 0.8 else 'val'
            
            img_path = self.output_dir / split / 'images' / f"synthetic_{i:04d}.jpg"
            label_path = self.output_dir / split / 'labels' / f"synthetic_{i:04d}.txt"
            
            cv2.imwrite(str(img_path), background)
            
            with open(label_path, 'w') as f:
                f.write('\n'.join(annotations))
        
        logger.info(f"Đã tạo {num_samples} synthetic images")
    
    def create_data_yaml(self, dataset_path: Optional[str] = None) -> str:
        """
        Tạo file data.yaml cho YOLO training.
        
        Args:
            dataset_path (Optional[str]): Đường dẫn dataset (default: output_dir)
        
        Returns:
            str: Đường dẫn file data.yaml
        """
        if not self.classes:
            self.load_class_names()
        
        if dataset_path is None:
            dataset_path = str(self.output_dir.absolute())
        
        data_yaml = {
            'path': dataset_path,
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.classes),
            'names': self.classes
        }
        
        yaml_path = self.output_dir / 'data.yaml'
        
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        logger.info(f"Đã tạo data.yaml tại {yaml_path}")
        return str(yaml_path)
    
    def train_model(self, 
                   model_size: str = 'n',
                   epochs: int = 100,
                   imgsz: int = 640,
                   batch: int = 16,
                   data_yaml: Optional[str] = None) -> str:
        """
        Train YOLO model.
        
        Args:
            model_size (str): Kích thước model ('n', 's', 'm', 'l', 'x')
            epochs (int): Số epochs
            imgsz (int): Kích thước ảnh đầu vào
            batch (int): Batch size
            data_yaml (Optional[str]): Đường dẫn data.yaml
        
        Returns:
            str: Đường dẫn model đã train
        """
        if data_yaml is None:
            data_yaml = self.create_data_yaml()
        
        logger.info(f"Bắt đầu training YOLO11{model_size} model...")
        logger.info(f"Epochs: {epochs}, Image size: {imgsz}, Batch: {batch}")
        
        # Khởi tạo model
        model = YOLO(f'yolo11{model_size}.pt')
        
        # Train
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=str(self.output_dir),
            name='train',
            patience=50,
            save=True,
            plots=True,
            device='cpu'  # Có thể đổi thành 'cuda' hoặc 'mps'
        )
        
        best_model_path = self.output_dir / 'train' / 'weights' / 'best.pt'
        logger.info(f"Training hoàn tất! Model được lưu tại: {best_model_path}")
        
        return str(best_model_path)
    
    def convert_annotations_from_labelimg(self, labelimg_dir: str) -> None:
        """
        Convert annotations từ LabelImg format sang YOLO format.
        
        Args:
            labelimg_dir (str): Thư mục chứa annotations từ LabelImg (XML format)
        """
        import xml.etree.ElementTree as ET
        
        logger.info("Đang convert annotations từ LabelImg format...")
        
        labelimg_path = Path(labelimg_dir)
        
        for xml_file in labelimg_path.glob("*.xml"):
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                # Lấy kích thước ảnh
                size = root.find('size')
                img_width = int(size.find('width').text)
                img_height = int(size.find('height').text)
                
                annotations = []
                
                # Parse objects
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    
                    if class_name not in self.classes:
                        self.classes.append(class_name)
                    
                    class_id = self.classes.index(class_name)
                    
                    bbox = obj.find('bndbox')
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)
                    
                    # Convert sang YOLO format
                    x_center = ((xmin + xmax) / 2) / img_width
                    y_center = ((ymin + ymax) / 2) / img_height
                    width = (xmax - xmin) / img_width
                    height = (ymax - ymin) / img_height
                    
                    annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                
                # Lưu annotation
                txt_file = xml_file.with_suffix('.txt')
                with open(txt_file, 'w') as f:
                    f.write('\n'.join(annotations))
                
            except Exception as e:
                logger.error(f"Lỗi convert {xml_file}: {e}")
        
        logger.info("Hoàn tất convert annotations")


def main():
    """Main function để demo usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train YOLO model cho item detection')
    parser.add_argument('--templates-dir', type=str, default='templates',
                       help='Thư mục chứa template images')
    parser.add_argument('--output-dir', type=str, default='yolo_dataset',
                       help='Thư mục output cho dataset')
    parser.add_argument('--mode', type=str, choices=['prepare', 'train', 'full'],
                       default='full', help='Mode: prepare dataset, train model, hoặc full pipeline')
    parser.add_argument('--backgrounds', type=str, nargs='+',
                       help='Danh sách đường dẫn background images cho synthetic data')
    parser.add_argument('--num-samples', type=int, default=100,
                       help='Số lượng synthetic samples')
    parser.add_argument('--model-size', type=str, default='n',
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='Kích thước YOLO model')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Số epochs training')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    
    args = parser.parse_args()
    
    trainer = YOLOTrainer(args.templates_dir, args.output_dir)
    
    if args.mode in ['prepare', 'full']:
        logger.info("=== Preparing dataset ===")
        trainer.create_dataset_structure()
        trainer.load_class_names()
        
        if args.backgrounds:
            trainer.create_synthetic_data(
                background_images=args.backgrounds,
                num_samples=args.num_samples
            )
        else:
            logger.warning("Không có background images. Bỏ qua tạo synthetic data.")
            logger.info("Bạn có thể tự thêm images và labels vào thư mục train/val/test")
        
        trainer.create_data_yaml()
    
    if args.mode in ['train', 'full']:
        logger.info("=== Training model ===")
        model_path = trainer.train_model(
            model_size=args.model_size,
            epochs=args.epochs,
            batch=args.batch
        )
        logger.info(f"Model đã được train và lưu tại: {model_path}")
        logger.info(f"Để sử dụng model này, cập nhật 'model_path' trong config.py: YOLO_CONFIG['model_path'] = '{model_path}'")


if __name__ == "__main__":
    main()

