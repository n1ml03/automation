# YOLOv11 Integration Guide

## Tổng quan

Hệ thống đã được nâng cấp để sử dụng YOLOv11 cho việc phát hiện vật phẩm thay vì template matching thủ công. YOLO cung cấp:

- **Độ chính xác cao hơn**: Học từ data thật thay vì matching pixel
- **Tốc độ nhanh hơn**: Phát hiện tất cả items trong một lần forward pass
- **Linh hoạt hơn**: Có thể detect items ở các góc độ, kích thước khác nhau
- **Đếm số lượng tốt hơn**: Kết hợp YOLO detection với OCR cho kết quả chính xác

## Cài đặt

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

Hoặc cài đặt trực tiếp:

```bash
pip install ultralytics>=8.0.0
```

### 2. Cấu hình YOLO

Mở file `config.py` và điều chỉnh `YOLO_CONFIG`:

```python
YOLO_CONFIG = {
    "enabled": True,              # Bật/tắt YOLO detection
    "model_path": "yolo11n.pt",   # Đường dẫn model (hoặc pretrained)
    "confidence": 0.25,           # Ngưỡng độ tin cậy
    "iou_threshold": 0.45,        # IoU threshold cho NMS
    "device": "cpu",              # 'cpu', 'cuda', hoặc 'mps' (Mac)
    "imgsz": 640,                 # Kích thước ảnh đầu vào
    "max_det": 300,               # Số detection tối đa
    "quantity_offset": {          # Vị trí OCR số lượng từ item
        "x": 30,
        "y": 0,
        "width": 80,
        "height": 30
    },
    "fallback_to_template": False  # Fallback về template matching nếu YOLO lỗi
}
```

## Sử dụng

### Option 1: Sử dụng Pretrained Model

Nếu bạn có model YOLO đã train sẵn cho game items:

```python
from agent import Agent
from extractor import Extractor

# Khởi tạo
agent = Agent()
config_overrides = {
    'yolo': {
        'model_path': 'path/to/your/trained_model.pt',
        'enabled': True
    }
}

extractor = Extractor(agent, 'templates', config_overrides)

# Sử dụng
import cv2
image = cv2.imread('screenshot.png')
items = extractor.extract(image)

for item in items:
    print(f"{item['item']}: {item['quantity']} (confidence: {item['confidence']:.2f})")
```

### Option 2: Train Custom Model

Nếu bạn chưa có model, bạn cần train model từ templates và game screenshots.

#### Bước 1: Chuẩn bị data

Có 2 cách:

**Cách 1: Sử dụng synthetic data (nhanh, dễ)**

```bash
# Chuẩn bị một số background images (game screenshots không có items)
python train_yolo.py \
    --templates-dir templates \
    --output-dir yolo_dataset \
    --mode prepare \
    --backgrounds screenshot1.png screenshot2.png screenshot3.png \
    --num-samples 200
```

**Cách 2: Annotate thủ công (chính xác hơn)**

1. Sử dụng [LabelImg](https://github.com/tzutalin/labelImg) để annotate game screenshots
2. Lưu annotations dưới dạng YOLO format
3. Tổ chức thư mục:
   ```
   yolo_dataset/
   ├── train/
   │   ├── images/
   │   │   ├── img001.jpg
   │   │   └── ...
   │   └── labels/
   │       ├── img001.txt
   │       └── ...
   ├── val/
   │   ├── images/
   │   └── labels/
   └── data.yaml
   ```

#### Bước 2: Train model

```bash
python train_yolo.py \
    --templates-dir templates \
    --output-dir yolo_dataset \
    --mode train \
    --model-size n \
    --epochs 100 \
    --batch 16
```

**Các model sizes:**
- `n` (nano): Nhanh nhất, nhẹ nhất, độ chính xác thấp nhất
- `s` (small): Cân bằng tốt
- `m` (medium): Chính xác hơn, chậm hơn
- `l` (large): Rất chính xác, cần GPU
- `x` (extra-large): Chính xác nhất, cần GPU mạnh

#### Bước 3: Sử dụng model đã train

Sau khi train xong, model sẽ được lưu tại `yolo_dataset/train/weights/best.pt`.

Cập nhật `config.py`:

```python
YOLO_CONFIG = {
    ...
    "model_path": "yolo_dataset/train/weights/best.pt",
    ...
}
```

### Option 3: Fallback Mode (Hybrid)

Nếu bạn muốn dùng YOLO nhưng vẫn giữ template matching làm backup:

```python
config_overrides = {
    'yolo': {
        'enabled': True,
        'fallback_to_template': True  # Tự động chuyển sang template nếu YOLO lỗi
    }
}
```

## Tối ưu hóa

### 1. Device Selection

Để tăng tốc độ detection:

```python
# Sử dụng GPU NVIDIA
YOLO_CONFIG['device'] = 'cuda'

# Sử dụng Apple Silicon GPU
YOLO_CONFIG['device'] = 'mps'

# Auto detect
YOLO_CONFIG['device'] = 'auto'
```

### 2. Confidence Threshold

Điều chỉnh `confidence` để cân bằng precision/recall:

- **Confidence thấp (0.1-0.3)**: Detect nhiều hơn, có thể có false positives
- **Confidence cao (0.5-0.8)**: Chỉ detect khi chắc chắn, có thể bỏ sót

### 3. Quantity OCR Offset

Điều chỉnh vị trí OCR số lượng theo layout game của bạn:

```python
YOLO_CONFIG['quantity_offset'] = {
    'x': 30,      # Khoảng cách từ cạnh phải item
    'y': 0,       # Khoảng cách từ cạnh dưới item
    'width': 80,  # Chiều rộng vùng OCR
    'height': 30  # Chiều cao vùng OCR
}
```

## So sánh YOLO vs Template Matching

| Tiêu chí | Template Matching | YOLOv11 |
|----------|------------------|---------|
| **Độ chính xác** | 70-85% | 90-98% |
| **Tốc độ** | Chậm (scan toàn bộ ảnh) | Nhanh (single pass) |
| **Setup** | Dễ (chỉ cần template images) | Cần training |
| **Linh hoạt** | Chỉ detect exact match | Detect nhiều variations |
| **Scale, rotation** | Không xử lý tốt | Xử lý tốt |
| **Lighting changes** | Nhạy cảm | Robust |

## Troubleshooting

### YOLO không detect được items

1. Kiểm tra confidence threshold (giảm xuống 0.1 để test)
2. Kiểm tra model đã train đúng classes chưa
3. Xem visualization: `results[0].plot()` để debug
4. Enable fallback mode để so sánh với template matching

### OCR số lượng không chính xác

1. Điều chỉnh `quantity_offset` trong config
2. Kiểm tra resolution ảnh (nên >= 720p)
3. Tăng contrast/brightness preprocessing nếu cần

### Training quá lâu

1. Giảm `epochs` (50-100 là đủ cho most cases)
2. Giảm `batch` size nếu out of memory
3. Sử dụng model size nhỏ hơn (`n` thay vì `s`)
4. Giảm số lượng training samples

### Model quá lớn

- Sử dụng `yolo11n.pt` (nano) thay vì `yolo11s.pt`
- Export sang ONNX hoặc TensorRT để tối ưu

## Advanced Usage

### Export Model cho Production

```python
from ultralytics import YOLO

model = YOLO('yolo_dataset/train/weights/best.pt')

# Export sang ONNX (faster inference)
model.export(format='onnx')

# Export sang TensorRT (NVIDIA GPU)
model.export(format='engine')
```

### Batch Processing

```python
# Process nhiều images cùng lúc
images = [cv2.imread(f) for f in image_files]
results = extractor.yolo_model.predict(images, batch=True)
```

### Custom Training với Augmentation

Thêm augmentation vào `train_yolo.py`:

```python
# Trong file train_yolo.py, modify train_model():
results = model.train(
    data=data_yaml,
    epochs=epochs,
    imgsz=imgsz,
    batch=batch,
    # Augmentation parameters
    augment=True,
    degrees=15,           # Rotation
    translate=0.1,        # Translation
    scale=0.5,           # Scale
    flipud=0.5,          # Flip up-down
    fliplr=0.5,          # Flip left-right
    mosaic=1.0,          # Mosaic augmentation
    mixup=0.1            # Mixup augmentation
)
```

## Kết luận

YOLOv11 integration mang lại improvement đáng kể cho item detection. Nếu bạn có đủ data để train, nó sẽ tốt hơn template matching rất nhiều. Nếu không, bạn vẫn có thể dùng pretrained models hoặc fallback về template matching.

Để bắt đầu nhanh, thử với pretrained model `yolo11n.pt` trước, sau đó fine-tune với data của game bạn.

