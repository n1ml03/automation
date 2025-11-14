# Festival Automation với Detector Guide

## Tổng quan

`festivals.py` đã được nâng cấp để hỗ trợ **Detector (YOLO hoặc Template Matching)** kết hợp với **OCR** nhằm tăng độ chính xác trong việc verify ROI.

### Lợi ích của Detector

1. **Verify vị trí object**: Đảm bảo object có xuất hiện trong ROI không
2. **Detect quantity**: Tự động trích xuất quantity từ vùng gần object
3. **Double verification**: Kết hợp cả detection và OCR để verify chính xác hơn
4. **Flexible**: Có thể chọn YOLO, Template Matching, hoặc Auto

---

## Các chế độ hoạt động

### 1. OCR Only (Truyền thống)
Chỉ sử dụng OCR để đọc text trong ROI.

```python
config = {
    'use_detector': False  # Tắt detector
}
festival = FestivalAutomation(agent, config)
festival.run('festivals.json', use_detector=False)
```

### 2. YOLO Detector + OCR
Sử dụng YOLO model để detect objects + OCR để đọc text.

```python
config = {
    'use_detector': True,
    'detector_type': 'yolo',
    'yolo_config': {
        'model_path': 'yolo11n.pt',  # Hoặc custom model
        'confidence': 0.25,
        'device': 'cpu'  # 'cpu', 'cuda', 'mps', 'auto'
    }
}
festival = FestivalAutomation(agent, config)
festival.run('festivals.json', use_detector=True)
```

### 3. Template Matching + OCR
Sử dụng template matching để detect objects + OCR để đọc text.

```python
config = {
    'use_detector': True,
    'detector_type': 'template',
    'template_config': {
        'templates_dir': './templates',
        'threshold': 0.85
    }
}
festival = FestivalAutomation(agent, config)
festival.run('festivals.json', use_detector=True)
```

### 4. Auto Mode (Khuyến nghị)
Tự động chọn detector: ưu tiên YOLO nếu có, fallback về Template Matching.

```python
config = {
    'use_detector': True,
    'detector_type': 'auto'
}
festival = FestivalAutomation(agent, config)
festival.run('festivals.json', use_detector=True)
```

---

## API Methods mới

### 1. `detect_roi()`
Detect objects trong ROI (chỉ detection, không OCR).

```python
result = festival.detect_roi('獲得アイテム', screenshot)

# Result format:
{
    'roi_name': '獲得アイテム',
    'detected': True,                   # Có detect được không
    'detections': [                     # Danh sách objects detected
        {
            'item': 'item_name',
            'quantity': 100,            # YOLO tự trích xuất quantity
            'x': 100, 'y': 200,
            'confidence': 0.95,
            'ocr_text': 'x100'         # Text từ OCR bên trong detector
        }
    ],
    'detection_count': 1               # Số objects detected
}
```

### 2. `scan_rois_detector()`
Scan nhiều ROIs chỉ với detector (không OCR).

```python
results = festival.scan_rois_detector(
    screenshot=screenshot,
    roi_names=['獲得アイテム', '獲得ザックマネー']
)

# Results format:
{
    '獲得アイテム': {
        'detected': True,
        'detections': [...],
        'detection_count': 1
    },
    '獲得ザックマネー': {
        'detected': True,
        'detections': [...],
        'detection_count': 1
    }
}
```

### 3. `scan_rois_combined()`
Scan nhiều ROIs với cả OCR và detector (comprehensive mode).

```python
results = festival.scan_rois_combined(
    screenshot=screenshot,
    roi_names=['獲得アイテム', '獲得ザックマネー']
)

# Results format:
{
    '獲得アイテム': {
        'text': 'Item x100',           # Từ OCR
        'detected': True,              # Từ detector
        'detections': [...],
        'detection_count': 1
    },
    '獲得ザックマネー': {
        'text': '5000',
        'detected': True,
        'detections': [...],
        'detection_count': 1
    }
}
```

### 4. `compare_results()`
So sánh dữ liệu OCR/Detector với expected data từ CSV.
Hỗ trợ cả simple OCR format và detector format.

```python
# Với detector (trả về details)
is_ok, message, details = festival.compare_results(
    extracted_data=results,  # Từ scan_rois_combined() hoặc scan_screen_roi()
    expected_data=csv_data,
    return_details=True  # Default
)

# Chỉ OCR (không cần details)
is_ok, message, _ = festival.compare_results(
    extracted_data=ocr_results,
    expected_data=csv_data,
    return_details=False  # Nhanh hơn
)

# Details format:
{
    '獲得アイテム': {
        'status': 'match',              # 'match' hoặc 'mismatch'
        'extracted_text': 'Item x100',
        'expected': 'Item x100',
        'detected': True,
        'detection_count': 1,
        'quantity': 100                 # Từ detections list
    }
}
```

---

## Flow với Detector

### Pre-battle Check (Step 6)
```
1. Snapshot màn hình
2. Scan ROIs với scan_rois_combined():
   - ['勝利点数', '推奨ランク', 'Sランクボーダー', '初回クリア報酬', 'Sランク報酬']
3. Mỗi ROI:
   a. OCR để đọc text
   b. Detect objects (nếu có detector)
   c. Kết hợp cả 2 results
4. Compare với CSV (comprehensive verification)
5. Log results
```

### Post-battle Check (Step 13)
```
1. Snapshot kết quả
2. Scan ROIs với scan_rois_combined():
   - ['獲得ザックマネー', '獲得アイテム',
      '獲得EXP-Ace', '獲得EXP-NonAce', 'エース', '非エース']
3. Mỗi ROI:
   a. OCR để đọc text
   b. YOLO/Template detect objects (items, rewards) với quantity
   c. Kết hợp results (text + detections)
4. Compare với CSV (comprehensive verification)
5. Log results với detection info + quantity
```

---

## Log Output Examples

### Với Detector (scan_rois_combined):
```
✓ ROI '獲得アイテム': detected=True (2 objects)
ROI '獲得アイテム' detected 2 objects:
  - memory_fragment x100 (conf: 0.95)
  - bonus_item x5 (conf: 0.87)
Scanned 6 ROIs (combined OCR + detector)
Pre-battle check (with detector): ✓ 5/5 matched
```

### Detector only (scan_rois_detector):
```
✓ ROI '獲得アイテム': detected=True (2 objects)
Scanned 6 ROIs with detector
```

### Không Detector (OCR only):
```
✓ 獲得アイテム: 'Memory Fragment x100'
Scanned 5 ROIs
Pre-battle check: ✓ 5/5 matched
```

---

## Requirements

### YOLO Mode
```bash
pip install ultralytics torch
```

### Template Matching Mode
Không cần thêm dependencies (sử dụng OpenCV có sẵn).

---

## Performance Tips

1. **YOLO Device Selection**:
   - `'cpu'`: Chậm nhưng ổn định
   - `'cuda'`: Nhanh (cần NVIDIA GPU + CUDA)
   - `'mps'`: Nhanh (Mac M1/M2)
   - `'auto'`: Tự động chọn tốt nhất

2. **Template Matching**:
   - Nhanh hơn YOLO trên CPU
   - Cần prepare template images trước
   - Tốt cho objects có hình dạng cố định

3. **Confidence Threshold**:
   - YOLO: 0.25 (default) - có thể tăng lên 0.4-0.5 để giảm false positives
   - Template: 0.85 (default) - có thể giảm xuống 0.75-0.8 nếu matching khó

---

## Troubleshooting

### YOLO không khả dụng
```python
# Fallback về Template Matching
config = {
    'detector_type': 'auto'  # Tự động fallback
}
```

### Template matching không tìm thấy objects
- Kiểm tra templates_dir có đúng không
- Thử giảm threshold (0.75 - 0.8)
- Đảm bảo template images có format đúng (.png, .jpg)

### Detection chậm
- Sử dụng `use_detector=False` cho các ROI không cần detect
- Chỉ enable detector cho post-battle ROIs (items, rewards)
- Sử dụng smaller YOLO model (yolo11n.pt vs yolo11x.pt)

---

## Example: Sử dụng các modes khác nhau

### Mode 1: Detector only (nhanh nhất khi chỉ cần detect)
```python
# Chỉ detect objects, không OCR text
results = festival.scan_rois_detector(
    screenshot=screenshot,
    roi_names=['獲得アイテム', '獲得ザックマネー']
)
```

### Mode 2: Combined mode (comprehensive)
```python
# Cả OCR và detection để verify toàn diện
results = festival.scan_rois_combined(
    screenshot=screenshot,
    roi_names=['獲得アイテム', '獲得ザックマネー']
)
```

### Mode 3: OCR only (legacy)
```python
# Chỉ OCR, không detection
results = festival.scan_screen_roi(
    screenshot=screenshot,
    roi_names=['獲得アイテム', '獲得ザックマネー']
)
```

---

## Kết luận

### Cấu trúc mới (Refactored)

**Separation of Concerns:**
- `detect_roi()`: Chỉ detection (focused)
- `scan_rois_detector()`: Scan nhiều ROIs với detector (detector-only)
- `scan_rois_combined()`: Scan với cả OCR và detector (comprehensive)
- `ocr_roi()` (from base.py): Chỉ OCR (legacy)

**Lợi ích:**
- ✅ **Tập trung**: Mỗi function có 1 mục đích rõ ràng
- ✅ **Hiệu quả**: Không crop ROI nhiều lần, không OCR không cần thiết
- ✅ **Linh hoạt**: Chọn mode phù hợp (detector-only, combined, OCR-only)
- ✅ **Dễ maintain**: Code sạch hơn, ít duplicate
- ✅ **Performance**: YOLO detector đã có OCR riêng, không cần chạy OCR thêm

Khuyến nghị: 
- Sử dụng **`scan_rois_combined()`** cho verification toàn diện (Step 6, 13)
- Sử dụng **`scan_rois_detector()`** khi chỉ cần detect items
- Sử dụng **Auto mode** detector type để có trải nghiệm tốt nhất!

