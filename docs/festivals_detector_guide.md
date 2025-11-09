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

### 1. `detect_and_ocr_roi()`
Detect object trong ROI và OCR để lấy text/quantity.

```python
result = festival.detect_and_ocr_roi('獲得アイテム', screenshot)

# Result format:
{
    'roi_name': '獲得アイテム',
    'text': 'Item Name x100',          # Text từ OCR
    'detected': True,                   # Có detect được không
    'detections': [                     # Danh sách objects detected
        {
            'item': 'item_name',
            'quantity': 100,
            'x': 100, 'y': 200,
            'confidence': 0.95
        }
    ],
    'detection_count': 1,              # Số objects detected
    'has_quantity': True,              # Có quantity không
    'quantity': 100                    # Quantity value
}
```

### 2. `scan_screen_roi_with_detector()`
Scan nhiều ROIs với detector (version nâng cao của `scan_screen_roi`).

```python
results = festival.scan_screen_roi_with_detector(
    screenshot=screenshot,
    roi_names=['獲得アイテム', '獲得ザックマネー'],
    use_detector=True
)

# Results format:
{
    '獲得アイテム': {
        'text': 'Item x100',
        'detected': True,
        'detection_count': 1,
        'quantity': 100
    },
    '獲得ザックマネー': {
        'text': '5000',
        'detected': True,
        'detection_count': 1,
        'quantity': 5000
    }
}
```

### 3. `compare_with_csv_enhanced()`
So sánh dữ liệu với detection info (enhanced version).

```python
is_ok, message, details = festival.compare_with_csv_enhanced(
    extracted_data=results,  # Từ scan_screen_roi_with_detector
    expected_data=csv_data
)

# Details format:
{
    '獲得アイテム': {
        'status': 'match',              # 'match' hoặc 'mismatch'
        'extracted_text': 'Item x100',
        'expected': 'Item x100',
        'detected': True,
        'detection_count': 1,
        'quantity': 100
    }
}
```

---

## Flow với Detector

### Pre-battle Check (Step 6)
```
1. Snapshot màn hình
2. Scan ROIs với detector:
   - ['フェス名', 'フェスランク', '勝利点数', '推奨ランク',
      'Sランクボーダー', '初回クリア報酬', 'Sランク報酬']
3. Mỗi ROI:
   a. Crop vùng ROI
   b. YOLO/Template detect objects
   c. OCR để đọc text
   d. Trích xuất quantity nếu có
4. Compare với CSV (enhanced mode)
5. Log results với detection info
```

### Post-battle Check (Step 13)
```
1. Snapshot kết quả
2. Scan ROIs với detector:
   - ['獲得ザックマネー', '獲得アイテム',
      '獲得EXP-Ace', '獲得EXP-NonAce', 'エース', '非エース']
3. Mỗi ROI:
   a. Crop vùng ROI
   b. YOLO/Template detect objects (items, rewards)
   c. OCR để đọc text + quantity
   d. Extract quantity từ detector
4. Compare với CSV (enhanced mode)
5. Log results với detection info + quantity
```

---

## Log Output Examples

### Với Detector:
```
✓ ROI '獲得アイテム': text='Memory Fragment x100', detected=True (2 objects), quantity=100
ROI '獲得アイテム' detected 2 objects:
  - memory_fragment x100 (conf: 0.95)
  - bonus_item x5 (conf: 0.87)
Pre-battle check (with detector): ✓ 7/7 matched
```

### Không Detector:
```
✓ 獲得アイテム: 'Memory Fragment x100'
Pre-battle check: ✓ 7/7 matched
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

## Example: Hybrid Mode

Bạn có thể mix & match: OCR cho pre-battle, Detector cho post-battle.

```python
# Trong run_festival_stage, tùy chỉnh:
if use_detector and self.detector is not None:
    # Pre-battle: chỉ OCR (nhanh)
    extracted_before = self.scan_screen_roi(screenshot_after, pre_battle_rois)
    
    # Post-battle: detector + OCR (chính xác hơn cho items)
    extracted_after = self.scan_screen_roi_with_detector(screenshot_result, post_battle_rois)
```

---

## Kết luận

Detector + OCR mang lại:
- ✅ **Độ chính xác cao hơn** nhờ double verification
- ✅ **Quantity extraction** tự động từ detector
- ✅ **Flexible** với 3 chế độ: YOLO, Template, Auto
- ✅ **Backward compatible** với OCR-only mode

Khuyến nghị: Sử dụng **Auto mode** để có trải nghiệm tốt nhất!

