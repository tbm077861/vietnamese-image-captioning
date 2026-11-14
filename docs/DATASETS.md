# Dữ liệu

Khảo sát thực hiện ngày 13/11/2026. Mọi thông tin giấy phép dưới đây được đọc tại nguồn gốc
(README của tác giả, trang dataset, bài báo); bộ nào không có tuyên bố thì ghi rõ là không có.

## Bộ được chọn: UIT-OpenViIC

| | |
|---|---|
| Nguồn | https://github.com/nghiangh/UIT-OpenViIC-dataset (repo của tác giả) |
| Bài báo | Bui, Nguyen, Nguyen. *UIT-OpenViIC: An open-domain benchmark for evaluating image captioning in Vietnamese*. Signal Processing: Image Communication 140 (2026). https://doi.org/10.1016/j.image.2025.117430 |
| Nội dung | Ảnh đời sống thường ngày ở Việt Nam, miền mở; caption do người viết bằng tiếng Việt |
| Quy mô | 13.100 ảnh, 61.241 caption |
| Cách tải | Tải trực tiếp từ Google Drive của tác giả, không cần đăng ký hay ký thỏa thuận |

### Giấy phép

Bộ dữ liệu **không có file giấy phép chính thức**. Căn cứ sử dụng là tuyên bố của tác giả:

- README: *"The UIT-OpenViIC dataset is publicly available for research purposes and freely
  accessible"*.
- Bài báo (bản arXiv 2305.04166): *"available online for non-commercial research"*.

Vì vậy dự án chỉ dùng bộ này cho **nghiên cứu phi thương mại**, không phân phối lại và không
đưa dữ liệu lên repo. Ảnh do tác giả thu thập từ Google và Bing, bản quyền từng ảnh thuộc chủ
sở hữu gốc. Khi dùng kết quả từ bộ này, hãy trích dẫn bài báo ở trên.

### Số liệu sau khi tải (đã kiểm tra bằng `scripts/verify_data.py`)

| Split | Ảnh | Caption |
|---|---:|---:|
| train | 9.088 | 41.238 |
| val (file `dev` của tác giả) | 2.011 | 10.002 |
| test | 2.001 | 10.001 |
| **Tổng** | **13.100** | **61.241** |

- Số caption mỗi ảnh: 1 đến 6, trung bình 4,67.
- Độ dài caption (tính theo âm tiết): trung bình 17,6; trung vị 16; phân vị 95 là 29; dài nhất 127.
- 2 caption rỗng (bị bỏ ở bước tiền xử lý), 5 caption trùng lặp trong cùng một ảnh.
- Ảnh: rộng 400–6000 px, cao 86–4032 px; đa số JPEG, có cả PNG, WEBP và các chế độ màu
  RGBA / P / CMYK / L (đều được chuyển về RGB khi đọc).
- 4 ảnh bị thiếu vài byte cuối file (`00000007169`, `00000001010`, `00000005095`,
  `00000000768`). Chúng vẫn giải mã được nên được giữ lại để không làm thay đổi các split
  chính thức; script kiểm tra liệt kê chúng ở mục "truncated".
- File nén chứa thêm 200 ảnh không có caption; chúng không được dùng.

### Tải và kiểm tra

```powershell
python scripts/download_data.py --config configs/datasets/uit_openviic.yaml
python scripts/verify_data.py --config configs/datasets/uit_openviic.yaml
```

- Tải khoảng 3,8 GB về `data/raw/uit_openviic/`. Bị ngắt giữa chừng thì chạy lại lệnh, script
  tải tiếp từ chỗ dở. File nén bị xóa sau khi giải nén (thêm `--keep-archives` để giữ).
- Nếu Google Drive chặn do vượt hạn mức, script báo lỗi rõ ràng: khi đó tải tay các file từ
  repo của tác giả, đặt vào `data/raw/uit_openviic/` rồi chạy lại.
- Kết quả: `data/processed/uit_openviic/annotations.json` (định dạng thống nhất) và
  `verify_report.json`. Script kiểm tra trả mã lỗi 1 nếu có ảnh thiếu, ảnh không đọc được hoặc
  ảnh không có caption.

## Các bộ đã khảo sát nhưng không dùng

| Bộ | Quy mô (ảnh / caption) | Caption | Giấy phép tại nguồn | Cách lấy | Lý do không dùng |
|---|---|---|---|---|---|
| KTVIC | 4.327 / 21.635 | Người viết | Không có tuyên bố nào | Tải trực tiếp (GitHub, Drive) | Không có giấy phép hay tuyên bố sử dụng; ảnh lấy từ UIT-EVJVQA chưa rõ điều khoản |
| UIT-ViIC | 3.850 / khoảng 19.250 | Người viết | Không có file giấy phép; trang của nhóm ghi "for research purposes" | Annotation tải trực tiếp, ảnh lấy từ MS COCO | Chỉ gồm ảnh thể thao với bóng, quy mô nhỏ. Là phương án dự phòng |
| VieCap4H | 10.068 / 11.563 | Người viết | Thỏa thuận: chỉ dùng cho nghiên cứu học thuật, phi thương mại | Phải có tài khoản và ký thỏa thuận | Phải ký thỏa thuận; miền y tế hẹp; khoảng 1 caption mỗi ảnh |
| Flickr30k bản tiếng Việt (Hugging Face) | 31.783 ảnh | Dịch, không rõ cách dịch | Thẻ `cc-by-4.0` do người tải lên gắn | Tải trực tiếp | Caption dịch chứ không phải viết bằng tiếng Việt; giấy phép chưa đối chiếu được với bản gốc |
| COCO 2017 bản tiếng Việt (Hugging Face) | Theo COCO 2017 | Dịch máy | `unknown` | Tải trực tiếp | Dịch máy, không rõ giấy phép |
| HVU_VIC | 29.970 / 29.970 | Sinh tự động | Thẻ `mit` | Tải trực tiếp | Caption do máy sinh |
| XM3600 (phần tiếng Việt) | 3.600 ảnh | Người viết | Annotation CC BY 4.0 | Tải trực tiếp | Quá nhỏ để huấn luyện; phù hợp làm tập đánh giá bổ sung về sau |

Không có bộ nào tốt hơn UIT-OpenViIC mà phải ký thỏa thuận. Bộ duy nhất cần thỏa thuận là
VieCap4H; nếu sau này muốn thử miền y tế:

1. Xem thể lệ và mẫu thỏa thuận tại https://vlsp.org.vn/vlsp2021/eval/vieCap4H.
2. Đăng ký tài khoản trên cổng thi của ban tổ chức (https://aihub.ml/competitions/40).
3. Gửi thỏa thuận đã ký cho ban tổ chức theo địa chỉ ghi trên trang VLSP và chờ cấp quyền tải.

## Thêm hoặc đổi dataset

Toàn bộ mã nguồn chỉ đọc một định dạng annotation thống nhất
(`data/processed/<tên>/annotations.json`, mô tả trong `src/vic/data/annotations.py`), nên:

1. Tạo `configs/datasets/<tên>.yaml`: tên, nguồn, giấy phép, danh sách file cần tải (`url` hoặc
   `gdrive_id`, kèm `bytes` để kiểm tra), thư mục ảnh và file annotation của từng split.
2. Chọn `format` có sẵn trong `src/vic/data/sources.py`:
   - `filename_captions`: mỗi split một JSON dạng `{"<file ảnh>": {"captions": [...]}}`;
   - `coco_captions`: JSON kiểu COCO (`images` + `annotations`), dùng cho UIT-ViIC, KTVIC.

   Nếu bố cục khác hẳn thì thêm một hàm adapter (khoảng 15 dòng) vào file đó.
3. Trong file cấu hình thí nghiệm, sửa dòng `dataset_config:` trỏ tới file YAML mới, rồi chạy
   lại các bước tải, kiểm tra, tiền xử lý.

Dataset không có sẵn split val / test thì bước tiền xử lý tự tách từ train theo tỉ lệ khai báo
trong mục `splits` của file YAML, cố định theo seed.
