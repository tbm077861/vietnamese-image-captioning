# Dữ liệu

Khảo sát thực hiện ngày 13/11/2026.

- **Bộ được chọn (UIT-OpenViIC) đã được xác minh trực tiếp tại nguồn:** README trong repo của
  tác giả, các link tải, và số liệu đếm lại sau khi tải.
- **Các bộ không dùng chưa được xác minh trực tiếp tại nguồn.** Thông tin về chúng đến từ một
  lượt khảo sát sơ bộ và chỉ để tham khảo; xem cảnh báo ở mục tương ứng.

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

- README trong repo của tác giả (đã đọc trực tiếp): *"The UIT-OpenViIC dataset is publicly
  available for research purposes and freely accessible"*.
- Bản arXiv của bài báo (2305.04166) được ghi nhận là có câu *"available online for
  non-commercial research"*; câu này lấy từ lượt khảo sát, chưa đối chiếu lại với bài báo.

Dự án chọn cách hiểu chặt hơn: chỉ dùng bộ này cho **nghiên cứu phi thương mại**, không phân
phối lại, không đưa dữ liệu lên repo, và không công bố mô hình đã huấn luyện từ nó. Bản quyền
từng ảnh thuộc chủ sở hữu gốc của ảnh. Khi dùng kết quả từ bộ này, hãy trích dẫn bài báo ở trên.

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

> **Chưa xác minh trực tiếp tại nguồn.** Mọi thông tin trong bảng dưới đây (quy mô, cách tạo
> caption, giấy phép, cách lấy) được ghi lại từ một lượt khảo sát sơ bộ và chưa được kiểm tra
> lại tại trang gốc của từng bộ. Cột giấy phép chỉ ghi điều lượt khảo sát tìm thấy, **không
> phải khẳng định** về tình trạng pháp lý của bộ dữ liệu. Trước khi dùng bất kỳ bộ nào trong
> bảng, phải tự đọc điều khoản tại nguồn của bộ đó.

| Bộ | Quy mô được ghi nhận (ảnh / caption) | Caption | Ghi nhận về giấy phép (chưa xác minh) | Cách lấy được ghi nhận | Lý do không dùng |
|---|---|---|---|---|---|
| KTVIC | 4.327 / 21.635 | Người viết | Chưa tìm thấy tuyên bố giấy phép | Tải trực tiếp | Chưa thấy căn cứ cho phép sử dụng; chưa rõ điều khoản của nguồn ảnh |
| UIT-ViIC | 3.850 / khoảng 19.250 | Người viết | Chưa tìm thấy file giấy phép; trang của nhóm tác giả có nhắc "for research purposes" | Annotation tải trực tiếp, ảnh lấy từ MS COCO | Chỉ gồm ảnh thể thao với bóng, quy mô nhỏ. Có thể là phương án dự phòng sau khi xác minh |
| VieCap4H | 10.068 / 11.563 | Người viết | Có vẻ yêu cầu thỏa thuận sử dụng cho nghiên cứu học thuật | Cần tài khoản và ký thỏa thuận | Phải ký thỏa thuận; miền y tế hẹp; khoảng 1 caption mỗi ảnh |
| Flickr30k bản tiếng Việt (Hugging Face) | 31.783 ảnh | Dịch, chưa rõ cách dịch | Thẻ giấy phép do người tải lên gắn, chưa đối chiếu với bản gốc | Tải trực tiếp | Caption dịch chứ không phải viết bằng tiếng Việt; giấy phép chưa rõ |
| COCO 2017 bản tiếng Việt (Hugging Face) | Theo COCO 2017 | Dịch máy | Chưa rõ | Tải trực tiếp | Caption dịch máy; giấy phép chưa rõ |
| HVU_VIC | 29.970 / 29.970 | Sinh tự động | Thẻ giấy phép do người tải lên gắn, chưa đối chiếu | Tải trực tiếp | Caption do máy sinh |
| XM3600 (phần tiếng Việt) | 3.600 ảnh | Người viết | Được ghi nhận là annotation theo CC BY 4.0, chưa đối chiếu | Tải trực tiếp | Quá nhỏ để huấn luyện; có thể làm tập đánh giá bổ sung sau khi xác minh |

Trong lượt khảo sát, không thấy bộ nào tốt hơn UIT-OpenViIC mà phải ký thỏa thuận. Bộ duy nhất
được ghi nhận là cần thỏa thuận là VieCap4H. Nếu sau này muốn thử miền y tế, các bước dưới đây
cũng **chưa được xác minh** và cần kiểm tra lại tại trang của ban tổ chức:

1. Xem thể lệ và mẫu thỏa thuận tại trang VLSP 2021 của cuộc thi
   (https://vlsp.org.vn/vlsp2021/eval/vieCap4H).
2. Đăng ký tài khoản trên cổng thi mà trang đó chỉ dẫn.
3. Gửi thỏa thuận đã ký cho ban tổ chức theo địa chỉ ghi trên trang đó và chờ cấp quyền tải.

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
