# Hướng dẫn phát triển

Tài liệu này ghi lại bối cảnh, môi trường, quy tắc làm việc và trạng thái hiện tại để bất kỳ ai
tiếp tục dự án đều nắm được. Cập nhật mục "Trạng thái hiện tại" ở cuối mỗi giai đoạn.

## Bối cảnh

Xây dựng hệ thống image captioning tiếng Việt: ảnh vào, câu mô tả tiếng Việt ra. Sản phẩm cuối:
mô hình đã huấn luyện, đánh giá BLEU-1..4 / CIDEr / METEOR, script suy luận cho ảnh bất kỳ, demo
Gradio, README hoàn chỉnh. Lộ trình chi tiết: [ROADMAP.md](ROADMAP.md).

## Môi trường

- Windows 11, PowerShell 7, Python 3.10.11, Git, GitHub CLI.
- GPU NVIDIA RTX 5070 Laptop, 8 GB VRAM (kiến trúc Blackwell, `sm_120`). Chỉ các bản PyTorch build
  với CUDA 12.8 trở lên mới chạy được trên GPU này; dự án dùng bản CUDA 13.0 ghi trong
  `requirements.txt`.
- Hệ điều hành đã chiếm khoảng 1,5 GB VRAM, nên thiết kế batch size và mô hình cho khoảng 6 GB,
  luôn bật mixed precision.

## Quy tắc làm việc

1. Trước mỗi giai đoạn: trình bày kế hoạch và chờ chủ dự án duyệt.
2. Dùng môi trường ảo `.venv` trong thư mục dự án; mọi thư viện ghi kèm phiên bản trong
   `requirements.txt`.
3. Git:
   - `main` luôn chạy được. Mỗi giai đoạn làm trên nhánh riêng (ví dụ `feat/data-pipeline`).
   - Commit theo Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`, `test:`, `refactor:`).
   - Commit message chỉ gồm tiêu đề và nội dung, không thêm trailer hay chữ ký công cụ.
   - Xong giai đoạn thì mở Pull Request bằng `gh` và chờ chủ dự án duyệt merge.
   - Không force-push lên `main`, không xóa repo.
4. Không bao giờ commit dữ liệu ảnh, checkpoint, `.venv` hay bí mật (token, key).
5. Repo công khai chỉ chứa nội dung của dự án: không thông tin cá nhân, không email riêng.
6. Dataset phải có giấy phép cho phép nghiên cứu; ghi nguồn và giấy phép trong README.
7. Thí nghiệm phải tái lập được: seed cố định, cấu hình trong file YAML, có log kết quả.
8. Gặp lỗi thì tự chẩn đoán và sửa; chỉ hỏi chủ dự án khi cần quyết định.
9. Cuối mỗi giai đoạn viết báo cáo ngắn: đã làm gì, kết quả, vấn đề còn lại, bước tiếp theo.

Ngôn ngữ: tài liệu và trao đổi bằng tiếng Việt; code, tên biến, comment trong code bằng tiếng Anh.

## Tài khoản và remote

- Repo thuộc tài khoản GitHub `tbm077861`, remote dùng HTTPS:
  `https://tbm077861@github.com/tbm077861/vietnamese-image-captioning.git`.
- Danh tính commit đặt cục bộ cho repo (không đổi cấu hình global): `user.name=tbm077861`,
  email dạng `<ID>+tbm077861@users.noreply.github.com`.
- Trước khi push, kiểm tra tác giả của các commit mới: `git log --format="%an <%ae>"`.

## Lệnh thường dùng

```powershell
.\.venv\Scripts\Activate.ps1          # kích hoạt môi trường
pip install -r requirements.txt       # cài thư viện
pip install -e . --no-deps            # cài gói vic ở chế độ phát triển
python scripts/check_env.py           # kiểm tra PyTorch / CUDA / GPU
pytest                                # chạy test
ruff check . ; ruff format .          # lint và định dạng code
```

## Quy ước code

- Mã nguồn chính nằm trong gói `vic` (`src/vic/`); `scripts/` chỉ chứa điểm vào dòng lệnh mỏng.
- Định dạng và lint bằng `ruff` (dòng tối đa 100 ký tự), có type hint cho hàm công khai.
- Mọi tham số thí nghiệm nằm trong `configs/*.yaml`, không viết cứng trong code.
- Đường dẫn dữ liệu và kết quả nằm dưới `data/` và `outputs/` (đều bị git bỏ qua).
- Mỗi module có test tương ứng trong `tests/`.

## Trạng thái hiện tại

- **Đã xong:** Giai đoạn 1 – khởi tạo dự án (repo, cấu trúc, môi trường, `check_env.py`).
- **Tiếp theo:** Giai đoạn 2 – khảo sát và chọn dataset (xem [ROADMAP.md](ROADMAP.md)).

## Quyết định đã chốt cho các giai đoạn tới

- Dataset: dùng bộ tốt nhất tải được tự do; bộ cần ký thỏa thuận thì ghi trong `docs/DATASETS.md`
  kèm hướng dẫn xin. Đổi dataset chỉ cần sửa YAML.
- Encoder đóng băng: trích xuất và cache đặc trưng ảnh (fp16) ra `data/`, huấn luyện đọc từ cache;
  vẫn giữ tùy chọn chạy encoder trực tiếp để fine-tune về sau.
- Đánh giá: BLEU và CIDEr là chỉ số chính. METEOR chỉ để tham khảo vì không có tài nguyên từ đồng
  nghĩa cho tiếng Việt (chỉ khớp chính xác theo từ).
