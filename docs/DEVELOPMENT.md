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

## Cài đặt trên máy mới

Chạy từ thư mục gốc của repo, đúng thứ tự:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
pip install -e . --no-deps
python scripts/check_env.py
```

- Bước nâng cấp `pip` là bắt buộc: `pip` 23 đi kèm Python 3.10 báo lỗi khi cài từ kho PyTorch.
- PyTorch cài riêng bằng `--index-url` để lấy bản CUDA 13.0. Trong `requirements.txt`, `torch`
  và `torchvision` được ghim không kèm hậu tố `+cu130`, nên bước sau coi bản CUDA đã cài là
  thỏa mãn: không lỗi, không ghi đè. Nếu bỏ qua bước cài PyTorch riêng, `pip` sẽ lấy bản CPU từ
  PyPI và `check_env.py` sẽ báo `FAILED`.
- Khi thêm thư viện: cài vào `.venv`, rồi ghi tên kèm phiên bản chính xác vào `requirements.txt`.

## Lệnh thường dùng

Mọi lệnh chạy từ thư mục gốc của repo, sau khi kích hoạt `.venv`.

```powershell
python scripts/check_env.py           # kiểm tra PyTorch / CUDA / GPU
pytest                                # chạy test
ruff check . ; ruff format .          # lint và định dạng code

# Dữ liệu (xem docs/DATASETS.md)
python scripts/download_data.py --config configs/datasets/uit_openviic.yaml
python scripts/verify_data.py --config configs/datasets/uit_openviic.yaml

# Tiền xử lý: chuẩn hóa, tách từ, từ điển (thêm key=value ở cuối để ghi đè cấu hình)
python scripts/prepare_data.py --config configs/baseline_lstm.yaml

# Cache đặc trưng ảnh (một lần cho mỗi encoder / kích thước ảnh)
python scripts/extract_features.py --config configs/baseline_lstm.yaml

# Huấn luyện, tiếp tục khi bị ngắt, chạy thử nhanh
python scripts/train.py --config configs/baseline_lstm.yaml
python scripts/train.py --config configs/baseline_lstm.yaml --resume auto
python scripts/train.py --config configs/baseline_lstm.yaml --max-steps 300 run_name=smoke

# Đánh giá và suy luận
python scripts/evaluate.py --checkpoint outputs/baseline_lstm/checkpoints/best.pt --split test
python scripts/predict.py --checkpoint outputs/baseline_lstm/checkpoints/best.pt --image anh.jpg
tensorboard --logdir outputs
```

Trên Windows, khi in tiếng Việt ra console bị lỗi mã hóa thì thêm `-X utf8` sau `python`.

## Quy ước code

- Mã nguồn chính nằm trong gói `vic` (`src/vic/`); `scripts/` chỉ chứa điểm vào dòng lệnh mỏng.
- Định dạng và lint bằng `ruff` (dòng tối đa 100 ký tự), có type hint cho hàm công khai.
- Mọi tham số thí nghiệm nằm trong `configs/*.yaml`, không viết cứng trong code.
- Đường dẫn dữ liệu và kết quả nằm dưới `data/` và `outputs/` (đều bị git bỏ qua).
- Mỗi module có test tương ứng trong `tests/`.
- Decoder mới hiện thực giao diện chung (`forward`, `init_state`, `step`, `reorder_state`) và
  đăng ký vào `DECODERS`; dataset mới thêm YAML trong `configs/datasets/` và, nếu cần, một
  adapter trong `src/vic/data/sources.py`.
- Mỗi thí nghiệm có một file YAML riêng trong `configs/` và một `run_name` riêng; không sửa
  file cấu hình của thí nghiệm đã chạy.
- Các Pull Request của nhiều giai đoạn xếp chồng lên nhau: merge theo đúng thứ tự, dùng
  "Create a merge commit" (không squash).

## Trạng thái hiện tại

- **Đã xong:** Giai đoạn 1 – khởi tạo dự án (repo, cấu trúc, môi trường, `check_env.py`).
- **Đã xong:** Giai đoạn 2 – chọn UIT-OpenViIC, script tải và kiểm tra dữ liệu, định dạng
  annotation thống nhất (xem [DATASETS.md](DATASETS.md)).
- **Đã xong:** Giai đoạn 3 – chuẩn hóa và tách từ tiếng Việt, từ điển, `Dataset` / collate /
  sampler tái lập được (`src/vic/text/`, `src/vic/data/`).
- **Đã xong:** Giai đoạn 4 – baseline ResNet-50 + LSTM, cache đặc trưng, vòng huấn luyện có
  resume, đánh giá, suy luận. Đã chạy thử 400 bước; **chưa huấn luyện đầy đủ**.
- **Tiếp theo:** huấn luyện đầy đủ baseline, rồi Giai đoạn 5 – mô hình attention và Transformer.
  Các bước cụ thể nằm trong [ROADMAP.md](ROADMAP.md); kiến trúc và lý do trong
  [DESIGN.md](DESIGN.md).

### Dữ liệu đã có sẵn trên máy phát triển

Các thư mục sau đã được tạo (không nằm trong git), không cần chạy lại trừ khi đổi cấu hình:

- `data/raw/uit_openviic/`: ảnh và annotation gốc;
- `data/processed/uit_openviic/word_minfreq3/`: caption đã tách từ và từ điển;
- `data/features/uit_openviic/resnet50_224/`: cache đặc trưng (2,45 GB);
- `outputs/smoke_baseline/`: kết quả chạy thử, xóa được.

### Việc còn dở

- Chưa có kết quả huấn luyện đầy đủ nên chưa có bảng điểm chính thức.
- Chưa có tùy chọn nạp trọng số để fine-tune encoder (xem Giai đoạn 6 trong lộ trình).
- Chưa có CI; chạy `pytest` và `ruff check .` trước mỗi lần mở Pull Request.

## Quyết định đã chốt cho các giai đoạn tới

- Dataset: UIT-OpenViIC (tải tự do, tác giả cho phép dùng cho nghiên cứu). Đổi dataset chỉ cần
  thêm một file YAML trong `configs/datasets/` và sửa dòng `dataset_config`.
- Encoder đóng băng: trích xuất và cache đặc trưng ảnh (fp16) ra `data/`, huấn luyện đọc từ cache;
  vẫn giữ tùy chọn chạy encoder trực tiếp để fine-tune về sau.
- Đánh giá: BLEU và CIDEr là chỉ số chính. METEOR chỉ để tham khảo vì không có tài nguyên từ đồng
  nghĩa cho tiếng Việt (chỉ khớp chính xác theo từ).
