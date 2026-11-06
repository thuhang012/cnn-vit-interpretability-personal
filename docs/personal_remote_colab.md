# Sửa trên máy, chạy trên Colab

Repo cá nhân public: https://github.com/thuhang012/cnn-vit-interpretability-personal

- Windows: `origin` là repo nhóm; `personal` là repo cá nhân.
- `git push` mặc định sang `personal`; upstream/pull của nhánh vẫn là repo nhóm.
- Colab: clone repo cá nhân, nhánh `feat/gradcam`; không cần PAT.
- Code đi qua Git; ImageNet, weights, checkpoint và kết quả để ngoài Git.

## Lần đầu trên Colab

1. Dùng input đã có tại `MyDrive/cv_2026/data_10k.zip`.
   ZIP chứa thư mục `data_10k/` với 14 file `validation-*.parquet`,
   `analysis_10000_seed42.csv`, `subset_info.json` và `README.md`.
   Không cần upload lại dataset mỗi lần sửa code.
2. Mở một trong hai notebook:
   - [Attention](https://colab.research.google.com/github/thuhang012/cnn-vit-interpretability-personal/blob/feat/gradcam/notebooks/03_attention.ipynb)
   - [Grad-CAM](https://colab.research.google.com/github/thuhang012/cnn-vit-interpretability-personal/blob/feat/gradcam/notebooks/02_gradcam.ipynb)
3. Runtime → Change runtime type → GPU (ví dụ T4 nếu có).
4. Chạy cell setup đầu tiên, cấp quyền mount Drive khi Colab hỏi.
   Cell tự clone code, giải nén 14 shard và metadata ra ổ Colab để đọc nhanh,
   đối chiếu manifest cùng ID/nhãn trong Parquet với 10.000 ảnh của repo,
   cài requirements một lần mỗi runtime/nội dung requirements, và đặt cache weights
   trên Drive. Lần đầu cần thời gian giải nén, cài thư viện và tải weights.
5. Chạy các cell cấu hình, chọn ảnh, experiment và analysis theo thứ tự.
   Giữ mặc định smoke test nhỏ trước khi chạy cohort lớn. Trên Colab,
   `ALLOW_DOWNLOAD` mặc định bật để tải pretrained weights khi cache chưa có.

Nếu không dùng ZIP, đặt rõ `DATA_DIR` trong cell setup, trỏ tới thư mục chứa
đủ 14 shard Parquet. Đường dẫn chỉ định rõ được ưu tiên hơn ZIP.
Đọc trực tiếp từ Drive có thể chậm hơn giải nén ZIP ra ổ Colab.

Mọi output/checkpoint nằm dưới `MyDrive/cv_2026/`:

```text
cv_2026/
  data_10k.zip                              # Input giữ nguyên
  metrics/experiments/<experiment>/<run>/   # Metrics và checkpoint
  figures/experiments/<experiment>/<run>/   # Hình
  torch-cache/                             # Cache pretrained weights
```

Không cần tải cache weights hoặc input về GitHub. Dataset giải nén nằm ở ổ tạm
`/content/imagenet-cache/`; setup tự chuẩn bị lại khi Colab cấp runtime mới.

## Mỗi vòng thử nghiệm

**Trên máy:** sửa code trong VS Code rồi chạy tại thư mục repo:

```powershell
.\sync.ps1
```

Hoặc viết thông điệp commit rõ hơn:

```powershell
.\sync.ps1 "Adjust attention visualization"
```

Script stage toàn bộ thay đổi không bị `.gitignore` loại, gồm file mới và các
lần xóa, commit nếu có thay đổi, rồi push **chỉ sang `personal/feat/gradcam`**.
Không có thay đổi thì chỉ thử push các commit chưa gửi. Script dừng nếu remote
hoặc nhánh không đúng; không reset/force-push. Xem `git diff` trước nếu cần chọn
thay đổi; dùng Git thủ công khi không muốn đưa mọi thay đổi vào cùng commit.
Các commit snapshot vẫn tồn tại trong lịch sử.

Nếu PowerShell chặn script do execution policy, dùng lệnh sau cho riêng lần chạy:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\sync.ps1
```

**Trên Colab:** dừng experiment đang chạy → chạy lại cell setup đầu tiên để pull
→ chạy lại cell cấu hình và các cell experiment/analysis cần thiết. Module Python
được autoreload trước các cell tiếp theo. Không cần clone lại, upload code hoặc
nhập token. Đừng chạy lại notebook cũ trong lúc experiment cũ vẫn đang thực thi.

Nếu sửa cấu trúc model/class, thay dependencies hoặc thấy object cũ còn được dùng,
chọn Runtime → Restart session rồi chạy lại các cell. Pull không tự tạo lại object
hay tensor đã nằm trong bộ nhớ. Dataset ở ổ Colab còn dùng được khi chỉ restart
Python; nếu runtime bị xóa hoàn toàn, setup tự clone và giải nén lại từ Drive.

**Nếu sửa cell notebook trên máy:** mở lại link notebook GitHub bên trên để lấy
cell mới. Pull trong runtime chỉ đổi file trên đĩa, không cập nhật cell đang mở
trong trình soạn thảo Colab. Nên tập trung chỉnh thuật toán trong `methods.py`,
`experiment.py`, `analysis.py` để ít phải mở lại notebook.

## Chạy tiếp và đọc kết quả

Giữ cùng đường dẫn kết quả và cấu hình để dùng cơ chế resume của experiment.
Sau khi đổi phương pháp hoặc tham số ảnh hưởng kết quả, kiểm tra run ID/cấu hình
được notebook tạo; không coi checkpoint từ phương pháp cũ là kết quả mới.
Xem hình/bảng trong Colab; file xuất và checkpoint được giữ trên Drive.
Để phân tích trên laptop sau mỗi lượt, tải hai thư mục `final/` của run cần xem:

| Từ Drive | Lưu trên laptop, tính từ thư mục repo |
| --- | --- |
| `cv_2026/metrics/experiments/<experiment>/<run>/final/` | `results/metrics/experiments/<experiment>/<run>/final/` |
| `cv_2026/figures/experiments/<experiment>/<run>/final/` | `results/figures/experiments/<experiment>/<run>/final/` |

`<experiment>` là `attention` hoặc `gradcam`; `<run>` xem biến `RUN_TAG` và
đường dẫn được notebook in ra. Nếu cần phân tích lại từ heatmap/checkpoint,
tải toàn bộ thư mục run ở nhánh `metrics/`, giữ nguyên cấu trúc. Không cần
tải `data_10k.zip` hoặc `torch-cache/`. Khi lượt chạy lỗi chưa export final,
gửi traceback và file `config.json` của checkpoint trước.

Git pull không tải output từ Drive. Kết nối Google Drive cho trợ lý, hoặc tải
thủ công các thư mục trên. Sau khi file có trên laptop, trợ lý đọc và phân tích
trực tiếp. Các file trong `final/` có thể được Git track theo `.gitignore` hiện
tại, nên kiểm tra trước lần sync kế tiếp nếu chưa muốn công khai kết quả.

## Chia sẻ với nhóm khi hài lòng

Script sync không push repo nhóm. Kiểm tra commit trước khi tự chia sẻ:

```powershell
git log --oneline origin/feat/gradcam..HEAD
git push --dry-run origin feat/gradcam
git push origin feat/gradcam
```

Push như vậy gửi cả lịch sử snapshot chưa có ở nhóm. Nếu muốn nhóm nhận một commit
gọn, chuẩn bị nhánh chia sẻ và squash/cherry-pick riêng trước, không force-push
nhánh nhóm. Repo cá nhân public: mọi người xem được code và lịch sử được push;
không đưa token, dữ liệu ImageNet hoặc file riêng tư vào commit.

Nguồn tham khảo: [GitHub CLI](https://cli.github.com/manual/gh_repo_create),
[Git push configuration](https://git-scm.com/docs/git-config).
