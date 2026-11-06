# Sửa trên máy, chạy trên Colab

Repo cá nhân public: https://github.com/thuhang012/cnn-vit-interpretability-personal

- Windows: `origin` là repo nhóm; `personal` là repo cá nhân.
- `git push` mặc định sang `personal`; upstream/pull của nhánh vẫn là repo nhóm.
- Colab: clone repo cá nhân, nhánh `feat/gradcam`; không cần PAT.
- Code đi qua Git; ImageNet, weights, checkpoint và kết quả để ngoài Git.

## Lần đầu trên Colab

1. Đặt ZIP dataset sẵn có `cnn-vit-interpretability.zip` ở ngay MyDrive.
   ZIP cần chứa đủ 14 file dưới `cnn-vit-interpretability/data/imagenet/data/`
   (hoặc `data/imagenet/data/`). Chỉ cần upload một lần, không cần đóng ZIP lại
   mỗi khi sửa code. Code trong ZIP không được dùng nữa.
2. Mở một trong hai notebook:
   - [Attention](https://colab.research.google.com/github/thuhang012/cnn-vit-interpretability-personal/blob/feat/gradcam/notebooks/03_attention.ipynb)
   - [Grad-CAM](https://colab.research.google.com/github/thuhang012/cnn-vit-interpretability-personal/blob/feat/gradcam/notebooks/02_gradcam.ipynb)
3. Runtime → Change runtime type → GPU (ví dụ T4 nếu có).
4. Chạy cell setup đầu tiên, cấp quyền mount Drive khi Colab hỏi.
   Cell tự clone code, giải nén riêng 14 shard ra ổ Colab để đọc nhanh,
   cài requirements một lần mỗi runtime/nội dung requirements, và đặt cache weights
   trên Drive. Lần đầu cần thời gian giải nén, cài thư viện và tải weights.
5. Chạy các cell cấu hình, chọn ảnh, experiment và analysis theo thứ tự.
   Giữ mặc định smoke test nhỏ trước khi chạy cohort lớn. Trên Colab,
   `ALLOW_DOWNLOAD` mặc định bật để tải pretrained weights khi cache chưa có.

Nếu không dùng ZIP, đặt đủ 14 shard vào
`MyDrive/cnn-vit-data/imagenet/data/` và bỏ/đổi tên ZIP mặc định, hoặc đặt rõ
`DATA_DIR` trong cell setup. Đường dẫn chỉ định rõ được ưu tiên hơn ZIP.
Đọc trực tiếp từ Drive có thể chậm hơn giải nén ZIP ra ổ Colab.

Kết quả/checkpoint mặc định: `MyDrive/cnn-vit-interpretability-results/`.
Weights: `MyDrive/torch-cache/`. Không cần tải các thư mục này về GitHub.

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
Nếu muốn xem kết quả trên máy, tải thư mục run cần thiết từ Drive về, hoặc dùng
Google Drive for desktop nếu đã có. Git pull không tải kết quả bị ignore.

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
