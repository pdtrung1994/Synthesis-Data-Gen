# Chạy Source Code trên HPC (ELTE Cluster)

Tài liệu này hướng dẫn các bước để đưa mã nguồn lên và thực thi trên HPC Cluster.
Mã nguồn đã được loại bỏ GPU và tối ưu hóa tối đa (chống oversubscription) cho việc chạy đa phân luồng trên CPU của SLURM.

## 1. Môi trường Container (Apptainer/Singularity)
Thay vì cài đặt thư viện lên OS hay sử dụng virtualenv/conda, chúng ta sử dụng Apptainer (Singularity) container theo chuẩn của hệ thống HPC. 

**Bước 1: Khởi tạo file định nghĩa môi trường (`myenvironment.recipe`)**
File `myenvironment.recipe` đã có sẵn trong source code với các thư viện cần thiết.

**Bước 2: Build Apptainer Image**
**Lưu ý quan trọng**: Do các cluster HPC thường dùng ổ đĩa mạng (NFS) với tính năng bảo mật chặn quyền ghi của root (`root_squash`), lệnh `sudo` không thể lưu file trực tiếp vào thư mục làm việc của bạn. Bạn cần build file vào thư mục tạm `/tmp` của máy chủ, sau đó copy về:

```bash
# 1. Build image và lưu tạm vào /tmp
sudo apptainer build /tmp/myenvironment.simg myenvironment.recipe

# 2. Copy file ảnh từ /tmp về thư mục hiện tại của bạn
cp /tmp/myenvironment.simg ./myenvironment.simg
```
*(Nếu cluster không có quyền sudo, bạn có thể thử `apptainer build --fakeroot myenvironment.simg myenvironment.recipe`, hoặc build sẵn file `.simg` trên máy tính cá nhân rồi copy lên HPC).*

Khi cần cài thêm thư viện, bạn hãy cập nhật file `myenvironment.recipe` (ở phần `%post`) và tiến hành build lại image.

## 2. Nộp công việc (Submit Job) cho SLURM (Job Array)
Hệ thống hiện được chia thành 2 phiên bản độc lập tùy theo mục đích chạy của bạn: **Short Tune** và **Full Tune**. Cả hai đều hỗ trợ **SLURM Job Array** giúp chia nhỏ job để chạy song song siêu nhanh.

**Cách chạy:** Bạn KHÔNG DÙNG lệnh `sbatch` trực tiếp, thay vào đó hãy chạy script như một file bash thông thường:

### Lựa chọn 1: Chạy Short Tune (Chạy bản nháp - Cực nhanh)
Mỗi Array Job chiếm trọn 1 Node (56 CPUs) và chạy song song tất cả các nếp gấp (folds). Thời gian chạy thường dưới 2-3 tiếng.
```bash
bash submit_hpc_short_tune.sh
```

### Lựa chọn 2: Chạy Full Tune (Chạy bản xịn - Dành cho Bài Báo Khoa Học)
Chia nhỏ thành ~135 Array Jobs siêu gọn, mỗi job chỉ xin 8 CPUs. Tuning siêu chuẩn theo phương pháp Validation trên dữ liệu thực để không bị Overfitting. 
```bash
bash submit_hpc_full_tune.sh
```

**Kịch bản tự động:** Khi bạn chạy bất kỳ lệnh nào ở trên, script sẽ:
1. Tự động kiểm tra và tạo thư mục chạy tiếp theo (ví dụ: `run_01` hoặc `full_tune`).
2. Đồng bộ tự động 2 thư mục: `logs/` (để chứa log in ra màn hình) và `Results/` (chứa file CSV kết quả).
3. Đẩy Array Job lên Slurm nền.

## 3. Quản lý và theo dõi quá trình chạy
Sau khi nộp, hệ thống sẽ trả về một `job_id` (ví dụ: `12345`). Do dùng Array, các job con sẽ có dạng `<job_id>_<task_id>` (ví dụ: `12345_0`, `12345_1`, v.v...).

- **Xem danh sách job đang chờ / chạy:**
  ```bash
  squeue -u <username>
  ```
- **Hủy toàn bộ mảng job hoặc một job cụ thể:**
  ```bash
  scancel 12345      # Hủy tất cả job thuộc mảng này
  scancel 12345_2    # Hủy riêng job có index 2
  ```
- **Xem log đầu ra:** 
  Toàn bộ log được ghi gọn gàng vào thư mục `logs/run_XX/slurm_<job_id>_<task_id>.out`.
  Bạn có thể theo dõi trực tiếp bằng lệnh (ví dụ):
  ```bash
  tail -f logs/run_01/slurm_12345_0.out
  ```

---

**Lưu ý khi chạy trên máy cá nhân:**
Các file `Runners/runner_hpc_short_tune.py` và `Runners/runner_full_tune.py` đã được lập trình thông minh. Nếu bạn chạy trên máy bàn (không thông qua `sbatch`), code sẽ tự động nhận diện không có hệ thống HPC và tự động chuyển về cơ chế chạy đa luồng (`N_JOBS=-1`), tối ưu hóa toàn bộ số cores có sẵn trên máy bàn của bạn thay vì giới hạn tài nguyên như trên Slurm.

---

## 4. [Bonus] Hướng dẫn quản lý Git trên HPC
Khi chạy trên HPC, dữ liệu sinh ra thường rất lớn (file csv nặng, log dài, file ảnh container `.simg` khổng lồ). Điều này rất dễ gây kẹt khi đẩy lên GitHub.

**1. Hủy bỏ lệnh `git add .` khi bị đơ**
Nếu bạn lỡ gõ `git add .` và nó chạy mãi không xong, hãy bấm `Ctrl + C` để hủy. Nguyên nhân là thư mục `Dataset/` hoặc `anaconda_projects/` quá nặng. File `.gitignore` đã được cấu hình để chặn chúng, nhưng để tháo những file lỡ bị dính vào hàng đợi, hãy chạy:
```bash
git reset
```

**2. Lỗi "Push thất bại do file quá giới hạn 100MB"**
Nếu bạn lỡ commit một file khổng lồ (như `myenvironment.simg`) vào một thời điểm nào đó, GitHub sẽ liên tục từ chối lệnh `git push`. Cách dọn dẹp an toàn nhất (gom commit chưa push) mà **không làm mất file thật**:
```bash
# Tua ngược lịch sử Git cục bộ về bằng với trên Github (giữ nguyên file trên máy)
git reset --soft origin/main

# Gỡ bỏ file nặng khỏi lịch sử chờ commit
git rm --cached myenvironment.simg

# Gom mọi thứ còn lại thành 1 commit sạch sẽ
git commit -m "Update code and results"
git push origin main
```

**3. Xử lý kẹt (Conflict) file .gitignore trên HPC**
Nếu HPC báo lỗi Conflict trên file `.gitignore` khi bạn gõ `git pull`, hãy ép hệ thống chọn đúng phiên bản mới nhất trên nhánh remote (GitHub):
```bash
git checkout --theirs .gitignore
git add .gitignore
git commit -m "Resolve .gitignore conflict"
```
