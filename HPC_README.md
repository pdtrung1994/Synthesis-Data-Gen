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
Hệ thống nay đã được cấu hình dùng **SLURM Job Array** (`--array=0-4`). Điều này cho phép bạn **chỉ cần chạy một lệnh submit duy nhất, nhưng HPC sẽ chia thành 5 job riêng biệt**, mỗi job chạy song song cho một dataset khác nhau (Plant_oil, Brewed_vinegar, Wine_spoilage, Chinese_wine, Coffee) giúp hoàn thành nhanh hơn gấp 5 lần so với chạy nối tiếp.

Bạn có thể chỉnh sửa file `submit_cpu.sh`:
- `--array=0-4`: Đang thiết lập 5 job (từ index 0 đến 4).
- `--cpus-per-task=56`: Thay đổi con số `56` thành số lõi CPU bạn muốn dùng cho *mỗi job*. 
- `--time=12:00:00`: Giới hạn thời gian chạy tối đa.

Gửi các job vào hàng đợi bằng lệnh:
```bash
sbatch submit_cpu.sh
```

## 3. Quản lý và theo dõi quá trình chạy
Sau khi nộp, hệ thống sẽ trả về một `job_id` (ví dụ: `12345`). Do dùng Array, các job con sẽ có dạng `<job_id>_<task_id>` (ví dụ: `12345_0`, `12345_1`, v.v...).

- **Xem danh sách job đang chờ / chạy:**
  ```bash
  squeue -u <username>
  ```
- **Hủy toàn bộ mảng job hoặc một job cụ thể:**
  ```bash
  scancel 12345      # Hủy tất cả 5 job thuộc mảng này
  scancel 12345_2    # Hủy riêng job có index 2
  ```
- **Xem log đầu ra của từng dataset:** 
  Output của màn hình console sẽ được ghi thành 5 file riêng biệt `slurm_<job_id>_<task_id>.out`.
  Bạn có thể theo dõi trực tiếp bằng lệnh (ví dụ với job 0):
  ```bash
  tail -f slurm_12345_0.out
  ```
- Các log chi tiết của từng dataset/seed được `runner.py` tự động ghi vào thư mục `Runners/logs/`.

---

**Lưu ý khi chạy trên máy cá nhân:**
File `runner.py` đã được lập trình thông minh. Nếu bạn chạy lệnh `python Runners/runner.py` trên máy bàn (không thông qua `sbatch`), code sẽ tự động nhận diện không có HPC, chuyển về cơ chế chạy đa luồng cũ (N_JOBS=-1) và không bị giới hạn 1 core mỗi tiến trình như HPC. Bạn cũng có thể test việc chạy riêng 1 database (ví dụ: `python Runners/runner.py --datasets Plant_oil`). File `runner.ipynb` cũng không bị ảnh hưởng.
