# Chạy Source Code trên HPC (ELTE Cluster)

Tài liệu này hướng dẫn các bước để đưa mã nguồn lên và thực thi trên HPC Cluster.
Mã nguồn đã được loại bỏ GPU và tối ưu hóa tối đa (chống oversubscription) cho việc chạy đa phân luồng trên CPU của SLURM.

## 1. Môi trường Container (Apptainer/Singularity)
Thay vì cài đặt thư viện lên OS hay sử dụng virtualenv/conda, chúng ta sử dụng Apptainer (Singularity) container theo chuẩn của hệ thống HPC. 

**Bước 1: Khởi tạo file định nghĩa môi trường (`myenvironment.recipe`)**
File `myenvironment.recipe` đã có sẵn trong source code với các thư viện cần thiết.

**Bước 2: Build Apptainer Image**
Chạy lệnh sau trên HPC để build image:
```bash
sudo apptainer build myenvironment.simg myenvironment.recipe
```
*(File ảnh `myenvironment.simg` sẽ được tạo ra. Nếu cluster không có quyền sudo cho lệnh build, bạn có thể build trên máy tính cá nhân đã cài apptainer và copy file .simg lên HPC).*

Khi cần cài thêm thư viện, bạn hãy cập nhật file `myenvironment.recipe` (ở phần `%post`) và tiến hành build lại image.

## 2. Nộp công việc (Submit Job) cho SLURM
Chúng ta sử dụng file `submit_cpu.sh` để cấu hình số lượng core và yêu cầu SLURM cấp phát tài nguyên. Kịch bản chạy đã được cập nhật để sử dụng Singularity/Apptainer.

Bạn có thể chỉnh sửa file `submit_cpu.sh`:
- `--cpus-per-task=56`: Thay đổi con số `56` thành số lõi CPU bạn muốn dùng. Code Python (`runner.py`) sẽ **tự động phát hiện con số này** để tạo đúng bấy nhiêu luồng xử lý song song.
- `--time=12:00:00`: Giới hạn thời gian chạy tối đa.

Gửi job vào hàng đợi bằng lệnh:
```bash
sbatch submit_cpu.sh
```

## 3. Quản lý và theo dõi quá trình chạy
Sau khi nộp, hệ thống sẽ trả về một `job_id` (ví dụ: `12345`).

- **Xem danh sách job đang chờ / chạy:**
  ```bash
  squeue -u <username>
  ```
- **Hủy job nếu chạy lỗi hoặc muốn dừng:**
  ```bash
  scancel 12345
  ```
- **Xem log đầu ra:** 
  Output của màn hình console sẽ được ghi trực tiếp vào file `logs/slurm_<job_id>.out`. 
  Bạn có thể theo dõi trực tiếp bằng lệnh:
  ```bash
  tail -f logs/slurm_12345.out
  ```
- Các log chi tiết của từng dataset/seed được `runner.py` tự động ghi vào thư mục `Runners/logs/`.

---

**Lưu ý khi chạy trên máy cá nhân:**
File `runner.py` đã được lập trình thông minh. Nếu bạn chạy lệnh `python Runners/runner.py` trên máy bàn (không thông qua `sbatch`), code sẽ tự động nhận diện không có HPC, chuyển về cơ chế chạy đa luồng cũ (N_JOBS=-1) và không bị giới hạn 1 core mỗi tiến trình như HPC. File `runner.ipynb` cũng không bị ảnh hưởng.
