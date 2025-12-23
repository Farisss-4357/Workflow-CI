# Mulai dari image dasar Python yang ringan
FROM python:3.12-slim

# Tetapkan direktori kerja di dalam container
WORKDIR /app

# Salin file environment Conda yang sudah ada
COPY requirements.txt /app/

# Semua dependensi yang berhasil diinstal di CI sebelumnya akan terinstal di sini
RUN pip install --no-cache-dir -r requirements.txt

COPY mlruns /app/mlruns

# Buka port 5000 agar aplikasi model bisa diakses
EXPOSE 5000

# Atau Anda bisa menggunakan Run ID terakhir yang sukses, jika itu yang Anda inginkan
CMD ["mlflow", "models", "serve", "-m", "runs:/22c229c604184683a51318072cc0ed80/model", "--port", "5000", "--host", "0.0.0.0"]