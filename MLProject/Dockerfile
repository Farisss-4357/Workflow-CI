# Base image python
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy semua file MLProject ke container
COPY . /app

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies (sesuai conda.yaml tapi pakai pip)
RUN pip install \
    mlflow==2.19.0 \
    pandas \
    scikit-learn \
    numpy \
    scipy

# Default command
CMD ["python", "modelling.py"]