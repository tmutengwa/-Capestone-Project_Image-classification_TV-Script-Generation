# Dockerfile

FROM agrigorev/zoomcamp-model:2025
WORKDIR /app

# --- MODIFIED: Use requirements.txt instead ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# ----------------------------------------------

# Copy the rest of the application files (homework5.py, model file, etc.)
COPY homework5.py .
COPY pipeline_v1.bin .

# Run uvicorn
CMD ["uvicorn", "homework5:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]