# ---------- Build stage ----------
FROM python:3.11-slim AS base

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# ---------- CLI entry-point ----------
# Default: run the evaluation pipeline
CMD ["python", "main.py"]

# ---------- Dashboard entry-point ----------
# To run the Streamlit dashboard instead:
#   docker run -p 8501:8501 netflix-rec streamlit run app.py --server.port=8501 --server.address=0.0.0.0
EXPOSE 8501
