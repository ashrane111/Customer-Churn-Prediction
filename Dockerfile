FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files
COPY models/xgboost_extreme.pkl models/
COPY models/optimal_threshold.txt models/

# Copy source code
COPY src/ src/

# Expose port
EXPOSE 8000

# Run API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]