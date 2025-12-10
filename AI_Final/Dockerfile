FROM python:3.10-slim

WORKDIR /app

# ✅ Install required OS libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# ✅ Upgrade pip so it understands CPU torch wheels
RUN pip install --upgrade pip

# ✅ Copy only requirements first (for caching)
COPY requirements.txt /app/requirements.txt

# ✅ Install CPU-safe dependencies
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r /app/requirements.txt

# ✅ Copy the rest of the app
COPY . /app

EXPOSE 7860

CMD ["python", "app.py"]
