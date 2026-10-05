FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir -p data/orig data/processed
COPY . .

RUN python3 setup.py build_ext --inplace

CMD ["python3", "test_module.py"]