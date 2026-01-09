FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir -p data/orig data/processed
COPY . .

RUN python setup.py build_ext --inplace

CMD ["python", "test_module.py"]