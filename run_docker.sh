docker build -t hrv-tester .
docker run --rm -v "$(pwd)/data:/app/data" hrv-tester