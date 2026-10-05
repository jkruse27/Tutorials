docker build --rm --force-rm -t hrv-tester .
docker run --rm --gpus all -t -v "$(pwd)/data:/app/data" hrv-tester
