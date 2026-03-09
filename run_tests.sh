docker build --rm --force-rm -t hrv-tester .
docker run --rm -t -v "$(pwd)/data:/app/data" hrv-tester
