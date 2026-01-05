docker build -t langgraph-google-genai -f Dockerfile.dev .
docker run --rm \
    --env-file .env \
    -p 8000:8000 \
    -v "$(pwd):/app" \
    langgraph-google-genai