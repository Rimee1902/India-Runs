# Self-contained sandbox recipe (fallback per submission_spec.md 10.5, if
# a hosted sandbox link isn't used). Builds and runs unmodified.
#
# Build:  docker build -t redrob-ranker .
# Run:    docker run -p 8501:8501 redrob-ranker
# Then open http://localhost:8501 and use the bundled sample candidates.

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY app.py .
COPY data/sample_candidates.json ./data/sample_candidates.json

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
