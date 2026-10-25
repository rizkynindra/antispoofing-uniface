FROM python:3.10-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml requirements.txt README.md ./
COPY uniface ./uniface
COPY face_compare_api ./face_compare_api

ENV HF_SPACE=1 \
    HOST=0.0.0.0 \
    PORT=7860 \
    WORKERS=1 \
    UNIFACE_CACHE_DIR=/app/.uniface/models \
    PYTHONPATH=/app/face_compare_api \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN pip install --no-cache-dir -e ".[cpu]" \
    && pip install --no-cache-dir -r requirements.txt \
    && mkdir -p /app/.uniface/models /app/face_compare_api/logs \
    && chmod -R a+rwX /app/.uniface /app/face_compare_api/logs \
    && python -c "from uniface.constants import AdaFaceWeights; from uniface.detection import RetinaFace; from uniface.recognition import AdaFace; from uniface.spoofing import MiniFASNet; RetinaFace(); AdaFace(model_name=AdaFaceWeights.IR_101); MiniFASNet()"

EXPOSE 7860

CMD ["python", "face_compare_api/run.py", "--host", "0.0.0.0", "--port", "7860"]
