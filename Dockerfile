FROM ultralytics/ultralytics:latest-cpu

WORKDIR /app

# FastAPI stack + python-multipart for uploads. ffmpeg ya viene en la base.
# ffmpeg for the H.264 re-encode of the annotated mp4 (the ultralytics base
# image ships libavcodec via opencv but not the standalone ffmpeg binary).
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi>=0.110 \
    "uvicorn[standard]>=0.27" \
    python-multipart>=0.0.9 \
    "supervision>=0.21,<1.0"

COPY app /app/app

# YOLOE descarga pesos al primer uso; cache va a /app/data/models vía volumen
ENV YOLO_CONFIG_DIR=/app/data/models \
    PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
