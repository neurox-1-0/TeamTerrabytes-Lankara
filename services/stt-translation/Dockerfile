FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
ENV PIP_DEFAULT_TIMEOUT=1000
# CPU torch first — avoids multi-GB CUDA wheels that often time out in Docker builds
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt
COPY app ./app
EXPOSE 8005
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8005"]
