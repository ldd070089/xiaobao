FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# 极重要的一步：安装编译工具，否则 mem0 会安装失败
RUN apt-get update && apt-get install -y --no-install-recommends gcc python3-dev && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get purge -y gcc python3-dev && apt-get autoremove -y

COPY main.py .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
