FROM python:3.11-slim

WORKDIR /app

# System deps some ML libs need for building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Mount your project folder here to ingest it inside the container, e.g.:
#   docker run -v /path/to/your/repo:/project codecompass ...
VOLUME ["/project"]

EXPOSE 8000

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
