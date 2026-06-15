FROM python:3.12-slim

WORKDIR /app

# System deps for lightgbm
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# The build context (set in Render to repo root subdir `live/`) is copied
# under /app/live/ so the package imports `from live.X import ...` resolve.
COPY requirements.txt /app/live/requirements.txt
RUN pip install --no-cache-dir -r /app/live/requirements.txt

COPY . /app/live/

# Move the Procfile to /app so honcho finds it from WORKDIR.
RUN cp /app/live/Procfile /app/Procfile

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

CMD ["honcho", "start"]
