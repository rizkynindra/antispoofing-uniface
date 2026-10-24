"""Gunicorn configuration optimized for 2x NVIDIA L4 (24GB each)."""

import os
import multiprocessing

# Server Socket
bind = os.getenv("BIND", "0.0.0.0:2112")
backlog = 2048

# Worker Processes
# For 2x 24GB L4 GPUs: 4-6 workers provide high throughput without VRAM or CPU contention
workers = int(os.getenv("WORKERS", "4"))
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
timeout = 120
keepalive = 5

# Memory & Process Management
max_requests = 5000         # Restart workers periodically to prevent memory fragmentation
max_requests_jitter = 500
graceful_timeout = 30

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Process Naming
proc_name = "svc-face-compare"
