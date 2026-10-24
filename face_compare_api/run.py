"""Server entry point to run FastAPI Face Comparison Service."""

import argparse
import uvicorn
from app.config import settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BPJS Ketenagakerjaan Face Comparison API Service")
    parser.add_argument("--host", type=str, default=settings.HOST, help="Host interface to bind")
    parser.add_argument("--port", type=int, default=settings.PORT, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code change")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  Starting {settings.SERVICE_NAME} v{settings.SERVICE_VERSION}")
    print(f"  Host: http://{args.host}:{args.port}")
    print(f"  Docs: http://{args.host}:{args.port}/docs")
    print(f"  Logs: {settings.LOG_DIR}/{settings.LOG_FILE_PATTERN}")
    print("=" * 60)

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
