"""Server entry point to run FastAPI Face Comparison Service."""

import argparse
import uvicorn
from app.config import settings


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the face comparison service"
            if settings.HF_SPACE
            else "Run BPJS Ketenagakerjaan Face Comparison API Service"
        )
    )
    parser.add_argument("--host", type=str, default=settings.HOST, help="Host interface to bind")
    parser.add_argument("--port", type=int, default=settings.PORT, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code change")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument(
        "--ssl-certfile",
        type=str,
        default=settings.SSL_CERTFILE or None,
        help="TLS certificate. Required for the camera page from another machine.",
    )
    parser.add_argument("--ssl-keyfile", type=str, default=settings.SSL_KEYFILE or None, help="TLS private key")
    args = parser.parse_args()

    scheme = "https" if args.ssl_certfile else "http"
    print("=" * 60)
    print(f"  Starting {settings.SERVICE_NAME} v{settings.SERVICE_VERSION}")
    display_host = "localhost" if args.host in {"0.0.0.0", "::"} else args.host
    print(f"  Host: {scheme}://{args.host}:{args.port}")
    print(f"  UI:   {scheme}://{display_host}:{args.port}/")
    print(f"  Camera demo: {scheme}://{display_host}:{args.port}/presensi")
    print(f"  Docs: {scheme}://{display_host}:{args.port}/docs")
    print(f"  Logs: {settings.LOG_DIR}/{settings.LOG_FILE_PATTERN}")
    print("=" * 60)

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        ssl_certfile=args.ssl_certfile,
        ssl_keyfile=args.ssl_keyfile,
        log_level="info",
    )


if __name__ == "__main__":
    main()
