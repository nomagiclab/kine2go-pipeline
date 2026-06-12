# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Serve the static project website locally."""

from __future__ import annotations

import argparse
import functools
import http.server
import socketserver
from pathlib import Path


class ReusableThreadingTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the kine2go project website locally.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind.")
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory to serve. Defaults to the website directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    directory = args.directory.resolve()

    if not directory.exists():
        raise SystemExit(f"Directory does not exist: {directory}")
    if not (directory / "index.html").exists():
        raise SystemExit(f"Directory does not contain index.html: {directory}")

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))

    with ReusableThreadingTCPServer((args.host, args.port), handler) as server:
        print(f"Serving {directory}")
        print(f"Open http://{args.host}:{args.port}/")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
