"""Sends one image to a running server and prints the results.

Usage:
    python scripts/smoke_test.py path/to/photo.jpg --url http://127.0.0.1:8000 --k 5
"""
import argparse
import json
import mimetypes
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    boundary = uuid.uuid4().hex
    content_type = mimetypes.guess_type(args.image.name)[0] or "application/octet-stream"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{args.image.name}\"\r\n"
            f"Content-Type: {content_type}\r\n\r\n").encode() + args.image.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(f"{args.url}/search?k={args.k}", data=body,
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read())
    except urllib.error.HTTPError as error:
        print(f"HTTP {error.code}: {error.read().decode()}")
        return 1
    except urllib.error.URLError as error:
        print(f"Cannot reach {args.url}: {error.reason}")
        return 1

    print(f"Query: {data['query']['filename']} ({data['query']['width']}x{data['query']['height']})")
    print("Timings (ms):", data["timings"])
    for hit in data["results"]:
        print(f"  #{hit['rank']}  {hit['score']:.3f}  id {hit['id']:>6}  {hit['article_type']:<18} {hit['base_colour']!s:<10} {hit['name']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
