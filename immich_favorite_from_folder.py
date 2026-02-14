#!/usr/bin/env python3
"""Set Immich assets as favorites based on files present in a local folder.

Computes SHA-1 checksums of local files, finds matching already-uploaded
assets in Immich, and marks them as favorites.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

BATCH_SIZE = 1000


def sha1_file(path: Path) -> str:
    """Return hex-encoded SHA-1 digest of a file."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def api_request(server: str, api_key: str, method: str, endpoint: str, body=None):
    """Make an Immich API request and return parsed JSON (or None for 204)."""
    url = f"{server.rstrip('/')}/api{endpoint}"
    data = json.dumps(body).encode() if body else None
    req = Request(url, data=data, method=method)
    req.add_header("x-api-key", api_key)
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    try:
        with urlopen(req) as resp:
            if resp.status == 204:
                return None
            return json.loads(resp.read())
    except HTTPError as e:
        body_text = e.read().decode(errors="replace")
        print(f"API error {e.code} on {method} {endpoint}: {body_text}", file=sys.stderr)
        sys.exit(1)
    except URLError as e:
        print(f"Connection error: {e.reason}", file=sys.stderr)
        sys.exit(1)


def scan_folder(folder: Path) -> list[Path]:
    """Recursively find all files in folder."""
    files = sorted(p for p in folder.rglob("*") if p.is_file())
    return files


def find_matching_assets(
    server: str, api_key: str, checksums: dict[str, Path], dry_run: bool
) -> list[tuple[str, Path]]:
    """Query Immich to find assets matching the given checksums.

    Returns list of (asset_id, local_path) tuples for matches.
    """
    items = list(checksums.items())
    matches = []

    for i in range(0, len(items), BATCH_SIZE):
        batch = items[i : i + BATCH_SIZE]
        assets = [{"id": str(path), "checksum": chk} for chk, path in batch]
        result = api_request(
            server, api_key, "POST", "/assets/bulk-upload-check", {"assets": assets}
        )
        for entry in result.get("results", []):
            asset_id = entry.get("assetId")
            if asset_id:
                local_path = Path(entry["id"])
                matches.append((asset_id, local_path))
        done = min(i + BATCH_SIZE, len(items))
        print(f"  Checked {done}/{len(items)} checksums...", file=sys.stderr)

    return matches


def set_favorites(
    server: str, api_key: str, asset_ids: list[str], dry_run: bool
) -> int:
    """Mark assets as favorites. Returns count of assets updated."""
    if dry_run:
        return len(asset_ids)

    for i in range(0, len(asset_ids), BATCH_SIZE):
        batch = asset_ids[i : i + BATCH_SIZE]
        api_request(server, api_key, "PUT", "/assets", {"ids": batch, "isFavorite": True})

    return len(asset_ids)


def main():
    parser = argparse.ArgumentParser(
        description="Set Immich favorites from a folder of files."
    )
    parser.add_argument(
        "--server",
        default=os.environ.get("IMMICH_SERVER_URL"),
        help="Immich server URL (or set IMMICH_SERVER_URL)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("IMMICH_API_KEY"),
        help="Immich API key (or set IMMICH_API_KEY)",
    )
    parser.add_argument(
        "--folder",
        required=True,
        type=Path,
        help="Folder containing favorite files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    args = parser.parse_args()

    if not args.server:
        parser.error("--server or IMMICH_SERVER_URL is required")
    if not args.api_key:
        parser.error("--api-key or IMMICH_API_KEY is required")
    if not args.folder.is_dir():
        parser.error(f"Not a directory: {args.folder}")

    if args.dry_run:
        print("[DRY RUN]", file=sys.stderr)

    # 1. Scan folder
    print(f"Scanning {args.folder}...", file=sys.stderr)
    files = scan_folder(args.folder)
    print(f"  Found {len(files)} files", file=sys.stderr)
    if not files:
        return

    # 2. Compute checksums
    print("Computing SHA-1 checksums...", file=sys.stderr)
    checksums: dict[str, Path] = {}
    for i, path in enumerate(files, 1):
        checksums[sha1_file(path)] = path
        if i % 100 == 0 or i == len(files):
            print(f"  Hashed {i}/{len(files)}...", file=sys.stderr)

    # 3. Find matches in Immich
    print("Checking against Immich...", file=sys.stderr)
    matches = find_matching_assets(args.server, args.api_key, checksums, args.dry_run)
    print(f"  Found {len(matches)} matching assets", file=sys.stderr)

    if not matches:
        print("No matching assets found in Immich.", file=sys.stderr)
        return

    # 4. Set favorites
    asset_ids = [aid for aid, _ in matches]
    if args.dry_run:
        print("\nWould favorite these files:", file=sys.stderr)
        for asset_id, path in matches:
            print(f"  {path.name} -> {asset_id}", file=sys.stderr)
    else:
        print("Setting favorites...", file=sys.stderr)

    count = set_favorites(args.server, args.api_key, asset_ids, args.dry_run)

    # 5. Summary
    unmatched = len(files) - len(matches)
    action = "Would favorite" if args.dry_run else "Favorited"
    print(f"\n{action} {count} assets ({unmatched} files had no match in Immich)")


if __name__ == "__main__":
    main()
