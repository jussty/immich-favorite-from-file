#!/usr/bin/env python3
"""Set Immich assets as favorites based on files present in a local folder.

Uses a multi-stage matching strategy:
1. SHA-1 checksum match (fast, handles byte-identical files)
2. EXIF date + dimensions match (handles re-encoded files)
3. Pixel-data hash comparison (handles ambiguous date matches)

Requires: Pillow (pip install Pillow)
"""

import argparse
import hashlib
import io
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from PIL import Image

BATCH_SIZE = 1000
CACHE_FILE = ".sha1cache.json"
PIXEL_CACHE_FILE = ".pixelcache.json"


def sha1_file(path: Path) -> str:
    """Return hex-encoded SHA-1 digest of a file."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def pixel_hash(path_or_bytes) -> str | None:
    """Return SHA-1 of decoded pixel data (ignores JPEG encoding/EXIF).

    Returns None for non-image files (e.g. videos).
    """
    try:
        if isinstance(path_or_bytes, bytes):
            img = Image.open(io.BytesIO(path_or_bytes))
        else:
            img = Image.open(path_or_bytes)
        img = img.convert("RGB")
        return hashlib.sha1(img.tobytes()).hexdigest()
    except Exception:
        return None


def exif_datetime(path: Path) -> datetime | None:
    """Extract EXIF DateTimeOriginal as a datetime object, or None."""
    try:
        img = Image.open(path)
        exif = img.getexif()
        ifd = exif.get_ifd(0x8769)
        dt = ifd.get(36867)  # DateTimeOriginal
        if not dt:
            dt = exif.get(306)  # DateTime fallback
        if not dt:
            return None
        return datetime.strptime(dt, "%Y:%m:%d %H:%M:%S")
    except Exception:
        return None


def image_dimensions(path: Path) -> tuple[int, int] | None:
    """Return (width, height) of an image file."""
    try:
        img = Image.open(path)
        return img.size
    except Exception:
        return None


def _cache_key(path: Path) -> str:
    """Cache key combining path, size, and mtime."""
    stat = path.stat()
    return f"{path}|{stat.st_size}|{stat.st_mtime_ns}"


def load_cache(folder: Path) -> dict[str, str]:
    """Load hash cache from folder. Returns {cache_key: sha1_hex}."""
    cache_path = folder / CACHE_FILE
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_cache(folder: Path, cache: dict[str, str]) -> None:
    """Save hash cache to folder."""
    cache_path = folder / CACHE_FILE
    cache_path.write_text(json.dumps(cache))


def load_pixel_cache(folder: Path) -> dict[str, str]:
    """Load pixel hash cache. Returns {asset_id|checksum: pixel_hash}."""
    cache_path = folder / PIXEL_CACHE_FILE
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_pixel_cache(folder: Path, cache: dict[str, str]) -> None:
    """Save pixel hash cache."""
    cache_path = folder / PIXEL_CACHE_FILE
    cache_path.write_text(json.dumps(cache))


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


def api_download(server: str, api_key: str, endpoint: str) -> bytes:
    """Download binary data from Immich API."""
    url = f"{server.rstrip('/')}/api{endpoint}"
    req = Request(url)
    req.add_header("x-api-key", api_key)
    with urlopen(req) as resp:
        return resp.read()


def scan_folder(folder: Path) -> list[Path]:
    """Recursively find all files in folder, skipping hidden files/dirs."""
    files = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(folder).parts)
    )
    return files


def match_by_checksum(
    server: str, api_key: str, file_checksums: dict[str, Path]
) -> tuple[list[tuple[str, Path]], list[Path]]:
    """Stage 1: Match by SHA-1 checksum via bulk-upload-check.

    Returns (matches, unmatched_files).
    """
    items = list(file_checksums.items())
    matches = []
    matched_paths = set()

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
                matched_paths.add(local_path)
        done = min(i + BATCH_SIZE, len(items))
        print(f"    Checked {done}/{len(items)}...", file=sys.stderr)

    unmatched = [p for p in file_checksums.values() if p not in matched_paths]
    return matches, unmatched


def match_by_exif(
    server: str, api_key: str, files: list[Path]
) -> tuple[list[tuple[str, Path]], list[Path]]:
    """Stage 2: Match by EXIF date + dimensions.

    Returns (matches, unmatched_files).
    """
    matches = []
    unmatched = []

    for i, path in enumerate(files, 1):
        dt = exif_datetime(path)
        dims = image_dimensions(path)
        if i % 50 == 0 or i == len(files):
            print(f"    Searched {i}/{len(files)}...", file=sys.stderr)

        if not dt:
            unmatched.append(path)
            continue

        # Search 1-second window (assume no timezone offset)
        dt_str = dt.strftime("%Y-%m-%dT%H:%M:%S")
        result = api_request(server, api_key, "POST", "/search/metadata", {
            "takenAfter": dt_str + ".000Z",
            "takenBefore": dt_str + ".999Z",
        })
        candidates = result.get("assets", {}).get("items", [])

        if not candidates:
            unmatched.append(path)
            continue

        # Filter by dimensions (keep candidates with unknown dims)
        if dims:
            w, h = dims
            filtered = [
                c for c in candidates
                if (c.get("width") is None or c.get("height") is None)
                or (c.get("width") == w and c.get("height") == h)
            ]
            if filtered:
                candidates = filtered

        if len(candidates) == 1:
            matches.append((candidates[0]["id"], path))
        elif len(candidates) > 1:
            # Ambiguous — leave for pixel hash stage
            unmatched.append(path)
        else:
            unmatched.append(path)

    return matches, unmatched


def match_by_timezone_offset(
    server: str, api_key: str, files: list[Path]
) -> tuple[list[tuple[str, Path]], list[Path]]:
    """Stage 2.5: Try common timezone offsets for EXIF date mismatches.

    Returns (matches, unmatched_files).
    """
    matches = []
    unmatched = []
    # Common timezone offsets: -12 to +14 hours
    common_offsets = [-12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
                      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    for i, path in enumerate(files, 1):
        dt = exif_datetime(path)
        dims = image_dimensions(path)
        if i % 50 == 0 or i == len(files):
            print(f"    Searched {i}/{len(files)}...", file=sys.stderr)

        if not dt:
            unmatched.append(path)
            continue

        found = False
        # Try each timezone offset
        for offset in common_offsets:
            dt_offset = dt + timedelta(hours=offset)
            dt_str = dt_offset.strftime("%Y-%m-%dT%H:%M:%S")
            result = api_request(server, api_key, "POST", "/search/metadata", {
                "takenAfter": dt_str + ".000Z",
                "takenBefore": dt_str + ".999Z",
            })
            candidates = result.get("assets", {}).get("items", [])

            if not candidates:
                continue

            # Filter by dimensions
            if dims:
                w, h = dims
                filtered = [
                    c for c in candidates
                    if (c.get("width") is None or c.get("height") is None)
                    or (c.get("width") == w and c.get("height") == h)
                ]
                if filtered:
                    candidates = filtered

            if len(candidates) == 1:
                matches.append((candidates[0]["id"], path))
                found = True
                break
            elif len(candidates) > 1:
                # Ambiguous - continue to next offset or give up
                continue

        if not found:
            unmatched.append(path)

    return matches, unmatched


def match_by_pixel_hash(
    server: str, api_key: str, files: list[Path],
    pcache: dict[str, str],
) -> tuple[list[tuple[str, Path]], list[Path]]:
    """Stage 3: Match by downloading originals and comparing pixel data.

    Uses pcache to avoid re-downloading assets seen in previous runs.
    Tries timezone offsets to minimize candidate downloads.
    Returns (matches, unmatched_files).
    """
    matches = []
    unmatched = []
    common_offsets = [0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6, -7, 7, -8, 8,
                      -9, 9, -10, 10, -11, 11, -12, 12, 13, 14]

    for i, path in enumerate(files, 1):
        dt = exif_datetime(path)
        if i % 10 == 0 or i == len(files):
            print(f"    Compared {i}/{len(files)}...", file=sys.stderr)

        if not dt:
            unmatched.append(path)
            continue

        local_phash = pixel_hash(path)
        if not local_phash:
            unmatched.append(path)
            continue

        found = False
        # Try each timezone offset to minimize candidates
        for offset in common_offsets:
            dt_offset = dt + timedelta(hours=offset)
            dt_str = dt_offset.strftime("%Y-%m-%dT%H:%M:%S")
            result = api_request(server, api_key, "POST", "/search/metadata", {
                "takenAfter": dt_str + ".000Z",
                "takenBefore": dt_str + ".999Z",
            })
            candidates = result.get("assets", {}).get("items", [])

            if not candidates:
                continue

            # Download and compare pixel hashes for candidates at this offset
            for candidate in candidates:
                cache_key = f"{candidate['id']}|{candidate.get('checksum', '')}"
                if cache_key in pcache:
                    remote_phash = pcache[cache_key]
                else:
                    original = api_download(
                        server, api_key, f"/assets/{candidate['id']}/original"
                    )
                    remote_phash = pixel_hash(original)
                    if remote_phash:
                        pcache[cache_key] = remote_phash

                if remote_phash == local_phash:
                    matches.append((candidate["id"], path))
                    found = True
                    break

            if found:
                break

        if not found:
            unmatched.append(path)

    return matches, unmatched


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
    parser.add_argument(
        "--copy-unmatched",
        type=Path,
        metavar="DIR",
        help="Copy unmatched files to this directory",
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

    # 2. Compute checksums (with cache)
    print("Computing SHA-1 checksums...", file=sys.stderr)
    cache = load_cache(args.folder)
    file_checksums: dict[str, Path] = {}
    cache_hits = 0
    new_cache: dict[str, str] = {}
    for i, path in enumerate(files, 1):
        key = _cache_key(path)
        if key in cache:
            sha1 = cache[key]
            cache_hits += 1
        else:
            sha1 = sha1_file(path)
        new_cache[key] = sha1
        file_checksums[sha1] = path
        if i % 100 == 0 or i == len(files):
            print(f"  Hashed {i}/{len(files)} ({cache_hits} cached)...", file=sys.stderr)
    save_cache(args.folder, new_cache)

    all_matches: list[tuple[str, Path]] = []

    # 3. Stage 1: Checksum matching
    print("Stage 1: Matching by checksum...", file=sys.stderr)
    matches, remaining = match_by_checksum(args.server, args.api_key, file_checksums)
    all_matches.extend(matches)
    print(f"  Matched {len(matches)}, remaining {len(remaining)}", file=sys.stderr)

    # 4. Stage 2: EXIF date + dimensions (no timezone offset)
    if remaining:
        print("Stage 2: Matching by EXIF date + dimensions...", file=sys.stderr)
        matches, remaining = match_by_exif(args.server, args.api_key, remaining)
        all_matches.extend(matches)
        print(f"  Matched {len(matches)}, remaining {len(remaining)}", file=sys.stderr)

    # 5. Stage 2.5: Timezone offset retry
    if remaining:
        print("Stage 2.5: Matching with timezone offsets...", file=sys.stderr)
        matches, remaining = match_by_timezone_offset(args.server, args.api_key, remaining)
        all_matches.extend(matches)
        print(f"  Matched {len(matches)}, remaining {len(remaining)}", file=sys.stderr)

    # 6. Stage 3: Pixel hash comparison
    if remaining:
        print("Stage 3: Matching by pixel data (downloading originals)...", file=sys.stderr)
        pcache = load_pixel_cache(args.folder)
        matches, remaining = match_by_pixel_hash(
            args.server, args.api_key, remaining, pcache
        )
        save_pixel_cache(args.folder, pcache)
        all_matches.extend(matches)
        print(f"  Matched {len(matches)}, remaining {len(remaining)}", file=sys.stderr)

    if not all_matches:
        print("No matching assets found in Immich.", file=sys.stderr)
        return

    # 7. Set favorites
    asset_ids = [aid for aid, _ in all_matches]
    if args.dry_run:
        print("\nWould favorite these files:", file=sys.stderr)
        for asset_id, path in all_matches:
            print(f"  {path.name} -> {asset_id}", file=sys.stderr)
    else:
        print("Setting favorites...", file=sys.stderr)

    count = set_favorites(args.server, args.api_key, asset_ids, args.dry_run)

    # 8. Summary and unmatched file handling
    if remaining:
        print(f"\nUnmatched files ({len(remaining)}):", file=sys.stderr)
        for path in remaining:
            print(f"  {path.name}", file=sys.stderr)

        if args.copy_unmatched:
            if not args.dry_run:
                import shutil
                args.copy_unmatched.mkdir(parents=True, exist_ok=True)
                for path in remaining:
                    shutil.copy2(path, args.copy_unmatched / path.name)
                print(f"\nCopied {len(remaining)} unmatched files to {args.copy_unmatched}", file=sys.stderr)
            else:
                print(f"\nWould copy {len(remaining)} unmatched files to {args.copy_unmatched}", file=sys.stderr)

    print(f"\n{'Would favorite' if args.dry_run else 'Favorited'} {count} assets"
          f" ({len(remaining)} files had no match in Immich)")


if __name__ == "__main__":
    main()
