# immich-favorite-from-folder

Sync your phone's favorite photos to [Immich](https://immich.app) -- even when your gallery app doesn't support exporting favorites.

## The problem

Most phone gallery apps let you mark photos as favorites, but almost none provide a way to export that information. When you self-host your photos with Immich, your favorites don't come along for the ride. This is especially painful when you remove photos from your phone to free up storage -- your carefully curated favorites are just gone.

## The workaround

The one thing every gallery app *can* do is copy files to a folder. So:

1. Select all your favorites in your gallery app
2. Copy them to a folder (e.g. "Foto Favoriten")
3. Sync that folder to your computer
4. Run this tool to mark the matching Immich assets as favorites

## How matching works

Gallery apps often re-encode photos when copying them, so the files in your favorites folder aren't byte-identical to what was uploaded to Immich. This tool uses a three-stage matching strategy to handle that:

| Stage | Method | Speed | Handles |
|-------|--------|-------|---------|
| 1 | SHA-1 file checksum | Fast (batched) | Byte-identical copies |
| 2 | EXIF date + image dimensions | Medium (1 API call/file) | Re-encoded files with preserved EXIF |
| 3 | Decoded pixel data comparison | Slow (downloads originals) | Ambiguous date matches (burst shots) |

Each stage only processes files that the previous stage couldn't match.

## Installation

Requires Python 3.10+ and [Pillow](https://pillow.readthedocs.io/).

```bash
pip install Pillow
```

## Usage

```bash
python immich_favorite_from_folder.py \
  --server http://your-immich-server:2283 \
  --api-key YOUR_API_KEY \
  --folder /path/to/favorites \
  --dry-run
```

Remove `--dry-run` once you've verified the matches look correct.

### Options

| Flag | Env var | Description |
|------|---------|-------------|
| `--server` | `IMMICH_SERVER_URL` | Immich server URL |
| `--api-key` | `IMMICH_API_KEY` | Immich API key ([how to create one](https://immich.app/docs/features/command-line-interface#obtain-the-api-key)) |
| `--folder` | | Folder containing your favorite files |
| `--dry-run` | | Show what would be changed without modifying anything |

### Example output

```
Scanning /home/user/favorites...
  Found 667 files
Computing SHA-1 checksums...
  Hashed 667/667 (0 cached)...
Stage 1: Matching by checksum...
  Matched 63, remaining 604
Stage 2: Matching by EXIF date + dimensions...
  Matched 529, remaining 75
Stage 3: Matching by pixel data (downloading originals)...
  Matched 22, remaining 53

Favorited 614 assets (53 files had no match in Immich)
```

## Caching

The tool uses multiple caches to speed up subsequent runs:

- **`.sha1cache.json`**: SHA-1 checksums of local files (keyed by path, size, and modification time)
- **`.pixelcache.json`**: Pixel hashes of downloaded Immich assets (keyed by asset ID and checksum)
- **`.cache/`**: Downloaded original assets from Stage 3 (enables instant reprocessing after dry-run)

All caches are stored in the scanned folder and persist across runs.

## Limitations

- **Add-only**: The tool only sets `isFavorite=true`. It never removes favorites or modifies any other asset metadata.
- **Unmatched files**: Files without EXIF data and no checksum match in Immich cannot be matched. These are typically files that were never uploaded to Immich in the first place.
- **API key permissions**: Your API key needs `asset.read` and `asset.update` permissions.

## License

[MIT](LICENSE)
