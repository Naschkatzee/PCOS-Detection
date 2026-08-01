from __future__ import annotations

import argparse
import hashlib
import json
import imagehash
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def find_near_duplicates(
    df: pd.DataFrame,
    max_distance: int = 5,
) -> list[dict]:
    """Find visually similar images using perceptual hash distance."""
    records = df[["path", "label", "phash"]].to_dict("records")
    groups: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()

    for index, first in enumerate(records):
        first_hash = imagehash.hex_to_hash(first["phash"])

        for second in records[index + 1:]:
            pair = tuple(sorted((first["path"], second["path"])))

            if pair in seen_pairs:
                continue

            second_hash = imagehash.hex_to_hash(second["phash"])
            distance = first_hash - second_hash

            if distance <= max_distance:
                groups.append({
                    "file_1": first["path"],
                    "label_1": first["label"],
                    "file_2": second["path"],
                    "label_2": second["label"],
                    "phash_distance": int(distance),
                })
                seen_pairs.add(pair)

    return groups


def audit_dataset(root: Path, output_dir: Path) -> None:
    root = root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        raise FileNotFoundError(f'Dataset directory does not exist: {root}')

    image_paths = sorted(
        p for p in root.rglob('*')
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise RuntimeError(f'No image files found under {root}')

    rows: list[dict] = []
    corrupted: list[str] = []

    for path in image_paths:
        relative = path.relative_to(root)
        label = relative.parts[0] if len(relative.parts) > 1 else 'unknown'

        try:
            with Image.open(path) as image:
                image.verify()

            with Image.open(path) as image:
                fmt = image.format
                image_rgb = image.convert("RGB")

                width, height = image_rgb.size
                mode = image_rgb.mode
                perceptual_hash = str(imagehash.phash(image_rgb))

        except (UnidentifiedImageError, OSError, ValueError) as exc:
            corrupted.append(f"{relative}: {exc}")
            continue

        rows.append({
            "path": str(relative),
            "label": label,
            "extension": path.suffix.lower(),
            "format": fmt,
            "width": width,
            "height": height,
            "mode": mode,
            "file_size_bytes": path.stat().st_size,
            "sha256": sha256(path),
            "phash": perceptual_hash,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError('All discovered images were corrupted or unreadable.')

    duplicate_groups = (
        df.groupby('sha256')['path']
        .apply(list)
        .loc[lambda s: s.map(len) > 1]
        .sort_values(key=lambda s: s.map(len), ascending=False)
    )

    near_duplicates = find_near_duplicates(df, max_distance=5)

    class_counts = df['label'].value_counts().sort_index()
    resolutions = (
        df.groupby(['width', 'height'])
        .size()
        .sort_values(ascending=False)
        .rename('count')
        .reset_index()
    )

    summary = {
        'dataset_root': str(root),
        'valid_images': int(len(df)),
        'corrupted_images': int(len(corrupted)),
        'class_counts': {k: int(v) for k, v in class_counts.items()},
        'unique_resolutions': int(len(resolutions)),
        'exact_duplicate_groups': int(len(duplicate_groups)),
        'images_in_duplicate_groups': int(sum(len(x) for x in duplicate_groups)),
        'extensions': {k: int(v) for k, v in Counter(df['extension']).items()},
        'color_modes': {k: int(v) for k, v in Counter(df['mode']).items()},
        'near_duplicate_pairs': len(near_duplicates),
    }

    df.to_csv(output_dir / 'image_inventory.csv', index=False)
    resolutions.to_csv(output_dir / 'resolution_counts.csv', index=False)
    (output_dir / 'corrupted_files.txt').write_text('\n'.join(corrupted), encoding='utf-8')

    duplicate_payload = [
        {'sha256': digest, 'files': files, 'count': len(files)}
        for digest, files in duplicate_groups.items()
    ]
    (output_dir / 'exact_duplicates.json').write_text(
        json.dumps(duplicate_payload, indent=2), encoding='utf-8'
    )
    (output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8'
    )
    (output_dir / "near_duplicates.json").write_text(
    json.dumps(near_duplicates, indent=2),
    encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f'\nSaved audit files to: {output_dir.resolve()}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Audit an image-classification dataset.')
    parser.add_argument('--data', type=Path, required=True, help='Root folder containing class subfolders.')
    parser.add_argument('--output', type=Path, default=Path('reports/dataset_audit'))
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    audit_dataset(args.data, args.output)
