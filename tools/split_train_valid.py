#!/usr/bin/env python3
"""Randomly split a Lamia COCO training set into train and valid."""

import argparse
import random
from pathlib import Path

from _dataset_splits import (edit_dataset, exists, load_coco, move,
                             replace_annotation, sample_files, write_coco)


def subset(data, names):
    images = [image for image in data['images']
              if Path(image['file_name']) in names]
    image_ids = {image['id'] for image in images}
    return dict(
        data,
        images=images,
        annotations=[ann for ann in data['annotations']
                     if ann['image_id'] in image_ids],
    )


def split(root, ratio, seed):
    train_dir = root / 'train'
    valid_dir = root / 'valid'
    if not train_dir.is_dir():
        raise FileNotFoundError(f'Missing training directory: {train_dir}')
    if not 0 < ratio < 1:
        raise ValueError('--valid-ratio must be between 0 and 1')

    with edit_dataset(root) as (staging, rollback):
        if exists(valid_dir) or any(root.glob('valid_annotations*.json')):
            raise FileExistsError(f'Existing valid directory or annotations under {root}')
        train_annotations = sorted(root.glob('train_annotations*.json'))
        if not train_annotations:
            raise FileNotFoundError(f'Missing train_annotations*.json under {root}')
        primary = root / 'train_annotations.json'
        if primary not in train_annotations:
            primary = train_annotations[0]
        primary_data, names = load_coco(primary)
        if len(names) < 2:
            raise ValueError('At least two training images are required')
        count = min(len(names) - 1, max(1, round(len(names) * ratio)))
        selected = set(random.Random(seed).sample(sorted(names), count))

        sources = sorted({source for name in selected
                          for source in sample_files(train_dir, name)})
        for source in sources:
            if not source.is_file():
                raise FileNotFoundError(f'Missing training sample file: {source}')

        updates = []
        for source in train_annotations:
            data, source_names = (primary_data, names) if source == primary else load_coco(source)
            valid_name = source.name.replace('train_', 'valid_', 1)
            train_out = staging / source.name
            valid_out = staging / valid_name
            write_coco(train_out, subset(data, source_names - selected))
            write_coco(valid_out, subset(data, selected))
            updates.append((source, train_out, valid_out))

        valid_dir.mkdir()
        rollback.callback(valid_dir.rmdir)
        for directory in sorted({source.parent.name for source in sources}):
            destination = valid_dir / directory
            destination.mkdir()
            rollback.callback(destination.rmdir)
        for source in sources:
            move(source, valid_dir / source.relative_to(train_dir), rollback)
        for source, train_out, valid_out in updates:
            replace_annotation(source, train_out, staging, rollback)
            move(valid_out, root / valid_out.name, rollback)
    print(f'Train images: {len(names) - count}   Valid images: {count}   Seed: {seed}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset_root', type=Path)
    parser.add_argument('--valid-ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    try:
        split(args.dataset_root.expanduser().resolve(), args.valid_ratio, args.seed)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f'{parser.prog}: {error}\n')


if __name__ == '__main__':
    main()
