#!/usr/bin/env python3
"""Merge a Lamia valid split into train and retain its files in an archive."""

import argparse
import shutil
from datetime import datetime
from pathlib import Path

from _dataset_splits import (edit_dataset, exists, load_coco, move,
                             relative_name, replace_annotation, write_coco)


def merged_ids(train_rows, valid_rows):
    used = {row['id'] for row in train_rows}
    next_id = max(used | {row['id'] for row in valid_rows}, default=0) + 1
    mapping = {}
    for row in valid_rows:
        old_id = row['id']
        if old_id in used:
            mapping[old_id] = next_id
            next_id += 1
        else:
            mapping[old_id] = old_id
    return mapping


def merge_coco(train, valid):
    if train['categories'] != valid['categories']:
        raise ValueError('Train and valid COCO categories differ')
    image_ids = merged_ids(train['images'], valid['images'])
    annotation_ids = merged_ids(train['annotations'], valid['annotations'])
    return dict(
        train,
        images=train['images'] + [
            dict(image, id=image_ids[image['id']]) for image in valid['images']
        ],
        annotations=train['annotations'] + [
            dict(ann, id=annotation_ids[ann['id']], image_id=image_ids[ann['image_id']])
            for ann in valid['annotations']
        ],
    )


def merge(root, archive_name):
    train_dir = root / 'train'
    valid_dir = root / 'valid'
    archive = root / relative_name(archive_name)
    if archive.name in ('train', 'valid', 'test'):
        raise ValueError('Choose a separate archive directory name')
    for path in (train_dir, valid_dir):
        if not path.is_dir():
            raise FileNotFoundError(f'Missing data directory: {path}')

    with edit_dataset(root) as (staging, rollback):
        if exists(archive):
            raise FileExistsError(f'Archive already exists: {archive}')
        valid_annotations = sorted(root.glob('valid_annotations*.json'))
        train_annotations = set(root.glob('train_annotations*.json'))
        matching = {root / path.name.replace('valid_', 'train_', 1)
                    for path in valid_annotations}
        if not valid_annotations or matching != train_annotations:
            raise ValueError('Train and valid need matching COCO annotation files')

        updates = []
        for valid_path in valid_annotations:
            train_path = root / valid_path.name.replace('valid_', 'train_', 1)
            staged_file = staging / train_path.name
            train, train_names = load_coco(train_path)
            valid, valid_names = load_coco(valid_path)
            collisions = train_names & valid_names
            if collisions:
                raise ValueError(f'Image appears in both splits: {min(collisions)}')
            for name in valid_names:
                for directory in ('pre_event', 'post_event', 'target'):
                    path = valid_dir / directory / (
                        name.with_suffix('.tif') if directory == 'target' else name)
                    if not path.is_file():
                        raise FileNotFoundError(f'Missing valid sample file: {path}')
            write_coco(staged_file, merge_coco(train, valid))
            updates.append((train_path, staged_file, valid_path))

        files = sorted(path for path in valid_dir.rglob('*') if path.is_file())
        for source in files:
            destination = train_dir / source.relative_to(valid_dir)
            if exists(destination):
                raise FileExistsError(f'Destination already exists: {destination}')
        for valid_path in valid_annotations:
            if exists(valid_dir / valid_path.name):
                raise FileExistsError(f'Archive already contains {valid_path.name}')

        staged_files = staging / 'files'
        for source in files:
            relative = source.relative_to(valid_dir)
            staged_file = staged_files / relative
            staged_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, staged_file, follow_symlinks=False)
        for source in files:
            relative = source.relative_to(valid_dir)
            destination = train_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            move(staged_files / relative, destination, rollback)
        for train_path, staged_file, valid_path in updates:
            replace_annotation(train_path, staged_file, staging, rollback)
            move(valid_path, valid_dir / valid_path.name, rollback)
        move(valid_dir, archive, rollback)
    print(f'Merged {len(files)} files into {train_dir}')
    print(f'Archived validation files and annotations in {archive}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset_root', type=Path)
    parser.add_argument('--archive-name', help='defaults to valid_merged_YYYYmmdd_HHMMSS')
    args = parser.parse_args()
    archive_name = args.archive_name or datetime.now().strftime('valid_merged_%Y%m%d_%H%M%S')
    try:
        merge(args.dataset_root.expanduser().resolve(), archive_name)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f'{parser.prog}: {error}\n')


if __name__ == '__main__':
    main()
