"""Shared file and COCO helpers for the train/valid tools."""

import json
import shutil
from contextlib import ExitStack, contextmanager
from pathlib import Path


def exists(path):
    return path.exists() or path.is_symlink()


def relative_name(name):
    path = Path(name)
    if (not name or path.name != name or name in ('.', '..')
            or any(char in name for char in '/\\:')):
        raise ValueError(f'Expected a single file or directory name, got {name!r}')
    return path


def load_coco(path):
    with path.open(encoding='utf-8') as stream:
        data = json.load(stream)
    image_ids = {image['id'] for image in data['images']}
    names = {relative_name(image['file_name']) for image in data['images']}
    annotation_ids = {annotation['id'] for annotation in data['annotations']}
    if len(image_ids) != len(data['images']) or len(names) != len(data['images']):
        raise ValueError(f'Duplicate image IDs or file names in {path}')
    if len(annotation_ids) != len(data['annotations']):
        raise ValueError(f'Duplicate annotation IDs in {path}')
    if any(ann['image_id'] not in image_ids for ann in data['annotations']):
        raise ValueError(f'Annotation references an unknown image in {path}')
    return data, names


def write_coco(path, data):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False)
        stream.write('\n')


@contextmanager
def edit_dataset(root):
    """Serialize the two commands and keep original files available for rollback."""
    staging = root / '.lamia-train-valid'
    staging.mkdir()
    rollback = ExitStack()
    try:
        yield staging, rollback
    except BaseException:
        rollback.close()
        shutil.rmtree(staging)
        raise
    else:
        rollback.pop_all()
        shutil.rmtree(staging)


def move(source, destination, rollback):
    if exists(destination):
        raise FileExistsError(f'Destination already exists: {destination}')
    source.rename(destination)
    rollback.callback(destination.rename, source)


def replace_annotation(path, staged_file, staging, rollback):
    move(path, staging / f'original-{path.name}', rollback)
    move(staged_file, path, rollback)


def sample_files(split_dir, image_name):
    """Paths associated with a COCO image in the Lamia dataset layout."""
    yield split_dir / 'pre_event' / image_name
    yield split_dir / 'post_event' / image_name
    mask = split_dir / 'target' / image_name.with_suffix('.tif')
    yield mask
    instance = split_dir / 'target_instance' / image_name.with_suffix('.json')
    if exists(instance):
        yield instance
