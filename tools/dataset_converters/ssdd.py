# Copyright (c) OpenMMLab. All rights reserved.
"""Download SSDD and lay it out as the COCO tree the HGS configs expect.

The configs under ``configs/_hgs_retina`` read::

    <out>/annotations/train.json    928 images, 2041 annotations
    <out>/annotations/test.json     232 images,  546 annotations
    <out>/images/train/
    <out>/images/test/

That is the official SSDD split (the 928 is train+val of the upstream YOLO
packaging). The source used here is the ``dronefreak/SSDD`` mirror on the
HuggingFace Hub, which carries all 1160 official images and, importantly, a
``metadata.jsonl`` per split that already stores absolute COCO ``xywh`` boxes
-- so no XML/YOLO coordinate conversion is involved.

huggingface.co and Google Drive are unreachable from some networks while
``hf-mirror.com`` works, hence the configurable ``--endpoint``. The official
release (``TianwenZhang0825/Official-SSDD``) only offers Google Drive / Baidu
Pan links, both of which are impractical to script here.

Usage::

    python tools/dataset_converters/ssdd.py \
        --out data/SSDD/raw/Official-SSDD-OPEN/BBox_SSDD/coco_style
"""
import argparse
import concurrent.futures as futures
import json
import os
import os.path as osp
import re
import shutil
import sys
import tempfile
import urllib.request

from PIL import Image

# Upstream YOLO splits -> config splits. train + valid together reproduce the
# official 928-image training set.
SPLIT_MAP = {'train': ('train', 'valid'), 'test': ('test',)}
EXPECTED = {'train': (928, 2041), 'test': (232, 546)}
USER_AGENT = {'User-Agent': 'curl/8.5.0'}


def _get_json(url):
    """Fetch JSON, returning the body and the ``Link`` header for paging."""
    request = urllib.request.Request(url, headers=USER_AGENT)
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response), response.headers.get('Link', '')


def list_images(endpoint, repo, split):
    """Return every ``.jpg`` path under one upstream split, paging as needed."""
    url = (f'{endpoint}/api/datasets/{repo}/tree/main/data/images/{split}'
           f'?recursive=true&expand=true&limit=100')
    paths = []
    while url:
        # urllib sends no User-Agent by default and the mirror answers 403.
        data, link = _get_json(url)
        paths.extend(f['path'] for f in data
                     if f['type'] == 'file' and f['path'].endswith('.jpg'))
        match = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = (match.group(1).replace('https://huggingface.co', endpoint)
               if match else None)
    return sorted(paths)


def download(endpoint, repo, path, dest, retries=4):
    """Fetch one file from the Hub, skipping anything already on disk."""
    if osp.exists(dest) and osp.getsize(dest) > 0:
        return None
    url = f'{endpoint}/datasets/{repo}/resolve/main/{path}'
    tmp = dest + '.part'
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers=USER_AGENT)
            with urllib.request.urlopen(request, timeout=120) as response:
                with open(tmp, 'wb') as handle:
                    while chunk := response.read(1 << 16):
                        handle.write(chunk)
            os.replace(tmp, dest)
            return None
        except Exception as error:  # noqa: BLE001 - reported to the caller
            if attempt == retries - 1:
                return f'{path}: {error}'
    return None


def fetch_split(endpoint, repo, split, staging, workers):
    """Download the images and the metadata file of one upstream split."""
    images = list_images(endpoint, repo, split)
    if not images:
        raise RuntimeError(f'no images listed for upstream split {split!r}')
    target = osp.join(staging, split)
    os.makedirs(target, exist_ok=True)

    jobs = [(path, osp.join(target, osp.basename(path))) for path in images]
    jobs.append((f'data/images/{split}/metadata.jsonl',
                 osp.join(target, 'metadata.jsonl')))

    failures = []
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(
                lambda job: download(endpoint, repo, job[0], job[1]), jobs):
            if result:
                failures.append(result)
    if failures:
        raise RuntimeError(
            f'{len(failures)} download(s) failed, first: {failures[0]}')
    print(f'  {split}: {len(images)} images')
    return target


def build_coco(source_dirs, image_dir):
    """Build one COCO dict and stage its images into ``image_dir``."""
    os.makedirs(image_dir, exist_ok=True)
    images, annotations = [], []
    image_id = annotation_id = 0
    for source_dir in source_dirs:
        with open(osp.join(source_dir, 'metadata.jsonl')) as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        for record in records:
            name = record['file_name']
            destination = osp.join(image_dir, name)
            if not osp.exists(destination):
                shutil.copy2(osp.join(source_dir, name), destination)
            with Image.open(destination) as image:
                width, height = image.size
            image_id += 1
            images.append(
                dict(id=image_id, file_name=name, width=width, height=height))
            for bbox in record['objects']['bbox']:
                x, y, w, h = (float(value) for value in bbox)
                annotation_id += 1
                annotations.append(
                    dict(
                        id=annotation_id,
                        image_id=image_id,
                        category_id=1,
                        bbox=[x, y, w, h],
                        area=w * h,
                        iscrowd=0,
                        segmentation=[]))
    return dict(
        info=dict(
            description='SAR Ship Detection Dataset (SSDD), official split, '
                        'COCO format',
            version='1.0'),
        licenses=[],
        images=images,
        annotations=annotations,
        categories=[dict(id=1, name='ship', supercategory='none')])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--out', required=True,
        help='output COCO root, i.e. the data_root the configs should use')
    parser.add_argument(
        '--endpoint', default='https://hf-mirror.com',
        help='HuggingFace endpoint (huggingface.co is blocked on some hosts)')
    parser.add_argument(
        '--repo', default='dronefreak/SSDD', help='Hub dataset repo id')
    parser.add_argument(
        '--workers', type=int, default=12, help='parallel downloads')
    parser.add_argument(
        '--cache', default=None,
        help='keep the raw download here instead of a temporary directory')
    parser.add_argument(
        '--skip-count-check', action='store_true',
        help='do not fail when the split sizes differ from the expected ones')
    return parser.parse_args()


def main():
    args = parse_args()
    cache = args.cache or tempfile.mkdtemp(prefix='ssdd_raw_')
    os.makedirs(cache, exist_ok=True)
    print(f'downloading from {args.endpoint} ({args.repo}) into {cache}')

    try:
        for target_split, source_splits in SPLIT_MAP.items():
            source_dirs = [
                fetch_split(args.endpoint, args.repo, source, cache,
                            args.workers) for source in source_splits
            ]
            coco = build_coco(
                source_dirs, osp.join(args.out, 'images', target_split))
            annotation_path = osp.join(args.out, 'annotations',
                                       f'{target_split}.json')
            os.makedirs(osp.dirname(annotation_path), exist_ok=True)
            with open(annotation_path, 'w') as handle:
                json.dump(coco, handle)

            got = (len(coco['images']), len(coco['annotations']))
            expected = EXPECTED[target_split]
            status = 'OK' if got == expected else 'MISMATCH'
            print(f'{target_split}: {got[0]} images, {got[1]} annotations '
                  f'(expected {expected[0]}/{expected[1]}) -> {status}')
            if got != expected and not args.skip_count_check:
                # A different split silently invalidates any comparison against
                # previously reported numbers, so refuse to pretend it worked.
                raise RuntimeError(
                    f'{target_split} split does not match the official one; '
                    f'pass --skip-count-check only if that is intended')
    finally:
        if not args.cache:
            shutil.rmtree(cache, ignore_errors=True)

    print(f'\ndone: {osp.abspath(args.out)}')


if __name__ == '__main__':
    sys.exit(main())
