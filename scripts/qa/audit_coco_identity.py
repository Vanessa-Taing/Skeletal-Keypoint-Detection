#!/usr/bin/env python3
"""Verify COCO annotation ID uniqueness and per-image lookups; read-only."""
from __future__ import annotations
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def audit(folder: Path):
    folder=folder.resolve()
    pycoco = None
    try:
        from pycocotools.coco import COCO
        pycoco = COCO
    except ImportError:
        print('pycocotools unavailable: raw-JSON audit only; install it to exercise COCO.getAnnIds/loadAnns')
    failures=[]
    for split in ('train','valid','test'):
        file = folder/'annotations'/f'{split}.json'
        if not file.is_file():
            raise FileNotFoundError(file)
        source=json.loads(file.read_text(encoding='utf-8'))
        annotations=source['annotations']
        counts=Counter(int(a['id']) for a in annotations)
        colliding=sum(c-1 for c in counts.values())
        by_image=defaultdict(list)
        for a in annotations:
            by_image[int(a['image_id'])].append(a)
        print(f'{split}: {len(source["images"])} images, {len(annotations)} annotations, {colliding} reused global annotation IDs')
        if colliding:
            failures.append(f'{split}: {colliding} reused annotation IDs')
        if pycoco:
            data=pycoco(str(file))
            actual_total=0
            for image in source['images']:
                image_id=int(image['id'])
                anns=data.loadAnns(data.getAnnIds(imgIds=[image_id]))
                actual_total+=len(anns)
                expected=by_image.get(image_id,[])
                if len(anns)!=len(expected):
                    failures.append(f'{split}: image {image_id} yielded {len(anns)} rows instead of {len(expected)}')
                for a in anns:
                    if int(a['image_id'])!=image_id:
                        failures.append(f'{split}: image {image_id} yielded foreign annotation image_id={a["image_id"]}')
            if actual_total != len(annotations):
                failures.append(f'{split}: loaded {actual_total} rows vs raw {len(annotations)}')
    if failures:
        print('\nFAILED COCO integrity audit (first 15 issues):')
        for err in failures[:15]: print(' -',err)
        raise SystemExit(1)
    print('\nPASS: globally unique annotation IDs within each split, COCO image assignments verified where available.')

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True,help='Prepared COCO root with annotations/train.json etc')
    audit(p.parse_args().dataset)
