#!/usr/bin/env python3
"""Non-destructive conversion of ModelReady/Frozen split JSON to COCO with globally unique ann IDs.

Output layout: annotations/{train,valid,test}.json, images/{train,valid,test}/
Unmodified source remains immutable. Mapping + checksums included for provenance.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import shutil

SPLITS = ('train', 'valid', 'test')

def digest(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def annotation_dir_for(source: Path, split: str) -> Path:
    return source / 'splits' / split

def convert(source: Path, output: Path, images: str = 'copy'):
    source = source.resolve()
    output = output.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('Output must be a NEW sibling directory, not a source or parent/child of the source.')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite pre-existing output: {output}')
    if images not in {'copy', 'symlink', 'none'}:
        raise ValueError(images)

    # Preflight everything before writing anything.
    inputs = {}
    for split in SPLITS:
        split_dir = annotation_dir_for(source, split)
        ann_path = split_dir / 'annotations.json'
        if not ann_path.is_file():
            raise FileNotFoundError(ann_path)
        record = json.loads(ann_path.read_text(encoding='utf-8'))
        if not isinstance(record.get('annotations'), list) or not isinstance(record.get('images'), list):
            raise ValueError(f'{ann_path}: invalid COCO arrays')
        images_by_id = {int(im['id']): im for im in record['images']}
        if len(images_by_id) != len(record['images']):
            raise ValueError(f'{ann_path}: duplicate image IDs')
        if any(int(a['image_id']) not in images_by_id for a in record['annotations']):
            raise ValueError(f'{ann_path}: annotation references an unknown image')
        if images != 'none':
            for im in record['images']:
                img = split_dir / 'images' / im['file_name']
                if not img.is_file():
                    raise FileNotFoundError(f'Image missing for {split}: {img}. Use --images none for a metadata-only audit.')
        inputs[split] = (ann_path, record, split_dir)

    output.mkdir(parents=True, exist_ok=False)
    (output / 'annotations').mkdir()
    (output / 'images').mkdir()
    (output / 'provenance').mkdir()
    for split, (ann_path, record, split_dir) in inputs.items():
        original = record['annotations']
        id_counts = Counter(int(a['id']) for a in original)
        mapping_path = output / 'provenance' / f'{split}_annotation_id_mapping.csv'
        remapped = []
        with mapping_path.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['row_index','image_id','source_annotation_id','new_annotation_id','duplicate_source_id'])
            writer.writeheader()
            for row_idx, annotation in enumerate(original):
                ann = dict(annotation)
                ann['id'] = row_idx + 1
                # Preserve the old pair explicitly: every instance retains its provenance.
                ann['source_annotation_id'] = int(annotation['id'])
                ann['source_image_id'] = int(annotation['image_id'])
                ann['source_row_index'] = row_idx
                remapped.append(ann)
                writer.writerow(dict(row_index=row_idx,image_id=ann['image_id'],source_annotation_id=ann['source_annotation_id'],new_annotation_id=ann['id'],duplicate_source_id=int(id_counts[ann['source_annotation_id']] > 1)))
        converted = dict(record)
        converted['annotations'] = remapped
        target = output / 'annotations' / f'{split}.json'
        target.write_text(json.dumps(converted, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        out_img_dir = output / 'images' / split
        out_img_dir.mkdir()
        if images != 'none':
            for im in record['images']:
                image_name = im['file_name']
                img_src = split_dir / 'images' / image_name
                img_dst = out_img_dir / image_name
                img_dst.parent.mkdir(parents=True, exist_ok=True)
                if images == 'symlink':
                    img_dst.symlink_to(img_src.resolve())
                else:
                    shutil.copy2(img_src, img_dst)
        # Assert that no geometry or identity fields changed (ID and provenance excepted).
        for src_ann, new_ann in zip(original, converted['annotations']):
            assert all(src_ann[k] == new_ann[k] for k in src_ann if k != 'id')
        assert len({a['id'] for a in converted['annotations']}) == len(original)
        manifest = dict(split=split, image_count=len(record['images']), annotation_count=len(original), repeated_original_id_occurrences=sum(c-1 for c in id_counts.values()), input_json_sha256=digest(ann_path), output_json_sha256=digest(target), image_mode=images, source=str(ann_path))
        (output / 'provenance' / f'{split}_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
        print(f"{split:<5} images={manifest['image_count']:<5} annotations={len(original):<6} duplicate-id extras={manifest['repeated_original_id_occurrences']:<5} unique IDs verified")
    print(f'\nNEW derived dataset: {output}\nSource dataset untouched: {source}')
    if images == 'none':
        print('WARNING: --images none is for metadata validation ONLY; cannot be used for training/inference.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True,help='Source dataset containing splits/train|valid|test/annotations.json')
    p.add_argument('--output',type=Path,required=True,help='NEW output directory; must not exist')
    p.add_argument('--images',choices=['copy','symlink','none'],default='copy',help='copy for portable training, symlink for disk savings, none for metadata-only audit')
    args=p.parse_args()
    convert(args.source,args.output,args.images)
if __name__ == '__main__':
    main()
