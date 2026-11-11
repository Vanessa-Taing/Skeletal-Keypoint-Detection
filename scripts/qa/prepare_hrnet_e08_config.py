#!/usr/bin/env python3
"""Prepare a clean E08 HRNet evaluation config; never alter frozen data.

Copies the archived E08 test YAML to a NEW config output, pointing
DATASET.ROOT to a prepared unique-ID COCO export and OUTPUT_DIR to a
NEW experiment directory. This DOES NOT start inference or training.
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path
import yaml


def prepare(repo: Path, source_config: Path, dataset: Path, output_dir: Path, config_output: Path):
    repo=repo.resolve()
    def loc(p):
        p=Path(p).expanduser()
        return p.resolve() if p.is_absolute() else (repo/p).resolve()
    source_config=loc(source_config);dataset=loc(dataset)
    output_dir=loc(output_dir);config_output=loc(config_output)
    if not source_config.is_file(): raise FileNotFoundError(source_config)
    if not dataset.is_dir(): raise FileNotFoundError(dataset)
    if dataset.name == 'InnoCount_KeyPoint_Frozen_v3_Optimized':
        raise ValueError('Do not target immutable frozen dataset')
    ann_file=dataset/'annotations/test.json'
    img_dir=dataset/'images/test'
    if not ann_file.is_file() or not img_dir.is_dir() or not any(img_dir.iterdir()):
        raise FileNotFoundError(f'Incomplete evaluation dataset: {ann_file}, {img_dir}')
    obj=json.loads(ann_file.read_text(encoding='utf-8'))
    annotations=obj['annotations']; images=obj['images']
    if len({int(a['id']) for a in annotations}) != len(annotations):
        raise ValueError('Duplicate annotation IDs; unique-ID preparation failed')
    if len({int(i['id']) for i in images}) != len(images):
        raise ValueError('Duplicate image IDs')
    image_ids={int(x['id']) for x in images}
    if any(int(a['image_id']) not in image_ids for a in annotations):
        raise ValueError('One or more annotations refer to absent images')
    missing=[x['file_name'] for x in images if not (img_dir / x['file_name']).is_file()]
    if missing: raise FileNotFoundError(f'{len(missing)} missing images, first: {missing[:3]}')
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'Output directory not empty: {output_dir}')
    if config_output.exists():raise FileExistsError(f'Config already exists: {config_output}')
    cfg=yaml.safe_load(source_config.read_text(encoding='utf-8'))
    cfg.setdefault('DATASET',{})['ROOT']=str(dataset)
    cfg['DATASET']['TEST_SET']='test'
    cfg['OUTPUT_DIR']=str(output_dir)
    cfg.setdefault('TEST',{}).pop('MODEL_FILE',None)
    config_output.parent.mkdir(parents=True,exist_ok=True)
    config_output.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    print(f'Config created: {config_output}')
    print(f'Unique-ID test set: {len(images)} images, {len(annotations)} annotations')
    print(f'HRNet dataset root: {dataset}')
    print(f'New output dir: {output_dir} (not created)')
    print('Model checkpoint is supplied at inference time using TEST.MODEL_FILE')
    print('Note: custom InnoCount loader feeds GT-bbox crops, so this is pose-only inference.')
    return config_output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',default='.')
    p.add_argument('--source-config',default='experiments/HRNet_Runs/E08_HRNet_W32_FinalTest/2026-09-04_00-17-06/hrnet_eval_config.yaml')
    p.add_argument('--dataset',default='datasets/InnoCount_COCO_UniqueIDs_v1')
    p.add_argument('--output-dir',default='experiments/HRNet_Runs/E08_Reeval_UniqueIDs_MetricFix')
    p.add_argument('--config-output',default='configs/innocount/hrnet_e08_uniqueids_test_2026_10_22.yaml')
    a=p.parse_args()
    prepare(Path(a.repo),Path(a.source_config),Path(a.dataset),Path(a.output_dir),Path(a.config_output))
if __name__=='__main__': main()
