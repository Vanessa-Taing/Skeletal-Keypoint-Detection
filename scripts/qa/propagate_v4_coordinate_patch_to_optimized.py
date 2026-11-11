#!/usr/bin/env python3
"""Propagate the *reviewed* v4->v8 901-point coordinate-only patch to a new
Frozen_v3_Optimized-derived dataset without altering original split or IDs.

This tool does NOT edit v4, the original optimized dataset, or historical runs.
Use --bbox-proposals only AFTER reviewing the patch kit's warnings/CSV.

Example:
  python propagate_v4_coordinate_patch_to_optimized.py \
    --optimized datasets/InnoCount_KeyPoint_Frozen_v3_Optimized \
    --manifest datasets/InnoCount_KeyPoint_Frozen_v3/reports/master_manifest.csv \
    --changes ~/Downloads/v4_v8_patch_kit/changed_keypoints.csv \
    --output datasets/InnoCount_KeyPoint_Frozen_v3_Optimized_Corrected_v1
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

SPLITS = ('train','valid','test')

def load_json(p):
    with p.open(encoding='utf8') as f: return json.load(f)

def sha256(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def flags(ann, threshold=1.):
    a=ann['keypoints']; pts=[(a[i],a[i+1],a[i+2]) for i in range(0,len(a),3)]
    return [(i,j) for i,(x,y,v) in enumerate(pts) for j,(xx,yy,vv) in enumerate(pts) if j>i and v>0 and vv>0 and math.hypot(x-xx,y-yy)<threshold]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--optimized',required=True,type=Path)
    p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--changes',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--bbox-proposals',type=Path,default=None, help='OPTIONAL: reviewed bbox expansion proposals; changes bbox and area as well as point coordinates')
    a=p.parse_args()
    root=a.optimized.expanduser().resolve(); output=a.output.expanduser().resolve(); manifest_path=a.manifest.expanduser().resolve(); patch_path=a.changes.expanduser().resolve()
    if not root.is_dir(): p.error(f'Missing dataset: {root}')
    if output.exists(): p.error(f'Refusing to overwrite: {output}')
    if root in output.parents or output == root: p.error('Output must not be inside source dataset')
    if not patch_path.is_file() or not manifest_path.is_file(): p.error('Missing manifest or changed_keypoints.csv')
    with manifest_path.open(newline='',encoding='utf8') as f: manifest=list(csv.DictReader(f))
    m={int(r['master_id']):r for r in manifest}
    assert len(m)==len(manifest), 'Duplicated master ID in manifest'
    with patch_path.open(newline='',encoding='utf8') as f: changes=list(csv.DictReader(f))
    assert len(changes)==901, f'Expected exactly 901 reviewed coordinate changes, got {len(changes)}'
    ch={}
    for r in changes:
        k=(r['roboflow_image_uuid'],int(r['v4_annotation_id']))
        if k in ch: raise ValueError(f'Duplicate patch key {k}')
        ch[k]=r
    proposals={}
    if a.bbox_proposals:
        with a.bbox_proposals.open(newline='',encoding='utf8') as f: prs=list(csv.DictReader(f))
        for r in prs:
            k=(r['roboflow_image_uuid'],int(r['v4_annotation_id']))
            if k not in ch: raise ValueError(f'BBox proposal not in coordinate patch: {k}')
            proposals[k]=r
        if len(proposals)!=194: raise ValueError(f'Expected 194 bbox proposals, got {len(proposals)}')
    # Preflight all annotations in memory first; don't create output until all pass.
    modified={}; seen=set(); original_counts={}; extra_data=[]; before_flags=0; after_flags=0
    for split in SPLITS:
        src=root/'splits'/split/'annotations.json'
        if not src.exists():raise FileNotFoundError(src)
        data=load_json(src)
        ids={im['id']:im for im in data['images']}
        prior=collections.Counter()
        for ann in data['annotations']:
            im_id=ann['image_id']; img=ids[im_id]
            if im_id not in m: raise ValueError(f'Image {im_id} missing manifest')
            row=m[im_id]; uuid=Path(row['source_name']).stem
            if row['file_name']!=img['file_name']:raise ValueError(f'Manifest/optimized image mismatch: {im_id}')
            k=(uuid,int(ann['id']))
            if k not in ch:continue
            if k in seen:raise ValueError(f'Patch target appears more than once: {k}')
            seen.add(k); change=ch[k]
            if int(row['orig_image_id'])!=int(change['v4_image_id']):raise ValueError(f'Original image ID mismatch {k}')
            if ann['category_id']!=0:raise ValueError(f'Expected Channel category 0: {k}')
            index=int(change['keypoint_changed_0_based']);i=index*3
            if int(ann['keypoints'][i+2])==0:raise ValueError(f'Inactive patched keypoint {k}')
            for offset,key in [(0,'old_x'),(1,'old_y')]:
                if abs(float(ann['keypoints'][i+offset])-float(change[key]))>1e-6:
                    raise ValueError(f'Old keypoint differs from patch log {k} component {offset}: {ann["keypoints"][i+offset]} vs {change[key]}')
            if not flags(ann): raise ValueError(f'Expected near-coincident pair before patch: {k}')
            before_flags+=1
            ann['keypoints'][i:i+2]=[float(change['new_x']),float(change['new_y'])]
            if flags(ann):raise ValueError(f'Remaining coincident pair after patch: {k}')
            for xx,yy in [(ann['keypoints'][j],ann['keypoints'][j+1]) for j in range(0,len(ann['keypoints']),3) if ann['keypoints'][j+2]>0]:
                if not (0<=xx<=img['width'] and 0<=yy<=img['height']):raise ValueError(f'Patched object has point outside image {k}')
            if k in proposals:
                pr=proposals[k]
                oldbox=json.loads(pr['original_bbox'])
                if any(abs(float(x)-float(y))>1e-6 for x,y in zip(ann['bbox'],oldbox)):
                    raise ValueError(f'Existing bbox mismatch for {k}')
                newbox=json.loads(pr['proposed_min_bbox_with_0p02px_margin'])
                if newbox[0]>oldbox[0]+1e-5 or newbox[1]>oldbox[1]+1e-5 or newbox[0]+newbox[2]<oldbox[0]+oldbox[2]-1e-5 or newbox[1]+newbox[3]<oldbox[1]+oldbox[3]-1e-5:
                    raise ValueError(f'Proposal does not contain original bbox: {k}')
                ann['bbox']=[float(x) for x in newbox]
                ann['area']=round(ann['bbox'][2]*ann['bbox'][3],6)
            extra_data.append(dict(split=split,image=img['file_name'],original_uuid=uuid,annotation_id=ann['id'],changed_keypoint=index+1,bbox_adjusted=k in proposals))
        original_counts[split]=dict(images=len(data['images']),annotations=len(data['annotations']))
        modified[split]=data
    missing=set(ch)-seen
    if missing:raise ValueError(f'{len(missing)} patch records not matched. Examples: {list(missing)[:5]}')
    assert len(seen)==before_flags==901
    if original_counts!={'train':{'images':594,'annotations':12200}, 'valid':{'images':74,'annotations':1551},'test':{'images':74,'annotations':1626}}:
        raise ValueError(f'Unexpected split counts: {original_counts}')
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.{output.name}_',dir=output.parent) as temp:
        tmp=Path(temp)/'data'
        shutil.copytree(root,tmp)
        for split,data in modified.items():
            out=tmp/'splits'/split/'annotations.json'
            out.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
        report={'status':'validated','source_optimized':str(root),'source_manifest':str(manifest_path),'source_patch_log':str(patch_path),'source_patch_sha256':sha256(patch_path),'patch_records':len(seen),'changed_objects_per_split':dict(collections.Counter(item['split'] for item in extra_data)), 'bbox_proposals_used':bool(proposals),'bbox_adjustments':len(proposals),'split_counts':original_counts,'unchanged_dataset_structure':True,'notes':'New derived data; previous source metadata and historical freeze hashes are not assertions about the updated annotation JSON. Preserve provenance from the original and this report.'}
        (tmp/'patch_propagation_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
        with (tmp/'applied_patch_annotations.csv').open('w',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=list(extra_data[0]));w.writeheader();w.writerows(extra_data)
        tmp.rename(output)
    print(json.dumps(report,indent=2))
    print('DATASET READY:',output)

if __name__=='__main__':main()
