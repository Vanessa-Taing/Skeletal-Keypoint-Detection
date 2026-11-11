#!/usr/bin/env python3
"""Read-only, identity-checked four-panel annotation/model failure audit.

Consumes completed evaluation bundles (ZIPs or directories), and the unique-ID
COCO test dataset. Never changes input files; refuses an existing output dir.
"""
import argparse
import csv
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

MODELS = {
    'YOLO_E06': {'pred': 'yolo_predictions_pixels', 'gt': 'gt_pixels', 'vis': 'gt_visibility', 'img': 'image_ids', 'ann': 'annotation_ids', 'cat': 'category_ids', 'bbox': 'bboxes_pixels', 'valid': 'prediction_valid'},
    'HRNet_E08': {'pred': 'common_predictions', 'gt': 'common_gt', 'vis': 'common_gt_visibility', 'img': 'common_image_ids', 'ann': 'common_annotation_ids', 'cat': 'common_category_ids', 'bbox': 'common_bboxes'},
    'ViTPose_E02': {'pred': 'vitpose_predictions', 'gt': 'vitpose_gt', 'vis': 'vitpose_gt_visibility', 'img': 'vitpose_image_ids', 'ann': 'vitpose_annotation_ids', 'cat': 'vitpose_category_ids', 'bbox': 'vitpose_bboxes'},
}
COLORS = {'gt': (230, 70, 60), 'pred': (35, 132, 225), 'bbox': (255, 183, 40)}


def load_array(src: Path, stem: str):
    file_name = stem + '.npy'
    if src.is_file():
        if src.suffix.lower() != '.zip':
            raise ValueError(f'Expected ZIP bundle: {src}')
        with zipfile.ZipFile(src) as z:
            found = [n for n in z.namelist() if n.endswith('/' + file_name) or n == file_name]
            if len(found) != 1:
                raise ValueError(f'{src}: expected one {file_name}, found {len(found)}')
            return np.load(io.BytesIO(z.read(found[0])), allow_pickle=False)
    if not src.is_dir():
        raise FileNotFoundError(src)
    found = list(src.rglob(file_name))
    if len(found) != 1:
        raise ValueError(f'{src}: expected one {file_name}, found {len(found)}')
    return np.load(found[0], allow_pickle=False)


def load_model(src: Path, model: str):
    d = {field: load_array(src, stem) for field, stem in MODELS[model].items()}
    n = len(d['img'])
    if any(len(d[field]) != n for field in d):
        raise ValueError(f'{model}: inconsistent record count')
    if d['pred'].shape != (n, 6, 2) or d['gt'].shape != (n, 6, 2):
        raise ValueError(f'{model}: expected [N,6,2] prediction and GT arrays')
    if d['vis'].shape != (n, 6) or d['bbox'].shape != (n, 4):
        raise ValueError(f'{model}: unexpected visibility or bbox dimensions')
    if 'valid' in d and d['valid'].shape != (n, 6):
        raise ValueError(f'{model}: invalid mask shape')
    keys = [(int(im), int(an)) for im, an in zip(d['img'], d['ann'])]
    if len(set(keys)) != n:
        raise ValueError(f'{model}: duplicate (image_id,annotation_id) pair')
    d['index'] = dict(zip(keys, range(n)))
    return d


def load_coco(root):
    p = root / 'annotations' / 'test.json'
    if not p.is_file():
        raise FileNotFoundError(f'Missing {p}')
    data = json.loads(p.read_text())
    images = {int(v['id']): v for v in data['images']}
    ann = {(int(v['image_id']), int(v['id'])): v for v in data['annotations']}
    if len(ann) != len(data['annotations']):
        raise ValueError('Duplicate COCO composite annotation identities')
    categories = {int(v['id']): v['name'] for v in data['categories']}
    return images, ann, categories


def prepare_records(root, paths):
    images, anns, categories = load_coco(root)
    models = {k: load_model(Path(v), k) for k, v in paths.items()}
    keys = set(anns)
    for name, d in models.items():
        if set(d['index']) != keys:
            raise ValueError(f'{name}: identities do not match unique-ID COCO test set')
    rows = []
    for key in sorted(keys):
        gt_ann = anns[key]
        coco_xy = np.asarray(gt_ann['keypoints'], dtype=float).reshape(6, 3)[:, :2]
        coco_vis = np.asarray(gt_ann['keypoints'], dtype=float).reshape(6, 3)[:, 2] > 0
        bbox = np.asarray(gt_ann['bbox'], dtype=float)
        record = {
            'image_id': key[0], 'annotation_id': key[1],
            'file_name': images[key[0]]['file_name'], 'class': categories[int(gt_ann['category_id'])],
            'category_id': int(gt_ann['category_id']), 'bbox': bbox,
            'gt': coco_xy, 'visible': coco_vis, 'scores': {}, 'preds': {}, 'pred_valid': {},
        }
        if record['class'] == 'Channel' and coco_vis[2] and coco_vis[3]:
            record['G3_G4_px'] = float(np.linalg.norm(coco_xy[2] - coco_xy[3]))
        else:
            record['G3_G4_px'] = math.nan
        if record['class'] == 'Channel' and coco_vis[0] and coco_vis[1]:
            record['G1_G2_px'] = float(np.linalg.norm(coco_xy[0] - coco_xy[1]))
        else:
            record['G1_G2_px'] = math.nan
        for name, m in models.items():
            j = m['index'][key]
            if not np.allclose(m['gt'][j][coco_vis], coco_xy[coco_vis], atol=0.003, rtol=0):
                raise ValueError(f'{name}: GT coordinate mismatch at {key}')
            if not np.array_equal(m['vis'][j] > 0, coco_vis):
                raise ValueError(f'{name}: GT visibility mismatch at {key}')
            if int(m['cat'][j]) != record['category_id']:
                raise ValueError(f'{name}: GT class mismatch at {key}')
            if not np.allclose(m['bbox'][j], bbox, atol=0.003, rtol=0):
                raise ValueError(f'{name}: GT bbox mismatch at {key}')
            valid = m['valid'][j].astype(bool) if 'valid' in m else np.ones(6, dtype=bool)
            dist = np.linalg.norm(m['pred'][j] - coco_xy, axis=1)
            threshold = max(float(bbox[2]), float(bbox[3]), 1e-8) * 0.05
            correct = int(np.count_nonzero(coco_vis & valid & np.isfinite(dist) & (dist <= threshold)))
            denominator = int(coco_vis.sum())
            record['scores'][name] = correct / denominator if denominator else math.nan
            record['preds'][name] = m['pred'][j]
            record['pred_valid'][name] = valid
        rows.append(record)
    return rows, images


def spatial_spread(candidates):
    """Interleave spatial positions rather than taking adjacent boxes in dense scenes."""
    ordered = sorted(candidates, key=lambda r: (
        float(r['bbox'][0]) + float(r['bbox'][2])/2,
        float(r['bbox'][1]) + float(r['bbox'][3])/2))
    if not ordered:
        return []
    todo = [(0, len(ordered))]
    out = []
    while todo:
        new = []
        for lo, hi in todo:
            if lo >= hi: continue
            m = (lo + hi) // 2
            out.append(ordered[m])
            new.extend([(lo,m), (m+1,hi)])
        todo = new
    return out


def balanced_order(candidates):
    """Cycle through source images, spreading candidates spatially per image."""
    groups = {}
    for r in candidates:
        groups.setdefault(r['file_name'], []).append(r)
    names = sorted(groups, key=lambda k: (-len(groups[k]), k))
    lists = {name: iter(spatial_spread(groups[name])) for name in names}
    out = []
    while names:
        remain = []
        for name in names:
            try:
                out.append(next(lists[name]))
                remain.append(name)
            except StopIteration:
                pass
        names = remain
    return out


def choose_samples(records, per_stratum=8, max_per_image=35):
    """Target failure modes while avoiding adjacent objects in dense scenes.

    Image diversity is limited by the actual test split; Channel occurs in
    only two images, so max_per_image must be reasonably high for review.
    """
    channel = [r for r in records if r['class'] == 'Channel']
    others = [r for r in records if r['class'] != 'Channel']
    q = {
        '01_channel_near_G3G4': [r for r in channel if r['G3_G4_px'] < 1],
        '02_channel_separated_G3G4': [r for r in channel if r['G3_G4_px'] >= 15],
        '03_channel_model_disagreement': [r for r in channel if abs(r['scores']['ViTPose_E02'] - r['scores']['HRNet_E08']) >= 0.5],
        '04_other_shared_failures': [r for r in others if all(r['scores'][m] <= 0.25 for m in MODELS)],
        '05_vitpose_strong_yolo_weak': [r for r in records if r['scores']['ViTPose_E02'] >= 0.75 and r['scores']['YOLO_E06'] <= 0.25],
        '06_channel_near_G1G2': [r for r in channel if r['G1_G2_px'] < 1],
    }
    result = []
    seen = set()
    counts = {}
    for name, candidates in q.items():
        taken = 0
        for r in balanced_order(candidates):
            key = (r['image_id'], r['annotation_id'])
            if key in seen or counts.get(r['image_id'], 0) >= max_per_image:
                continue
            seen.add(key)
            counts[r['image_id']] = counts.get(r['image_id'], 0) + 1
            result.append((name, r))
            taken += 1
            if taken >= per_stratum:
                break
    return result, {s: len(v) for s, v in q.items()}


def image_path(root, name):
    # Preserve source naming; never fuzzy-match to another image.
    p = root / 'images' / 'test' / name
    if not p.is_file():
        raise FileNotFoundError(f'Missing test image: {p}')
    return p


def font():
    for p in ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf']:
        if Path(p).is_file():
            return ImageFont.truetype(p, 15)
    return ImageFont.load_default()


def crop_bounds(bbox, size, pad_factor=0.35):
    W,H=size
    x,y,w,h = map(float,bbox)
    margin = max(18, max(w,h)*pad_factor)
    x0,y0 = max(0,math.floor(x-margin)),max(0,math.floor(y-margin))
    x1,y1 = min(W,math.ceil(x+w+margin)),min(H,math.ceil(y+h+margin))
    if x1<=x0 or y1<=y0:
        raise ValueError('Degenerate crop rectangle')
    return x0,y0,x1,y1


def panel(img, bounds, bbox, gt, visible, pred, valid, heading, score, panel_w=430, panel_h=410):
    F=font(); head=55; max_img_h=panel_h-head-8
    x0,y0,x1,y1 = bounds
    crop=img.crop(bounds)
    sf=min(panel_w/crop.width,max_img_h/crop.height)
    scaled=crop.resize((max(1,round(crop.width*sf)),max(1,round(crop.height*sf))),Image.Resampling.BILINEAR)
    canvas=Image.new('RGB',(panel_w,panel_h),'#f4f6f8')
    draw=ImageDraw.Draw(canvas)
    draw.text((10,9),heading,fill='#111827',font=F)
    if score is not None:
        draw.text((10,30),f'Fixed PCK: {score:.1%}',fill='#374151',font=F)
    left=(panel_w-scaled.width)//2; top=head+(max_img_h-scaled.height)//2
    canvas.paste(scaled,(left,top))
    def xy(p): return (left+(float(p[0])-x0)*sf,top+(float(p[1])-y0)*sf)
    d=ImageDraw.Draw(canvas)
    gx,gy,gw,gh=map(float,bbox)
    bb0=xy([gx,gy]);bb1=xy([gx+gw,gy+gh]);d.rectangle([bb0,bb1],outline=COLORS['bbox'],width=2)
    for j,(p,vis) in enumerate(zip(gt,visible)):
        if not vis: continue
        px,py=xy(p)
        if 0<=px<panel_w and head<=py<panel_h:
            d.line((px-5,py,px+5,py),fill=COLORS['gt'],width=2)
            d.line((px,py-5,px,py+5),fill=COLORS['gt'],width=2)
            d.text((px+5,py+5),f'G{j+1}',fill=COLORS['gt'],font=F)
    if pred is not None:
        for j,(p,vis,v) in enumerate(zip(pred,visible,valid)):
            if not(vis and v):continue
            px,py=xy(p)
            if 0<=px<panel_w and head<=py<panel_h:
                d.ellipse((px-4,py-4,px+4,py+4),fill=COLORS['pred'],outline='white')
                d.text((px+5,py-13),f'G{j+1}',fill=COLORS['pred'],font=F)
    return canvas


def make_montage(root, r):
    with Image.open(image_path(root,r['file_name'])) as im:
        img=im.convert('RGB')
    b=crop_bounds(r['bbox'],img.size)
    models=['YOLO_E06','HRNet_E08','ViTPose_E02']
    p=[panel(img,b,r['bbox'],r['gt'],r['visible'],None,None,'Ground truth',None)]
    for model in models:
        heading = model
        if not bool(np.any(r['pred_valid'][model] & r['visible'])):
            heading += ' (no matched detection)'
        p.append(panel(img,b,r['bbox'],r['gt'],r['visible'],r['preds'][model],r['pred_valid'][model],heading,r['scores'][model]))
    out=Image.new('RGB',(430*4,410+45),'white')
    for i,pnl in enumerate(p):out.paste(pnl,(430*i,0))
    d=ImageDraw.Draw(out)
    d.text((8,416),f"{r['file_name']}  {r['class']}  image_id={r['image_id']}  annotation_id={r['annotation_id']}  red=GT  blue=prediction  yellow=bbox",fill='#222222',font=font())
    return out


def save_review(root, selected, pool_sizes, output, all_records=None):
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {output}')
    # Fail before writing partial output if any source image is unavailable.
    for _stratum,r in selected: image_path(root,r['file_name'])
    output.mkdir(parents=True)
    (output/'overlays').mkdir()
    worksheet=[]
    for i,(stratum,r) in enumerate(selected,1):
        fname=f'{i:03d}_{stratum}_im{r["image_id"]}_ann{r["annotation_id"]}.png'
        make_montage(root,r).save(output/'overlays'/fname,optimize=True)
        worksheet.append({
            'stratum':stratum, 'image_id':r['image_id'],'annotation_id':r['annotation_id'],
            'file_name':r['file_name'],'class':r['class'], 'overlay_file':str(Path('overlays')/fname),
            'G3_G4_distance_px':f'{r["G3_G4_px"]:.3f}' if math.isfinite(r['G3_G4_px']) else '',
            'G1_G2_distance_px':f'{r["G1_G2_px"]:.3f}' if math.isfinite(r['G1_G2_px']) else '',
            'YOLO_PCK':f'{r["scores"]["YOLO_E06"]:.3f}',
            'HRNet_PCK':f'{r["scores"]["HRNet_E08"]:.3f}',
            'ViTPose_PCK':f'{r["scores"]["ViTPose_E02"]:.3f}',
            'GT_schema_valid_(yes/no/uncertain)':'',
            'failure_type_(schema/annotation/occlusion/viewpoint/lighting/model/other)':'',
            'visibility_correct_(yes/no/uncertain)':'', 'reviewer_notes':'',
        })
    with (output/'review_worksheet.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(worksheet[0]));w.writeheader();w.writerows(worksheet)
    from collections import Counter, defaultdict
    distribution = defaultdict(Counter)
    for r in (all_records or []):
        distribution[r['class']][r['file_name']] += 1
    with (output/'class_image_distribution.csv').open('w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['class','file_name','annotated_objects'])
        for cls in sorted(distribution):
            for name,num in distribution[cls].most_common():
                w.writerow([cls,name,num])
    report=['# Three-model visual audit', '',
        f'- Reviewed dataset: `{root}`', f'- Total sampled objects: {len(worksheet)}',
        '- Scoring: fixed-index PCK@0.05 in original-image pixel space.',
        '- Model comparisons remain protocol-mixed: YOLO predicted boxes; HRNet/ViTPose GT boxes.',
        '- Red crosses = ground truth; blue dots = predicted points; yellow rectangle = ground-truth bbox.',
        '- Sampling is deliberately *targeted*, not representative of dataset-level prevalence.',
        '', '## Strata (eligible cases / selected)', '']
    for key,size in pool_sizes.items():
        report.append(f'- `{key}`: {size} eligible, {sum(s==key for s,_ in selected)} selected')
    report += ['', '## Test-set image concentration', '']
    for cls in sorted(distribution):
        counts=distribution[cls]
        report.append(f'- **{cls}**: {sum(counts.values())} objects in {len(counts)} images; largest image {counts.most_common(1)[0][0]} contains {counts.most_common(1)[0][1]}')
    report += ['', '**Do not treat 598 Channel objects from two images as 598 independent visual scenes.**',
              'These targeted test samples are for diagnosing existing results, not tuning hyperparameters on the held-out test set.',
              '', '## Reviewer instructions', '',
       'Open PNGs in `overlays/` and fill `review_worksheet.csv` without editing the dataset.',
       'For a suspected wrong label, record the **image_id + annotation_id + affected G number(s)** and why.',
       'Distinguish physical keypoint ambiguity from equivalence under mirror/rotation.',
       'Do not infer occlusion, lighting, or viewpoint categories solely from PCK values.',
       'If a full-image context is required, open the matching source in the dataset `images/test/` folder.',
       '', '## Model-ground-truth consistency', '',
       'All three prediction bundles were checked for identical composite identities and COCO GT visible coordinates/bboxes prior to sampling.',
       '']
    (output/'SUMMARY.md').write_text('\n'.join(report))
    print(f'Wrote {len(worksheet)} object montages and worksheet to {output}')


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset',type=Path,required=True,help='InnoCount_COCO_UniqueIDs_v1')
    ap.add_argument('--yolo',type=Path,required=True,help='E06 results ZIP or extracted folder')
    ap.add_argument('--hrnet',type=Path,required=True,help='E08 results ZIP or extracted folder')
    ap.add_argument('--vitpose',type=Path,required=True,help='E02 results ZIP or extracted folder')
    ap.add_argument('--output',type=Path,required=True,help='New output folder: must not exist')
    ap.add_argument('--per-stratum',type=int,default=8)
    ap.add_argument('--max-per-image',type=int,default=35)
    args=ap.parse_args(argv)
    if args.per_stratum<1 or args.max_per_image<1:ap.error('sample counts must be >=1')
    rows,_=prepare_records(args.dataset,{'YOLO_E06':args.yolo,'HRNet_E08':args.hrnet,'ViTPose_E02':args.vitpose})
    selected,sizes=choose_samples(rows,args.per_stratum,args.max_per_image)
    if not selected:raise ValueError('No candidates selected for review')
    save_review(args.dataset,selected,sizes,args.output,rows)

if __name__=='__main__': main()
