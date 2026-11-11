#!/usr/bin/env python3
"""Read-only per-object v1/v2 T-beam annotation comparison for 25 matched pairs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import math
from PIL import Image, ImageDraw
import numpy as np
from review_three_model_failures import crop_bounds, font

C1=(211,54,71)
C2=(38,118,226)

def load_train(root):
    p = root/'splits'/'train'/'annotations.json'
    coco=json.loads(p.read_text())
    images={int(im['id']): im['file_name'] for im in coco['images']}
    ann={}
    for a in coco['annotations']:
        key=(images[int(a['image_id'])],int(a['id']))
        if key in ann: raise ValueError(f'Duplicate annotation pair {key}')
        ann[key]=a
    return ann


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def draw_pairs(image, bounds, a,b, title, filename):
    x0,y0,x1,y1=bounds
    crop=image.crop(bounds)
    W,H=650,550; head=45
    scale=min(W/crop.width,(H-head-10)/crop.height)
    resized=crop.resize((max(1,round(crop.width*scale)),max(1,round(crop.height*scale))),Image.Resampling.BILINEAR)
    target=Image.new('RGB',(W,H),'#f8fafc');dx=(W-resized.width)//2;dy=head+((H-head)-resized.height)//2
    target.paste(resized,(dx,dy));d=ImageDraw.Draw(target);f=font()
    d.text((12,12),title,fill='#111827',font=f)
    def px(pt):return (dx+(float(pt[0])-x0)*scale,dy+(float(pt[1])-y0)*scale)
    for item,color,prefix in [(a,C1,'v1'),(b,C2,'v2')]:
        x,y,w,h=map(float,item['bbox']); d.rectangle((px((x,y)),px((x+w,y+h))),outline=color,width=2)
        k=np.asarray(item['keypoints'],dtype=float).reshape(-1,3)
        for j,(xx,yy,v) in enumerate(k):
            if v<=0:continue
            ox,oy=px((xx,yy))
            d.ellipse((ox-4,oy-4,ox+4,oy+4),outline=color,width=3)
            d.text((ox+5,oy+(0 if prefix=='v1' else 13)),f'{prefix}:G{j+1}',fill=color,font=f)
    return target


def process(v1,v2,pairs,output):
    if output.exists():raise FileExistsError(f'Output already exists: {output}')
    a1=load_train(v1); a2=load_train(v2)
    with pairs.open(newline='') as f: rows=list(csv.DictReader(f))
    if not rows:raise ValueError('Empty pair list')
    checks=[]
    for r in rows:
        name=r['image']; k1=(name,int(r['v1_annotation_id']));k2=(name,int(r['v2_annotation_id']))
        if k1 not in a1 or k2 not in a2:raise ValueError(f'Missing pair {k1} {k2}')
        path1=v1/'splits'/'train'/'images'/name;path2=v2/'splits'/'train'/'images'/name
        if not path1.is_file() or not path2.is_file():raise FileNotFoundError(f'Missing image {path1} or {path2}')
        checks.append((r,path1,path2,a1[k1],a2[k2]))
    output.mkdir(parents=True);(output/'overlays').mkdir()
    worksheet=[];hashes={}
    for i,(r,path1,path2,a,b) in enumerate(checks,1):
        name=r['image']
        if name not in hashes:hashes[name]=(sha256(path1),sha256(path2))
        with Image.open(path1) as im:im1=im.convert('RGB')
        with Image.open(path2) as im:im2=im.convert('RGB')
        if im1.size != im2.size: raise ValueError(f'Images differ in dimensions: {name}')
        bboxes=np.array([a['bbox'],b['bbox']],dtype=float)
        x=min(bboxes[:,0]);y=min(bboxes[:,1]);xend=max(bboxes[:,0]+bboxes[:,2]);yend=max(bboxes[:,1]+bboxes[:,3])
        region=crop_bounds((x,y,xend-x,yend-y),im1.size,pad_factor=0.20)
        pa=draw_pairs(im1,region,a,b,f'Image v1: {name} / ID {r["v1_annotation_id"]}',name)
        pb=draw_pairs(im2,region,a,b,f'Image v2: {name} / ID {r["v2_annotation_id"]}',name)
        canvas=Image.new('RGB',(1300,585),'white')
        canvas.paste(pa,(0,0));canvas.paste(pb,(650,0))
        d=ImageDraw.Draw(canvas)
        d.text((12,560),f'v1 red / v2 blue  IoU={float(r["match_iou"]):.3f}  max Δkeypoint={float(r["max_keypoint_displacement_px"]):.1f}px  images_sha256_equal={hashes[name][0]==hashes[name][1]}',fill='#222',font=font())
        file=f'{i:02d}_{name.replace(".jpg","")}_v1{r["v1_annotation_id"]}_v2{r["v2_annotation_id"]}.png'
        canvas.save(output/'overlays'/file,optimize=True)
        worksheet.append({'image':name,'v1_annotation_id':r['v1_annotation_id'],'v2_annotation_id':r['v2_annotation_id'],'bbox_iou':r['match_iou'],'max_keypoint_displacement_px':r['max_keypoint_displacement_px'],'same_source_image_hash':int(hashes[name][0]==hashes[name][1]),'overlay':f'overlays/{file}','same_physical_object_(yes/no/uncertain)':'','keypoint_order_change_(yes/no/uncertain)':'','preferred_label_(v1/v2/neither/uncertain)':'','notes':''})
    with (output/'tbeam_review_worksheet.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(worksheet[0]));w.writeheader();w.writerows(worksheet)
    with (output/'image_sha256.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['file_name','v1_image_sha256','v2_image_sha256','same_bytes'])
        for name,(h1,h2) in sorted(hashes.items()):w.writerow([name,h1,h2,int(h1==h2)])
    (output/'SUMMARY.md').write_text(
      '# ModelReady v1/v2 T-beam discrepancy audit\n\n'
      f'Generated {len(checks)} paired crop overlays from {len(hashes)} distinct training images.\n\n'
      'Red = v1 annotation, blue = v2 annotation. Each panel draws **both versions** on the corresponding source image. '
      'These correspondences were supplied from an earlier bbox-based Hungarian audit and still need physical-object verification.\n\n'
      '**Do not edit either source dataset.** Record per-pair decisions in the worksheet; build only a separately versioned corrected derivative if needed.\n')
    print(f'Wrote {len(checks)} paired overlays to {output}')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v1',type=Path,required=True)
    p.add_argument('--v2',type=Path,required=True)
    p.add_argument('--pairs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    v=p.parse_args(argv)
    process(v.v1,v.v2,v.pairs,v.output)
if __name__=='__main__':main()
