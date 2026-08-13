#!/usr/bin/env python3
import argparse, json, math, random
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

SEED = 19
SPLITS = ('train','valid','test')
CLASSES = {0:'Channel',1:'T-beam',2:'HI-Beam',3:'Angle Bar',4:'Sheet Pile'}
COLORS = {0:(30,144,255),1:(255,140,0),2:(220,20,60),3:(148,0,211),4:(0,150,70)}


def font(size=16, bold=False):
    paths = ['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
             'C:/Windows/Fonts/arialbd.ttf' if bold else 'C:/Windows/Fonts/arial.ttf']
    for p in paths:
        try: return ImageFont.truetype(p, size)
        except Exception: pass
    return ImageFont.load_default()

F = font(16); FS = font(13); FL = font(22, True)

def load_all(root):
    records=[]
    for split in SPLITS:
        p=root/'splits'/split/'annotations.json'
        if not p.exists(): continue
        d=json.loads(p.read_text(encoding='utf-8'))
        cats={c['id']:c for c in d['categories']}
        imgs={i['id']:i for i in d['images']}
        anns=defaultdict(list)
        for a in d['annotations']: anns[a['image_id']].append(a)
        for iid,img in imgs.items():
            records.append({'split':split,'image':img,'categories':cats,'annotations':anns[iid]})
    if not records: raise FileNotFoundError('No frozen COCO annotations found.')
    return records

def flags(ann,img,cat):
    k=ann.get('keypoints',[]); expected=len(cat.get('keypoints',[]))
    out={'malformed':len(k)!=expected*3,'occluded':False,'out_of_image':False,'out_of_bbox':False}
    if out['malformed']: return out
    W,H=img['width'],img['height']; b=ann.get('bbox')
    for i in range(expected):
        x,y,v=k[3*i:3*i+3]
        if v==1: out['occluded']=True
        if v>0:
            if x<0 or x>=W or y<0 or y>=H: out['out_of_image']=True
            if b:
                bx,by,bw,bh=b
                if not (bx<=x<=bx+bw and by<=y<=by+bh): out['out_of_bbox']=True
    return out

def dashed(draw,p1,p2,fill,width=3,dash=8):
    x1,y1=p1; x2,y2=p2; dx=x2-x1; dy=y2-y1; dist=math.hypot(dx,dy)
    if not dist: return
    ux,uy=dx/dist,dy/dist; pos=0
    while pos<dist:
        a=pos; b=min(pos+dash,dist)
        draw.line([(x1+ux*a,y1+uy*a),(x1+ux*b,y1+uy*b)],fill=fill,width=width)
        pos += dash*2

def label(draw,x,y,text,fill=(255,255,255)):
    bb=draw.textbbox((x,y),text,font=FS); pad=2
    draw.rectangle((bb[0]-pad,bb[1]-pad,bb[2]+pad,bb[3]+pad),fill=(0,0,0))
    draw.text((x,y),text,font=FS,fill=fill)

def visualize(image,ann,cat,record,outfile):
    im=image.convert('RGB'); d=ImageDraw.Draw(im); W,H=im.size; color=COLORS.get(ann['category_id'],(255,255,0))
    b=ann.get('bbox')
    if b:
        x,y,w,h=b; d.rectangle((x,y,x+w,y+h),outline=color,width=4)
        label(d,x,max(0,y-22),f"{cat['name']} | ann {ann.get('id')}",color)
    kp=ann.get('keypoints',[]); names=cat.get('keypoints',[]); points={}
    for i in range(len(names)):
        x,y,v=kp[3*i:3*i+3]; points[i+1]=(x,y,v)
    for edge in cat.get('skeleton',[]):
        if len(edge)!=2 or edge[0] not in points or edge[1] not in points: continue
        x1,y1,v1=points[edge[0]]; x2,y2,v2=points[edge[1]]
        if v1==0 or v2==0: continue
        outside=not(0<=x1<W and 0<=y1<H and 0<=x2<W and 0<=y2<H)
        if outside:
            dashed(
                d,
                (x1, y1),
                (x2, y2),
                fill=color,
                width=3,
            )
        else:
            d.line(
                [(x1, y1), (x2, y2)],
                fill=color,
                width=3,
            )
        if not outside: d.line([(x1,y1),(x2,y2)],fill=color,width=3)
    for i,name in enumerate(names,1):
        x,y,v=kp[3*(i-1):3*i]
        c=(0,220,80) if v==2 else (255,190,0) if v==1 else (180,180,180)
        inside=0<=x<W and 0<=y<H
        if inside:
            r=7; d.ellipse((x-r,y-r,x+r,y+r),fill=c,outline=(0,0,0),width=2); label(d,x+8,y-9,f'K{i}',c)
        else:
            cx=max(0,min(W-1,x)); cy=max(0,min(H-1,y)); r=9
            d.ellipse((cx-r,cy-r,cx+r,cy+r),outline=(255,60,60),width=4)
            label(d,cx+10,cy-10,f'K{i} OUT',(255,60,60))
    d.rectangle((0,0,W,30),fill=(0,0,0)); d.text((7,6),f"{cat['name']} | ann={ann.get('id')} | image={record['image']['id']} | split={record['split']}",font=FS,fill=(255,255,255))
    im.save(outfile,quality=95)

def main():
    ap=argparse.ArgumentParser(description='Create visual QA for frozen InnoCount keypoints.')
    ap.add_argument('--dataset',required=True,type=Path)
    ap.add_argument('--output',default=Path('InnoCount_KeyPoint_Visual_QA'),type=Path)
    ap.add_argument('--samples-per-class',default=5,type=int)
    args=ap.parse_args()
    if args.output.exists(): raise FileExistsError(f'Output already exists: {args.output}')
    random.seed(SEED); args.output.mkdir(parents=True)
    records=load_all(args.dataset)
    schema={}
    by_class=defaultdict(list)
    special={k:defaultdict(list) for k in ('occluded','out_of_image','out_of_bbox')}
    for r in records:
        for cid,cat in r['categories'].items():
            if cid in CLASSES: schema[cid]={'id':cid,'name':cat['name'],'keypoints':cat.get('keypoints',[]),'skeleton':cat.get('skeleton',[])}
        for a in r['annotations']:
            if a['category_id'] not in CLASSES: continue
            cid=a['category_id']; by_class[cid].append((r,a)); fl=flags(a,r['image'],r['categories'][cid])
            for key in special:
                if fl[key]: special[key][cid].append((r,a))
    (args.output/'keypoint_schema.json').write_text(json.dumps(schema,indent=2),encoding='utf-8')
    with (args.output/'keypoint_schema.txt').open('w',encoding='utf-8') as f:
        for cid in sorted(schema):
            c=schema[cid]; f.write(f"Class {cid}: {c['name']}\nKeypoints: {len(c['keypoints'])}\n")
            for i,n in enumerate(c['keypoints'],1): f.write(f'  K{i}: {n}\n')
            f.write('Skeleton:\n'); [f.write(f'  K{e[0]} -- K{e[1]}\n') for e in c['skeleton']]; f.write('\n')
    csv=['split,image_id,file_name,annotation_id,class_id,class_name,occluded,out_of_image,out_of_bbox,malformed']
    for r in records:
        for a in r['annotations']:
            cid=a['category_id']
            if cid not in CLASSES: continue
            fl=flags(a,r['image'],r['categories'][cid])
            if any(fl.values()): csv.append(','.join(map(str,[r['split'],r['image']['id'],r['image']['file_name'],a['id'],cid,CLASSES[cid],fl['occluded'],fl['out_of_image'],fl['out_of_bbox'],fl['malformed']])))
    (args.output/'special_annotations.csv').write_text('\n'.join(csv)+'\n',encoding='utf-8')
    dirs=['representative','occluded','out_of_image','out_of_bbox']
    for x in dirs: (args.output/x).mkdir()
    for cid in sorted(CLASSES):
        candidates=by_class[cid][:]; random.shuffle(candidates); selected=[]; seen=set()
        for r,a in candidates:
            src=r['image']['file_name']
            if src not in seen: selected.append((r,a)); seen.add(src)
            if len(selected)>=args.samples_per_class: break
        classdir=args.output/'representative'/CLASSES[cid].replace(' ','_'); classdir.mkdir()
        for i,(r,a) in enumerate(selected,1):
            src=args.dataset/'splits'/r['split']/'images'/r['image']['file_name']; im=Image.open(src)
            visualize(im,a,r['categories'][cid],r,classdir/f'{i:02d}_{r["split"]}_image_{r["image"]["id"]}_ann_{a["id"]}.jpg')
    for typ in dirs[1:]:
        for cid in sorted(CLASSES):
            vals=special[typ][cid][:10]
            if not vals: continue
            classdir=args.output/typ/CLASSES[cid].replace(' ','_'); classdir.mkdir()
            for i,(r,a) in enumerate(vals,1):
                src=args.dataset/'splits'/r['split']/'images'/r['image']['file_name']; im=Image.open(src)
                visualize(im,a,r['categories'][cid],r,classdir/f'{i:02d}_{r["split"]}_image_{r["image"]["id"]}_ann_{a["id"]}.jpg')
    (args.output/'README.txt').write_text('Read-only visual QA report. K1/K2/... are local class-specific indices; original Roboflow keypoint names are preserved in keypoint_schema.json. No coordinates or annotations are modified. Out-of-image keypoints are not clipped.\n',encoding='utf-8')
    print(f'Loaded {len(records)} images.')
    print(f'Output: {args.output}')

if __name__=='__main__': main()
