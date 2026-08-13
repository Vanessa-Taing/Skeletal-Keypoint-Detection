#!/usr/bin/env python3
import argparse, csv, json, random, math
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

SPLITS = ("train", "valid", "test")
CLASSES = ("Channel", "T-beam", "HI-Beam", "Angle Bar", "Sheet Pile")

def font(size, bold=False):
    names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    ]
    for n in names:
        if Path(n).exists():
            return ImageFont.truetype(n, size)
    return ImageFont.load_default()

def load_records(dataset):
    out = []
    for split in SPLITS:
        p = dataset/"splits"/split/"annotations.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        cats = {c["id"]: c for c in data["categories"]}
        anns = defaultdict(list)
        for a in data["annotations"]:
            anns[a["image_id"]].append(a)
        for im in data["images"]:
            for a in anns.get(im["id"], []):
                out.append((split, im, a, cats[a["category_id"]]))
    return out

def status(im, ann, x, y, v):
    if v <= 0: return "not_labeled"
    W,H = im["width"], im["height"]
    bx,by,bw,bh = ann["bbox"]; ex,ey = bx+bw, by+bh
    oi = x < 0 or x > W or y < 0 or y > H
    ob = x < bx or x > ex or y < by or y > ey
    loc = "out_image+out_bbox" if oi and ob else "out_image" if oi else "out_bbox" if ob else "inside"
    vis = "visible" if v == 2 else "occluded" if v == 1 else "unknown"
    return f"{vis};{loc}"

def bucket(im, ann):
    k = ann.get("keypoints", [])
    vals = [status(im, ann, k[i], k[i+1], k[i+2]) for i in range(0,len(k),3)]
    if any("out_image+out_bbox" in s for s in vals): return "both_out"
    if any("out_image" in s for s in vals): return "out_image"
    if any("out_bbox" in s for s in vals): return "out_bbox"
    if any("occluded" in s for s in vals): return "occluded"
    return "normal"

def choose(records, n, seed):
    rng = random.Random(seed)
    by = defaultdict(list)
    for r in records: by[r[3]["name"]].append(r)
    result = {}
    for cls, rows in by.items():
        chosen = []
        # Prefer split diversity.
        for sp in SPLITS:
            x = [r for r in rows if r[0] == sp]
            if x: chosen.append(rng.choice(x))
        # Then difficult examples.
        for b in ("both_out","out_image","out_bbox","occluded","normal"):
            x = [r for r in rows if r not in chosen and bucket(r[1],r[2]) == b]
            if x: chosen.append(rng.choice(x))
            if len(chosen) >= n: break
        rest = [r for r in rows if r not in chosen]
        rng.shuffle(rest)
        chosen += rest[:max(0,n-len(chosen))]
        result[cls] = chosen[:n]
    return result

def render(r, dataset, out, idx):
    split, imeta, ann, cat = r
    src = dataset/"splits"/split/"images"/imeta["file_name"]
    im = Image.open(src).convert("RGB")
    scale = min(1, 1400/im.width, 1000/im.height)
    if scale < 1:
        im = im.resize((round(im.width*scale),round(im.height*scale)), Image.Resampling.LANCZOS)
    d = ImageDraw.Draw(im)
    sx,sy = im.width/imeta["width"], im.height/imeta["height"]
    bx,by,bw,bh = ann["bbox"]
    d.rectangle((bx*sx,by*sy,(bx+bw)*sx,(by+bh)*sy), outline="yellow", width=4)
    pts = [ann["keypoints"][i:i+3] for i in range(0,len(ann["keypoints"]),3)]
    for a,b in cat.get("skeleton",[]):
        x1,y1,_ = pts[a-1]; x2,y2,_ = pts[b-1]
        d.line((x1*sx,y1*sy,x2*sx,y2*sy), fill="cyan", width=4)
    f = font(20, True)
    for i,(x,y,v) in enumerate(pts):
        px,py=x*sx,y*sy
        rad=8
        if v==2:
            d.ellipse((px-rad,py-rad,px+rad,py+rad), fill="red", outline="white", width=2)
        elif v==1:
            d.ellipse((px-rad,py-rad,px+rad,py+rad), outline="red", width=4)
        else:
            d.rectangle((px-rad,py-rad,px+rad,py+rad), outline="red", width=3)
        d.text((px+12,py-12),f"K{i+1}",fill="black",font=f,
               stroke_width=3,stroke_fill="white")
    title = font(26, True)
    d.rectangle((0,0,im.width,45),fill="white")
    d.text((8,7),f"{cat['name']} | {split} | image={imeta['id']} | ann={ann['id']}",
           fill="black",font=title)
    im.save(out, quality=95)

def contact(paths, out, title):
    thumbs=[]
    for p in paths:
        im=Image.open(p).convert("RGB")
        s=min(500/im.width, 500/im.height)
        thumbs.append(im.resize((round(im.width*s),round(im.height*s)),Image.Resampling.LANCZOS))
    cols=2; rows=math.ceil(len(thumbs)/cols); cell=520
    sheet=Image.new("RGB",(cols*cell,60+rows*cell),"white")
    d=ImageDraw.Draw(sheet); d.text((10,10),title,fill="black",font=font(26,True))
    for i,im in enumerate(thumbs):
        sheet.paste(im,(i%cols*cell,60+i//cols*cell))
    sheet.save(out,quality=95)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--samples-per-class",type=int,default=8)
    p.add_argument("--seed",type=int,default=20260809)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    records=load_records(a.dataset)
    selected=choose(records,a.samples_per_class,a.seed)
    rows=[]
    for cls in CLASSES:
        cdir=a.output/cls.replace(" ","_"); cdir.mkdir()
        paths=[]
        for i,r in enumerate(selected.get(cls,[]),1):
            sp,im,ann,cat=r
            fn=f"{i:02d}_{sp}_image_{im['id']}_ann_{ann['id']}.jpg"
            op=cdir/fn
            render(r,a.dataset,op,i); paths.append(op)
            kp=ann.get("keypoints",[])
            for j in range(0,len(kp),3):
                rows.append({
                    "class_id":cat["id"],"class_name":cls,"split":sp,
                    "image_id":im["id"],"annotation_id":ann["id"],
                    "file_name":im["file_name"],"keypoint_index":j//3+1,
                    "keypoint_name":f"K{j//3+1}",
                    "source_keypoint_name":cat.get("keypoints",[])[j//3],
                    "x":kp[j],"y":kp[j+1],"visibility":kp[j+2],
                    "status":status(im,ann,kp[j],kp[j+1],kp[j+2])
                })
        contact(paths,cdir/"CONTACT_SHEET.jpg",f"{cls} - semantic keypoint inspection")
    with (a.output/"selected_keypoints.csv").open("w",newline="",encoding="utf-8") as f:
        fields=["class_id","class_name","split","image_id","annotation_id","file_name",
                "keypoint_index","keypoint_name","source_keypoint_name","x","y","visibility","status"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    manifest={cls:[{"split":r[0],"image_id":r[1]["id"],"annotation_id":r[2]["id"],"file_name":r[1]["file_name"]} for r in selected.get(cls,[])] for cls in CLASSES}
    (a.output/"inspection_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    (a.output/"INSTRUCTIONS.md").write_text("""# Semantic Keypoint Inspection

Open each class `CONTACT_SHEET.jpg` and inspect the labeled K1, K2, ... points.

For each class, identify the physical/geometric meaning of every keypoint.

Reply in this form:

Channel:
K1 = ...
K2 = ...
K3 = ...
K4 = ...

T-beam:
K1 = ...
K2 = ...
K3 = ...
K4 = ...

HI-Beam:
K1 = ...
K2 = ...
K3 = ...
K4 = ...
K5 = ...
K6 = ...

Angle Bar:
K1 = ...
K2 = ...
K3 = ...

Sheet Pile:
K1 = ...
K2 = ...
K3 = ...
K4 = ...

If uncertain, write `UNCERTAIN:` rather than guessing.

Do not edit the dataset. This inspection is only to establish semantic
landmark definitions before model-specific conversion.
""",encoding="utf-8")
    print("COMPLETE")
    print("Output:",a.output)

if __name__=="__main__":
    main()
