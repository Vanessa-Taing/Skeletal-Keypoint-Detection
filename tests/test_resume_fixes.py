"""Regression tests for the 2026-10-22 restart kit. Run: python -m pytest tests/test_resume_fixes.py -v"""
import ast
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts' / 'evaluate'))
from common_pose_metrics import calculate_all_metrics, calculate_order_invariant_pck


def make_case(category, gt_positions, pred_positions, bbox=(0, 0, 200, 100), visible=None, prediction_valid=None):
    gt = np.zeros((1, 6, 2), dtype=np.float32)
    pred = np.zeros_like(gt)
    vis = np.zeros((1, 6), dtype=np.float32)
    for j, xy in gt_positions.items():
        gt[0, j] = xy
        vis[0, j] = 1
    if visible is not None:
        vis[0, :] = 0
        for j in visible:
            vis[0, j] = 1
    for j, xy in pred_positions.items():
        pred[0, j] = xy
    valid = np.zeros((1, 6), dtype=bool)
    for j in pred_positions:
        valid[0, j] = True
    if prediction_valid is not None:
        valid = np.asarray([prediction_valid], dtype=bool)
    return (pred, gt, vis, np.asarray([bbox], dtype=np.float32),
            np.array([category]), {category: {3:'Angle Bar',1:'T-beam',0:'Channel',4:'Sheet Pile'}[category]}, valid)


def test_angle_bar_swapped_equivalent_tips_recover_100_percent():
    args = make_case(3, {0:(50,30),1:(20,60),2:(80,60)}, {0:(50,30),1:(80,60),2:(20,60)})
    result = calculate_all_metrics(*args[:6], threshold=.05, prediction_valid=args[6])
    assert result['standard_pck']['total_correct'] == 1
    assert result['order_invariant_pck']['correct'] == 3
    assert result['order_invariant_pck']['overall'] == pytest.approx(1.0)


def test_missing_prediction_is_not_free_credit():
    args = make_case(3, {0:(50,30),1:(20,60),2:(80,60)}, {0:(50,30),2:(20,60)})
    value = calculate_order_invariant_pck(*args[:6], threshold=.05, prediction_valid=args[6])
    assert value['correct'] == 2
    assert value['total'] == 3


def test_one_visible_tip_can_use_other_prediction_slot():
    args = make_case(3, {0:(50,30),1:(20,60),2:(80,60)}, {0:(50,30),2:(20,60)},visible={0,1})
    value = calculate_order_invariant_pck(*args[:6], threshold=.05, prediction_valid=args[6])
    assert value['correct'] == 2 and value['total'] == 2


def test_channel_keeps_strict_indices():
    args = make_case(0, {0:(20,20),1:(20,60),2:(80,60),3:(80,20)}, {0:(80,20),1:(20,60),2:(80,60),3:(20,20)})
    value = calculate_order_invariant_pck(*args[:6], threshold=.05, prediction_valid=args[6])
    assert value['correct'] == 2 and value['total'] == 4


def test_pixel_space_pck_on_non_square_image():
    # bbox max side 200 px => tolerance 10 px. 8px vertical error must pass.
    args = make_case(3, {0:(50,30),1:(20,60),2:(80,60)}, {0:(50,38),1:(20,60),2:(80,60)},bbox=(0,0,200,100))
    pixel = calculate_all_metrics(*args[:6],threshold=.05,prediction_valid=args[6])
    assert pixel['standard_pck']['total_correct'] == 3
    # Legacy anisotropic scaling wrongly judges the first point incorrect.
    pred, gt, vis, boxes, cat, cmap, valid = args
    pnorm, gnorm, bnorm = pred.copy(),gt.copy(),boxes.copy()
    pnorm[:,:,0]/=800; pnorm[:,:,1]/=400
    gnorm[:,:,0]/=800; gnorm[:,:,1]/=400
    bnorm[:,[0,2]]/=800; bnorm[:,[1,3]]/=400
    legacy=calculate_all_metrics(pnorm,gnorm,vis,bnorm,cat,cmap,threshold=.05,prediction_valid=valid)
    assert legacy['standard_pck']['total_correct'] == 2


def test_yolo_common_metrics_call_uses_pixel_arrays():
    file = ROOT/'scripts/evaluate/evaluate_yolo26_pose.py'
    tree=ast.parse(file.read_text())
    f=next(f for f in tree.body if isinstance(f,ast.FunctionDef) and f.name=='evaluate')
    calls=[x for x in ast.walk(f) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='calculate_all_metrics']
    assert len(calls)==1
    kwargs={kw.arg:kw.value.id for kw in calls[0].keywords if isinstance(kw.value,ast.Name)}
    assert kwargs['predictions']=='predictions_px'
    assert kwargs['gt']=='gt_px'
    assert kwargs['bboxes']=='bboxes_px'


def test_hrnet_raw_json_rows_resist_duplicate_ids(tmp_path,monkeypatch):
    # Stub dependencies: no GPU, MMPose, pycocotools or external HRNet needed.
    package = types.ModuleType('dataset')
    sub = types.ModuleType('dataset.JointsDataset')
    sub.JointsDataset = type('JointsDataset',(object,),{})
    coco_pkg = types.ModuleType('pycocotools')
    coco_sub = types.ModuleType('pycocotools.coco')
    coco_sub.COCO = type('COCO',(object,),{})
    for name,obj in [('dataset',package),('dataset.JointsDataset',sub),('pycocotools',coco_pkg),('pycocotools.coco',coco_sub)]:
        monkeypatch.setitem(sys.modules,name,obj)
    spec=importlib.util.spec_from_file_location('_hrnet_kit_loader', ROOT/'hrnet/innocount.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

    class FakeCoco:
        dataset={'annotations':[
            {'id':5,'image_id':1,'bbox':[0,0,40,40],'category_id':0,'keypoints':[5,5,2]*6},
            {'id':5,'image_id':2,'bbox':[0,0,80,80],'category_id':1,'keypoints':[9,9,2]*6}
        ]}
        def getImgIds(self): return [1,2]
        def loadImgs(self,img_id): return [{'file_name':f'{img_id}.jpg'}]
        def loadCats(self,cat_id): return [{'name':{0:'Channel',1:'T-beam'}[cat_id]}]
        def loadAnns(self, ids): raise AssertionError('Unsafe global annotation id lookup')
        def getAnnIds(self, imgIds): raise AssertionError('Unsafe ID lookup')
    loader=object.__new__(mod.InnoCountDataset)
    loader.coco=FakeCoco();loader.root=str(tmp_path);loader.image_set='train'
    rows=loader._get_db()
    assert len(rows)==2
    assert [r['image_id'] for r in rows]==[1,2]
    assert [r['category_id'] for r in rows]==[0,1]
    assert [r['bbox'][2] for r in rows]==[40,80]
    assert [r['annotation_id'] for r in rows]==[5,5]


def test_unique_id_conversion_is_read_only_and_geometry_preserving(tmp_path):
    spec=importlib.util.spec_from_file_location('_unique_kit',ROOT/'scripts/qa/prepare_unique_coco.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    source=tmp_path/'Source'
    for split in ('train','valid','test'):
        folder=source/'splits'/split
        folder.mkdir(parents=True)
        anns=[{'id':10,'image_id':1,'category_id':1,'bbox':[0,0,40,20],'keypoints':[1,2,2]*6,'num_keypoints':6},
              {'id':10,'image_id':2,'category_id':0,'bbox':[10,10,50,50],'keypoints':[3,4,2]*6,'num_keypoints':6}]
        record={'images':[{'id':1,'file_name':'a.jpg','width':640,'height':480},{'id':2,'file_name':'b.jpg','width':640,'height':480}], 'categories':[{'id':1,'name':'T-beam'},{'id':0,'name':'Channel'}], 'annotations':anns}
        (folder/'annotations.json').write_text(json.dumps(record))
    out=tmp_path/'UniqueIDs_v1'
    mod.convert(source,out,'none')
    assert not (source/'splits/train/images').exists()
    for split in ('train','valid','test'):
        old=json.loads((source/'splits'/split/'annotations.json').read_text())
        new=json.loads((out/'annotations'/f'{split}.json').read_text())
        assert [x['id'] for x in new['annotations']]==[1,2]
        for x,y in zip(old['annotations'],new['annotations']):
            assert x['bbox']==y['bbox'] and x['keypoints']==y['keypoints']
            assert y['source_annotation_id']==x['id']
    with pytest.raises(FileExistsError):
        mod.convert(source,out,'none')
    with pytest.raises(ValueError):
        mod.convert(source,source/'bad_output','none')


def test_real_test_annotations_unique_id_reindex_if_available(tmp_path):
    src = ROOT/'datasets/InnoCount_KeyPoint_ModelReady_v2'
    if not (src/'splits/test/annotations.json').is_file():
        pytest.skip('Real ModelReady v2 JSON not included in repository')
    spec=importlib.util.spec_from_file_location('_unique_kit2',ROOT/'scripts/qa/prepare_unique_coco.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    out=tmp_path/'UniqueIDs_v1';mod.convert(src,out,'none')
    train=json.loads((out/'annotations/train.json').read_text())
    valid=json.loads((out/'annotations/valid.json').read_text())
    test=json.loads((out/'annotations/test.json').read_text())
    assert [len(p['annotations']) for p in (train,valid,test)] == [12200,1551,1626]
    assert len({ann['id'] for ann in train['annotations']})==12200
    assert len({ann['id'] for ann in test['annotations']})==1626
    assert len({ann['id'] for ann in valid['annotations']})==1551


def test_hrnet_evaluator_rejects_mismatched_old_arrays():
    spec=importlib.util.spec_from_file_location('_hrnet_eval_kit',ROOT/'scripts/evaluate/evaluate_hrnet.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    coco={'annotations':[{'id':2,'image_id':7,'category_id':1,'bbox':[0,0,40,30],'keypoints':[3,4,2]*6}]}
    gt=np.asarray([[[3,4]]*6],dtype=np.float32)
    vis=np.ones((1,6),dtype=np.float32)
    bbox=np.array([[0,0,40,30]],dtype=np.float32)
    mod.assert_hrnet_gt_alignment(coco,gt,vis,bbox,[7],[1],[2])
    with pytest.raises(ValueError,match='GT mismatch'):
        mod.assert_hrnet_gt_alignment(coco,gt,vis,bbox,[7],[0],[2])
    with pytest.raises(ValueError,match='GT mismatch'):
        mod.assert_hrnet_gt_alignment(coco,gt+10,vis,bbox,[7],[1],[2])
    with pytest.raises(ValueError,match='unknown annotation'):
        mod.assert_hrnet_gt_alignment(coco,gt,vis,bbox,[8],[1],[2])

def test_audit_identity_fails_on_original_duplicates_and_passes_on_derived(tmp_path):
    import importlib.util
    spec=importlib.util.spec_from_file_location('_audit_kit',ROOT/'scripts/qa/audit_coco_identity.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    src=ROOT/'datasets/InnoCount_KeyPoint_ModelReady_v2'
    if not (src/'splits/test/annotations.json').is_file():
        pytest.skip('Real annotated dataset not available')
    # Build a small original-indexed COCO folder with genuinely duplicated IDs.
    rawdir=tmp_path/'original'/'annotations';rawdir.mkdir(parents=True)
    for split in ('train','valid','test'):
        (rawdir/f'{split}.json').write_bytes((src/'splits'/split/'annotations.json').read_bytes())
    with pytest.raises(SystemExit):
        mod.audit(rawdir.parent)
    uniquespec=importlib.util.spec_from_file_location('_unique3',ROOT/'scripts/qa/prepare_unique_coco.py')
    unique=importlib.util.module_from_spec(uniquespec);uniquespec.loader.exec_module(unique)
    out=tmp_path/'derived';unique.convert(src,out,'none')
    mod.audit(out)


def test_vitpose_unique_id_config_points_to_versioned_dataset():
    cfg=(ROOT/'configs/innocount/td-hm_ViTPose-base_innocount_unique_ids.py').read_text()
    assert "data_root = 'datasets/InnoCount_COCO_UniqueIDs_v1/'" in cfg

def test_hrnet_train_guard_preserves_existing_runtime_folder_and_experiments():
    cfg=(ROOT/'scripts/train/train_hrnet.py').read_text()
    assert 'Existing HRNet runtime dataset' in cfg
    assert 'Experiment directory already contains files' in cfg
    assert 'shutil.rmtree(data_dst)' not in cfg
    assert 'init_original.rstrip()' in cfg
