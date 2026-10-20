# Model Weights and External Dependencies

Model weights are excluded from Git to avoid oversized commits. This document distinguishes pretrained initialization weights from checkpoints produced by project experiments.

## 1. YOLO26-Pose

The experiment metadata references these pretrained model names:

- `yolo26n-pose.pt` — YOLO26n-Pose baseline and corrected experiments
- `yolo26s-pose.pt` — YOLO26s-Pose baseline and augmentation experiments
- `yolo26m-pose.pt` — YOLO26m-Pose augmentation experiment

The training script uses Ultralytics and supports automatic pretrained-weight download when the named model is available in the installed Ultralytics version. Run the relevant training command to trigger the download, or obtain the model through the official Ultralytics workflow and place it where the script expects it.

The corrected YOLO26n experiment references `checkpoints/yolo26n-pose.pt`; the training script's default model name is `yolo26n-pose.pt`. Check the experiment arguments before reproducing a run.

Pretrained model weights are initialization weights, not the trained InnoCount checkpoints.

### Trained YOLO checkpoints

Files such as the following are excluded:

- `experiments/YOLO26_Pose_Runs/**/weights/best.pt`
- `experiments/YOLO26_Pose_Runs/**/weights/last.pt`

Reproduce these by running the corresponding experiment configuration and training command. Use the saved metrics and experiment metadata to identify the run being reproduced.

The local file `checkpoints/yolo26n.pt` was not referenced by the searched current scripts or configuration files. It is not documented as a required dependency; verify its purpose before relying on it.

## 2. HRNet-W32

### Pretrained initialization weights

`scripts/train/train_hrnet.py` downloads the pretrained HRNet-W32 checkpoint from this OpenMMLab URL:

https://download.openmmlab.com/mmpose/pretrain_models/hrnet_w32-36af842e.pth

The script saves it as:

`models/pytorch/pose_coco/pose_hrnet_w32_256x192.pth`

The exact destination is inside the HRNet training repository used by the script. Run the training script to download the weights if they are missing or smaller than the script's expected size.

### Trained HRNet checkpoints

The following generated artifacts are excluded from Git:

- `model_best.pth`
- `checkpoint.pth`
- `final_state.pth`

These belong to trained experiments, not the pretrained initialization model. Reproduce them using the corresponding HRNet experiment configuration and training script.

Keep common evaluation results and experiment metadata separately from checkpoint files.

## 3. ViTPose-B

### MAE ViT-Base backbone initialization

The InnoCount ViTPose configuration expects the pretrained backbone at:

`/home/owner/.cache/torch/hub/checkpoints/mae_pretrain_vit_base_20230913.pth`

This is a machine-specific local cache path. The exact public download URL has not yet been verified. Obtain the matching MAE ViT-Base checkpoint from the official source referenced by the original ViTPose/MMPreTrain setup. Verify architecture and checkpoint compatibility before using it.

Update the config or place the file at the expected path on a new machine.

### Trained ViTPose checkpoints

The following checkpoints are excluded from Git:

- `experiments/ViTPose_Runs/E01_ViTPoseB_Smoke/epoch_1.pth`
- `experiments/ViTPose_Runs/E02_ViTPoseB_Full/epoch_90.pth`
- `experiments/ViTPose_Runs/E02_ViTPoseB_Full/epoch_100.pth`

These are outputs of project training, not public pretrained weights. Reproduce them by running the ViTPose-B configuration with the required dataset, pretrained backbone, and compatible environment.

The final test evaluation currently uses `epoch_100.pth`. Epoch 90 and epoch 100 are retained locally, but neither is stored in Git.

### MMPose source

The tested MMPose version is v1.3.2. Restore the source with:

```bash
mkdir -p third_party
git clone --branch v1.3.2 --depth 1 \
  https://github.com/open-mmlab/mmpose.git third_party/mmpose
```

The tested environment used Python 3.10, PyTorch 2.0.1+cu118, MMCV 2.1.0, MMEngine 0.10.7, MMPose 1.3.2, and MMPreTrain 1.2.0.

## 4. Evaluation and reproducibility

Keep compact experiment metadata, configurations, logs where useful, and JSON/CSV/TXT evaluation reports under version control.

Do not commit model checkpoints, datasets, large prediction arrays, or per-instance prediction audit files.

The ViTPose-B final test evaluation used the frozen test split at `datasets/ViTPose_COCO/annotations/test.json`, with ground-truth bounding boxes. The dataset is not included in Git and must be restored separately.

For comparisons across model families, record the dataset version, checkpoint identity, evaluation protocol, and whether bounding boxes are ground truth or detector predictions.