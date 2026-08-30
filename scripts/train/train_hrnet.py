import os
from pathlib import Path
import argparse
import shutil
import subprocess
import sys
import importlib.util
import yaml
from datetime import datetime

'''
python scripts/train/train_hrnet.py \
    --name E07_HRNet_W32 \
    --batch 2 \
    --imgsz 640 \
    --hrnet-repo /home/owner/HRNet-Human-Pose-Estimation
'''

# ------------------------------------------------------------------
# Install missing dependencies
# ------------------------------------------------------------------

REQUIRED = [
    "tensorboard",
    "tensorboardX",
    "yacs",
    "json_tricks",
    "pycocotools",
]

missing = [
    pkg for pkg in REQUIRED
    if importlib.util.find_spec(pkg) is None
]

if missing:
    print("Installing missing HRNet dependencies...")
    subprocess.run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "tensorboard",
        "tensorboardX",
        "yacs",
        "easydict",
        "json_tricks",
        "pycocotools",
        "scipy",
        "scikit-image",
        "opencv-python==4.10.0.84",
        "pyyaml",
    ], check=True)

ROOT = Path(__file__).resolve().parents[2]

def patch_train_scheduler(hrnet_repo: Path):
    """
    Fix PyTorch warning by moving lr_scheduler.step()
    after train() inside HRNet's tools/train.py.
    """
    train_py = hrnet_repo / "tools" / "train.py"

    text = train_py.read_text(encoding="utf-8")

    old = """    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        lr_scheduler.step()

        train("""
    new = """    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):

        train("""

    if old in text:
        text = text.replace(old, new, 1)

        marker = """        train(
            cfg, train_loader, model, criterion, optimizer, epoch,
            final_output_dir, tb_log_dir, writer_dict
        )
"""
        replacement = marker + "\n        lr_scheduler.step()\n"

        text = text.replace(marker, replacement, 1)

        train_py.write_text(text, encoding="utf-8")
        print("✓ Patched HRNet LR scheduler order")
    else:
        print("✓ HRNet LR scheduler already patched")

def train(args):

    hrnet_repo = Path(args.hrnet_repo).resolve()

    if not hrnet_repo.exists():
        raise FileNotFoundError(hrnet_repo)
    
        # --------------------------------------------------
    # Create run directory and timestamp
    # --------------------------------------------------

    save_dir = (
        ROOT / "experiments" / "HRNet_Runs" / args.name
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    log_dir = save_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    config_archive_dir = save_dir / "config"
    config_archive_dir.mkdir(parents=True, exist_ok=True)

    # Fix PyTorch scheduler warning
    patch_train_scheduler(hrnet_repo)
    # --------------------------------------------------
    # Copy dataset loader
    # --------------------------------------------------

    shutil.copy2(
        ROOT / "hrnet" / "innocount.py",
        hrnet_repo / "lib" / "dataset" / "innocount.py",
    )

    # Patch dataset registry
    init_file = hrnet_repo / "lib" / "dataset" / "__init__.py"

    init_file.write_text(
        """from .JointsDataset import JointsDataset
from .innocount import InnoCountDataset

innocount = InnoCountDataset
""",
        encoding="utf-8",
    )

    # --------------------------------------------------
    # Patch YAML dynamically
    # --------------------------------------------------

    exp_dir = hrnet_repo / "experiments" / "innocount"
    exp_dir.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(
        open(ROOT / "hrnet" / "hrnet_w32_640.yaml")
    )

    cfg["TRAIN"]["BATCH_SIZE_PER_GPU"] = args.batch
    cfg["TEST"]["BATCH_SIZE_PER_GPU"] = args.batch

    cfg["MODEL"]["IMAGE_SIZE"] = [args.imgsz, args.imgsz]
    cfg["MODEL"]["HEATMAP_SIZE"] = [
        args.imgsz // 4,
        args.imgsz // 4,
    ]

    cfg_dst = exp_dir / "hrnet_w32_640.yaml"

    with open(cfg_dst, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    config_archive_dir = save_dir / "config"
    config_archive_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(
        cfg_dst,
        config_archive_dir / f"{timestamp}_hrnet_w32_640.yaml"
    )

    # --------------------------------------------------
    # Copy dataset
    # --------------------------------------------------

    data_dst = hrnet_repo / "data" / "InnoCount"

    if data_dst.exists():
        shutil.rmtree(data_dst)

    shutil.copytree(
        ROOT / "datasets" / "HRNet_COCO",
        data_dst,
    )

    # --------------------------------------------------
    # Pretrained weights
    # --------------------------------------------------

    weights_dir = (
        hrnet_repo / "models" / "pytorch" / "pose_coco"
    )
    weights_dir.mkdir(parents=True, exist_ok=True)

    pretrained = (
        weights_dir / "pose_hrnet_w32_256x192.pth"
    )

    if (
        not pretrained.exists()
        or pretrained.stat().st_size < 100_000_000
    ):

        if pretrained.exists():
            pretrained.unlink()

        print("Downloading pretrained HRNet weights...")

        subprocess.run([
            "wget",
            "-O",
            str(pretrained),
            "https://download.openmmlab.com/mmpose/pretrain_models/hrnet_w32-36af842e.pth",
        ], check=True)

    # --------------------------------------------------
    # Launch training
    # --------------------------------------------------

    # Create run directory BEFORE training starts so the log
    # survives even if training is interrupted.
    log_file = log_dir / f"{timestamp}.log"

    print("=" * 60)
    print("Launching HRNet training")
    print("=" * 60)
    print(f"Run name : {args.name}")
    print(f"Log file : {log_file}")
    print("=" * 60)

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    cmd = [
        sys.executable,
        "-u",
        "tools/train.py",
        "--cfg",
        "experiments/innocount/hrnet_w32_640.yaml",
    ]

    # Record run information immediately.
    with open(log_file, "w", encoding="utf-8") as log:
        log.write("=" * 80 + "\n")
        log.write("HRNet TRAINING RUN\n")
        log.write("=" * 80 + "\n")
        log.write(f"Run name: {args.name}\n")
        log.write(f"Start time: {datetime.now().isoformat()}\n")
        log.write(f"Repository: {hrnet_repo}\n")
        log.write(f"Config: {hrnet_repo / 'experiments/innocount/hrnet_w32_640.yaml'}\n")
        log.write(f"Command: {' '.join(cmd)}\n")
        log.write(f"Batch size: {args.batch}\n")
        log.write(f"Image size: {args.imgsz}\n")
        log.write("=" * 80 + "\n\n")
        log.flush()

        try:
            process = subprocess.Popen(
                cmd,
                cwd=hrnet_repo,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Write output continuously.
            for line in process.stdout:
                timestamp_line = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                formatted = (
                    f"[{timestamp_line}] {line}"
                )

                print(formatted, end="")
                log.write(formatted)
                log.flush()

            return_code = process.wait()

            log.write("\n")
            log.write("=" * 80 + "\n")
            log.write(
                f"End time: {datetime.now().isoformat()}\n"
            )
            log.write(
                f"Process return code: {return_code}\n"
            )

            if return_code == 0:
                log.write("STATUS: COMPLETED SUCCESSFULLY\n")
            else:
                log.write("STATUS: TRAINING FAILED / INTERRUPTED\n")

            log.write("=" * 80 + "\n")
            log.flush()

            if return_code != 0:
                raise subprocess.CalledProcessError(
                    return_code,
                    cmd,
                )

        except KeyboardInterrupt:
            # Ctrl+C in the parent process
            log.write("\n")
            log.write("=" * 80 + "\n")
            log.write(
                f"Interrupted at: {datetime.now().isoformat()}\n"
            )
            log.write("STATUS: INTERRUPTED BY USER\n")
            log.write("=" * 80 + "\n")
            log.flush()

            process.terminate()
            process.wait()

            raise

    # --------------------------------------------------
    # Copy outputs
    # --------------------------------------------------

    output = hrnet_repo / "output"

    if output.exists():
        for item in output.iterdir():
            dst = save_dir / item.name
            if item.is_dir():
                shutil.copytree(
                    item,
                    dst,
                    dirs_exist_ok=True
                )
            else:
                shutil.copy2(item, dst)

    print("\nTraining complete.")
    print(f"Run directory: {save_dir}")
    print(f"Log: {log_file}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hrnet-repo",
        default=str(ROOT.parent / "HRNet-Human-Pose-Estimation"),
    )

    parser.add_argument(
        "--name",
        default="E06_HRNet_W32",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=2,
        help="Batch size per GPU",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size",
    )

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()