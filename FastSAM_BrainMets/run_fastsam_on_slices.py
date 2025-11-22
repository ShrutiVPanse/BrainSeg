from pathlib import Path

import cv2
import numpy as np
from ultralytics import FastSAM
import torch

SLICES_DIR = Path("slices")
OUT_DIR = Path("fastsam_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# This will auto-download FastSAM-x weights the first time
model = FastSAM("FastSAM-x.pt")


def colorize_mask(mask: np.ndarray, color=(0, 255, 0)):
    h, w = mask.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[mask > 0] = color
    return overlay


def run_on_slice(slice_path: Path, out_subdir: Path):
    img = cv2.imread(str(slice_path))
    if img is None:
        print(f"Failed to read {slice_path}")
        return

    results = model.predict(
        source=img,
        device=device,
        imgsz=512,
        retina_masks=True,
        verbose=False,
    )

    if not results or results[0].masks is None:
        print(f"No masks for {slice_path}")
        return

    masks = results[0].masks.data.cpu().numpy()  # (N, H, W)

    combined = np.zeros_like(img)
    for m in masks:
        m_bin = (m > 0.5).astype(np.uint8)
        color = np.random.randint(0, 255, size=3, dtype=np.uint8)
        overlay = colorize_mask(m_bin, color=tuple(int(c) for c in color))
        combined = np.where(overlay > 0, overlay, combined)

    blended = cv2.addWeighted(img, 0.6, combined, 0.4, 0)
    out_path = out_subdir / slice_path.name
    cv2.imwrite(str(out_path), blended)


def main():
    for vol_dir in sorted(SLICES_DIR.iterdir()):
        if not vol_dir.is_dir():
            continue
        print("Running FastSAM on volume:", vol_dir.name)
        out_subdir = OUT_DIR / vol_dir.name
        out_subdir.mkdir(parents=True, exist_ok=True)

        for slice_path in sorted(vol_dir.glob("*.png")):
            run_on_slice(slice_path, out_subdir)


if __name__ == "__main__":
    main()