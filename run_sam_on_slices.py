# run_sam_on_slices.py
# Runs Segment Anything on a folder of PNG slices and saves masks + overlays.
# Designed for grayscale MRI slices (converted from NIfTI). Avoids all-white/black masks
# by using multi-mask scoring, area gates, and simple postprocessing.

import argparse, os, json, glob
import numpy as np
import cv2
import torch
from tqdm import tqdm
from skimage.morphology import remove_small_holes, remove_small_objects
from segment_anything import sam_model_registry, SamPredictor


def auto_box(img, thr=180, min_area=100):
    """
    Crude hotspot finder for T1-post: threshold + largest component -> box.
    Falls back to point grid if nothing reasonable found.
    """
    # Otsu threshold; enforce a minimum threshold for stability on MRI
    _, t = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = max(int(t), thr)
    mask = (img >= t).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < min_area:
        return None
    x, y, w, h = cv2.boundingRect(cnt)
    return np.array([x, y, x + w, y + h])


def postprocess(mask: np.ndarray, min_obj: int = 50) -> np.ndarray:
    """Remove tiny blobs and fill tiny holes."""
    m = remove_small_objects(mask.astype(bool), min_size=min_obj)
    m = remove_small_holes(m, area_threshold=min_obj)
    return m.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices_dir", required=True, help="Folder with slice_XXXX.png images")
    ap.add_argument("--checkpoint", required=True, help="Path to SAM .pth weights")
    ap.add_argument("--model_type", default="vit_h", choices=["vit_b", "vit_l", "vit_h"])
    ap.add_argument("--outdir", required=True, help="Output folder for masks & overlays")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fg_points", type=int, default=3, help="Foreground points inside auto box")
    ap.add_argument("--min_area", type=int, default=50, help="Reject masks smaller than this #pixels")
    ap.add_argument("--max_area_frac", type=float, default=0.8, help="Reject masks larger than this frac of image")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load SAM
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=args.device)
    predictor = SamPredictor(sam)

    # Gather slices
    slices = sorted(glob.glob(os.path.join(args.slices_dir, "slice_*.png")))
    if not slices:
        raise FileNotFoundError(f"No slice_*.png found in {args.slices_dir}")

    results = []
    for sp in tqdm(slices, desc="SAM inference"):
        # SAM expects 3-channel images; convert grayscale -> RGB
        img = cv2.imread(sp, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        predictor.set_image(rgb)

        # Try a box + a few positive points; fallback to dense positive points grid.
        box = auto_box(img)
        if box is not None:
            xs = np.random.randint(box[0], box[2], size=args.fg_points)
            ys = np.random.randint(box[1], box[3], size=args.fg_points)
            points = np.stack([xs, ys], axis=1)
            labels = np.ones(len(points), dtype=np.int32)

            masks, scores, _ = predictor.predict(
                point_coords=points,
                point_labels=labels,
                box=box[None, :],
                multimask_output=True,
            )
        else:
            H, W = img.shape
            grid = 6
            xs = np.linspace(0, W - 1, grid, dtype=int)
            ys = np.linspace(0, H - 1, grid, dtype=int)
            pts = np.array([[x, y] for y in ys for x in xs])
            labels = np.ones(len(pts), dtype=np.int32)

            masks, scores, _ = predictor.predict(
                point_coords=pts,
                point_labels=labels,
                multimask_output=True,
            )

        # Choose a non-trivial mask with the best score
        best = None
        best_score = -1.0
        total = img.size
        for m, s in zip(masks, scores):
            area = int(m.sum())
            if area < args.min_area:
                continue
            if area > args.max_area_frac * total:
                continue
            if float(s) > best_score:
                best = m
                best_score = float(s)

        # If all candidates were trivial, pick largest but clamp away from "almost all"
        if best is None:
            areas = [int(m.sum()) for m in masks]
            idx = int(np.argmax(areas))
            best = masks[idx]
            if best.sum() > 0.9 * total:
                best = (best * 0).astype(bool)  # force empty to avoid all-white
            best_score = float(scores[idx])

        # Post-process
        best = postprocess(best)

        # Save mask + overlay
        base = os.path.splitext(os.path.basename(sp))[0]
        mask_path = os.path.join(args.outdir, f"{base}_mask.png")
        ov_path = os.path.join(args.outdir, f"{base}_overlay.png")

        cv2.imwrite(mask_path, (best.astype(np.uint8) * 255))

        color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        overlay = color.copy()
        overlay[best.astype(bool)] = (
            overlay[best.astype(bool)] * 0.3 + np.array([0, 255, 0]) * 0.7
        ).astype(np.uint8)
        vis = cv2.addWeighted(color, 0.7, overlay, 0.3, 0)
        cv2.imwrite(ov_path, vis)

        results.append({"slice": base, "score": best_score})

    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"Done. Wrote {len(results)} masks to {args.outdir}")


if __name__ == "__main__":
    main()
