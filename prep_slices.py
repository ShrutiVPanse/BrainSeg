import os, json, argparse, numpy as np, nibabel as nib, cv2
from pathlib import Path

def normalize(img, p_low=1, p_high=99):
    lo, hi = np.percentile(img, [p_low, p_high])
    img = np.clip((img - lo) / max(1e-6, (hi - lo)), 0, 1)
    return (img * 255).astype(np.uint8)

ap = argparse.ArgumentParser()
ap.add_argument("--nifti", required=True)
ap.add_argument("--outdir", required=True)
ap.add_argument("--min_nonzero_frac", type=float, default=0.02)
args = ap.parse_args()

Path(args.outdir).mkdir(parents=True, exist_ok=True)
nii = nib.load(args.nifti)
vol = np.nan_to_num(nii.get_fdata().astype(np.float32))
vol = normalize(vol)

keep = [z for z in range(vol.shape[2]) if (vol[:,:,z] > 0).mean() >= args.min_nonzero_frac]
meta = {"nifti": args.nifti, "shape": vol.shape, "keep_slices": keep, "voxel_spacing": nii.header.get_zooms()[:3]}
with open(Path(args.outdir)/"meta.json", "w") as f: json.dump(meta, f, indent=2)

for i, z in enumerate(keep):
    cv2.imwrite(str(Path(args.outdir)/f"slice_{i:04d}.png"), vol[:,:,z])
