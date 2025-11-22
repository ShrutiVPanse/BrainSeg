from pathlib import Path

import nibabel as nib
import numpy as np
import cv2

# Path to your NIfTI brain mets data (already downloaded)
NIFTI_DIR = Path("../data/annotated_data/images/NIFTIs_Images")
# Where to save PNG slices for FastSAM
OUT_DIR = Path("slices")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_to_uint8(volume):
    v = volume.astype(np.float32)
    low, high = np.percentile(v, [1, 99])
    v = np.clip((v - low) / (high - low + 1e-8), 0, 1)
    v = (v * 255).astype(np.uint8)
    return v


def process_one_nifti(path: Path, max_slices: int | None = None):
    img = nib.load(str(path))
    data = img.get_fdata()  # typically (X, Y, Z)

    if data.ndim != 3:
        print(f"Skipping {path.name}, ndim={data.ndim}")
        return

    # Axial slices: (Z, H, W)
    vol = np.transpose(data, (2, 0, 1))
    vol = normalize_to_uint8(vol)

    num_slices = vol.shape[0]
    if max_slices is not None:
        indices = np.linspace(0, num_slices - 1, max_slices).astype(int)
    else:
        indices = range(num_slices)

    stem = path.stem
    out_subdir = OUT_DIR / stem
    out_subdir.mkdir(parents=True, exist_ok=True)

    for i in indices:
        slice_2d = vol[i]
        # Resize to 512x512 for FastSAM
        slice_resized = cv2.resize(slice_2d, (512, 512), interpolation=cv2.INTER_LINEAR)
        # Make 3-channel grayscale
        slice_rgb = np.stack([slice_resized] * 3, axis=-1)
        out_path = out_subdir / f"slice_{i:03d}.png"
        cv2.imwrite(str(out_path), slice_rgb)


def main():
    nifti_files = sorted(NIFTI_DIR.glob("*.nii*"))
    print(f"Found {len(nifti_files)} NIfTI files")
    # For first run, just try a few volumes
    for path in nifti_files[:3]:
        print(f"Processing {path.name}")
        process_one_nifti(path, max_slices=40)  # 40 slices per volume for now


if __name__ == "__main__":
    import numpy as np
    main()