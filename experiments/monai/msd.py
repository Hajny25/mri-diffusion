import os
import glob
import json
from pathlib import Path
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

import nibabel as nib
import numpy as np

# ----------------- CONFIGURE THESE PATHS -----------------
BRATS_ROOT = "data/brats-2021"      # folder containing BRATS cases
MSD_ROOT   = "data/brats-2021-msd"         # root where Task01_BrainTumour will be created
TASK_NAME  = "Task01_BrainTumour"

NUM_WORKERS = 30  # number of parallel processes (e.g., CPU cores)
# ---------------------------------------------------------

TASK_DIR   = os.path.join(MSD_ROOT, TASK_NAME)
IMAGES_TR  = os.path.join(TASK_DIR, "imagesTr")
LABELS_TR  = os.path.join(TASK_DIR, "labelsTr")

os.makedirs(IMAGES_TR, exist_ok=True)
os.makedirs(LABELS_TR, exist_ok=True)


def find_cases(brats_root: str):
    """
    Find all BRATS cases by locating *_flair.nii.gz files.
    Returns a list of dicts: {"case_id": ..., "flair": Path, "t1": Path, "t1ce": Path, "t2": Path, "seg": Path}.
    """
    flair_paths = glob.glob(os.path.join(brats_root, "**", "*_flair.nii.gz"), recursive=True)
    flair_paths.sort()

    cases = []
    for flair_path in flair_paths:
        flair_path = Path(flair_path)
        name = flair_path.name
        if "_flair.nii.gz" not in name:
            continue

        case_id = name.replace("_flair.nii.gz", "")
        case_dir = flair_path.parent

        t1_path   = case_dir / f"{case_id}_t1.nii.gz"
        t1ce_path = case_dir / f"{case_id}_t1ce.nii.gz"
        t2_path   = case_dir / f"{case_id}_t2.nii.gz"
        seg_path  = case_dir / f"{case_id}_seg.nii.gz"

        if not (t1_path.exists() and t1ce_path.exists() and t2_path.exists() and seg_path.exists()):
            print(f"[WARN] Skipping {case_id} (missing mods/label)")
            continue

        cases.append(
            {
                "case_id": case_id,
                "flair": flair_path,
                "t1": t1_path,
                "t1ce": t1ce_path,
                "t2": t2_path,
                "seg": seg_path,
            }
        )
    return cases


def convert_case(case, images_tr: str, labels_tr: str):
    """
    Convert a single BRATS case into MSD-style:
    - Load flair, t1, t1ce, t2, seg
    - Stack modalities into one 4D image [X, Y, Z, 4]
    - Save to imagesTr/<case_id>.nii.gz and labelsTr/<case_id>.nii.gz

    Returns a tuple: (case_id, relative_image_path, relative_label_path) on success,
                     or None on failure (logged).
    """
    case_id = case["case_id"]
    flair_path = case["flair"]
    t1_path    = case["t1"]
    t1ce_path  = case["t1ce"]
    t2_path    = case["t2"]
    seg_path   = case["seg"]

    try:
        # Load volumes
        flair_img = nib.load(str(flair_path))
        t1_img    = nib.load(str(t1_path))
        t1ce_img  = nib.load(str(t1ce_path))
        t2_img    = nib.load(str(t2_path))
        seg_img   = nib.load(str(seg_path))

        flair = flair_img.get_fdata(dtype=np.float32)
        t1    = t1_img.get_fdata(dtype=np.float32)
        t1ce  = t1ce_img.get_fdata(dtype=np.float32)
        t2    = t2_img.get_fdata(dtype=np.float32)
        seg   = seg_img.get_fdata(dtype=np.float32)

        # Sanity check shapes
        if not (flair.shape == t1.shape == t1ce.shape == t2.shape == seg.shape):
            print(f"[WARN] Shape mismatch in {case_id}, skipping.")
            return None

        # Stack modalities into one 4D volume [X, Y, Z, C]
        stacked = np.stack([flair, t1, t1ce, t2], axis=-1)  # [X, Y, Z, 4]

        affine = flair_img.affine

        out_image_name = f"{case_id}.nii.gz"
        out_label_name = f"{case_id}.nii.gz"

        image_out_path = os.path.join(images_tr, out_image_name)
        label_out_path = os.path.join(labels_tr, out_label_name)

        # Save combined image and label
        new_img = nib.Nifti1Image(stacked, affine)
        nib.save(new_img, image_out_path)
        nib.save(seg_img, label_out_path)

        rel_image = f"imagesTr/{out_image_name}"
        rel_label = f"labelsTr/{out_label_name}"
        return case_id, rel_image, rel_label

    except Exception as e:
        print(f"[ERROR] Failed converting {case_id}: {e}")
        return None


def main():
    cases = find_cases(BRATS_ROOT)
    print(f"Found {len(cases)} candidate cases.")

    train_entries = []

    # Partial function with fixed output dirs
    worker = partial(convert_case, images_tr=IMAGES_TR, labels_tr=LABELS_TR)

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(worker, case): case for case in cases}

        for i, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if result is not None:
                case_id, rel_image, rel_label = result
                train_entries.append({"image": rel_image, "label": rel_label})
                print(f"[{i}/{len(cases)}] Converted {case_id}")
            else:
                # error/warn already printed in worker
                pass

    print(f"Successfully converted {len(train_entries)} cases.")

    # ---- Create dataset.json ----
    dataset_json = {
        "name": TASK_NAME,
        "description": "BRATS 2021 Task 1 converted to MSD format",
        "tensorImageSize": "4D",
        "reference": "BRATS 2021",
        "licence": "CC-BY-SA 4.0",
        "release": "1.0",
        "modality": {
            "0": "FLAIR",
            "1": "T1w",
            "2": "T1gd",
            "3": "T2w"
        },
        "labels": {
            "0": "background",
            "1": "edema",
            "2": "non-enhancing tumor",
            "3": "enhancing tumor"
        },
        "numTraining": len(train_entries),
        "numTest": 0,
        "training": train_entries,
        "test": []
    }

    os.makedirs(TASK_DIR, exist_ok=True)
    dataset_json_path = os.path.join(TASK_DIR, "dataset.json")
    with open(dataset_json_path, "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"Wrote dataset.json to: {dataset_json_path}")
    print(f"MSD-style folder created at: {TASK_DIR}")


if __name__ == "__main__":
    main()