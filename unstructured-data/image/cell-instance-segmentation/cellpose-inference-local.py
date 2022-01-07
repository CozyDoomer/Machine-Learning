from pathlib import Path

import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from cellpose import io, models, plot


def rle_encode(img):
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


model_dict = {
    "experiment_cyto_diameter20": {
        "path": "models/experiment_cyto_diameter20/cellpose_residual_on_style_on_concatenation_off_train_converted_split_2021_12_24_07_14_36.376481_epoch_999",
        "diameter": 20,
        "omni": False,
    },
    "experiment_cyto_diameter0": {
        "path": "models/experiment_cyto_diameter0/cellpose_residual_on_style_on_concatenation_off_train_converted_split_2021_12_24_11_31_46.912635_epoch_999",
        "diameter": 0,
        "omni": False
    },
    "experiment_omni_diameter20": {
        "path": "models/experiment_omni_diameter20/cellpose_residual_on_style_on_concatenation_off_omni_train_converted_split_2021_12_25_02_50_03.864047_epoch_999",
        "diameter": 20,
        "omni": True
    },
    "experiment_omni_diameter0": {
        "path": "models/experiment_omni_diameter0/cellpose_residual_on_style_on_concatenation_off_omni_train_converted_split_2021_12_26_03_51_44.296669_epoch_999",
        "diameter": 0,
        "omni": True
    },
    "experiment_nuclei_diameter0": {
        "path": "models/experiment_nuclei_diameter0/cellpose_residual_on_style_on_concatenation_off_train_converted_split_2021_12_25_12_13_05.237773_epoch_999",
        "diameter": 0,
        "omni": False
    },
    "experiment_nuclei_diameter20": {
        "path": "models/experiment_nuclei_diameter20/cellpose_residual_on_style_on_concatenation_off_train_converted_split_2021_12_24_17_54_16.110001_epoch_999",
        "diameter": 20,
        "omni": False
    },
    "experiment_nuclei_diameter0_longer": {
        "path": "models/experiment_nuclei_diameter0_longer/cellpose_residual_on_style_on_concatenation_off_train_converted_split_2021_12_26_13_13_08.568327_epoch_1999",
        "diameter": 0,
        "omni": False
    },
    "experiment_nuclei_diameter16": {
        "path": "models/experiment_nuclei_diameter16/cellpose_residual_on_style_on_concatenation_off_train_converted_split_2021_12_27_08_52_55.231418_epoch_1999",
        "diameter": 16,
        "omni": False
    },
    "experiment_nuclei_diameter24": {
        "path": "models/experiment_nuclei_diameter24/cellpose_residual_on_style_on_concatenation_off_train_converted_split_2021_12_26_22_34_44.922115_epoch_1999",
        "diameter": 24,
        "omni": False
    },
}
# test_dir = Path("dataset/test")
test_dir = Path("dataset/validation_split")
test_files = [fname for fname in test_dir.iterdir()]

for exp_name, exp_dict in model_dict.items():
    print(f"predicting {exp_name}")

    omni = exp_dict["omni"]
    weights_path = exp_dict["path"]
    diameter = exp_dict["diameter"]

    output_folder = os.path.join(
        "eval",
        exp_name,
        weights_path.split("/")[-1]
    )
    if os.path.exists(output_folder):
        print(f"skipping {output_folder}")
        continue

    os.makedirs(output_folder, exist_ok=True)

    model = models.CellposeModel(
        gpu=True,
        omni=omni,
        pretrained_model=weights_path
    )

    ids, masks = [], []
    for fn in tqdm(test_files):
        preds, flows, _ = model.eval(
            io.imread(str(fn)),
            omni=omni,
            diameter=diameter,
            channels=[0, 0],
            augment=True,
            resample=True,
        )
        for i in range(1, preds.max() + 1):
            ids.append(fn.stem)
            masks.append(rle_encode(preds == i))

    pd.DataFrame({"id": ids, "predicted": masks}).to_csv(os.path.join(output_folder, "submission.csv"), index=False)
