# Copyright (c) DAMO Health

import glob
import os
import SimpleITK as sitk
from med_query.utils.itk import normalize_sitk_im
from torch.utils.data import Dataset


class FLAREDatasetTest(Dataset):
    def __init__(self, cfg, image_dir: str = None) -> None:
        super(FLAREDatasetTest, self).__init__()

        assert os.path.exists(image_dir), f"No such directory '{image_dir}'"
        self.image_dir = image_dir

        self.image_list = glob.glob(os.path.join(self.image_dir, "*.nii.gz"))

        self.normalization_params = cfg.normalization_params

        self.min_value = None
        self.max_value = None
        self.clip = True
        if self.normalization_params is not None:
            self.min_value = self.normalization_params.min_value
            self.max_value = self.normalization_params.max_value
            self.clip = self.normalization_params.clip

    def __len__(self) -> int:
        return len(self.image_list)

    def __repr__(self):
        return "testset for flare_experimentation"

    def __getitem__(self, index):
        image_path = self.image_list[index]
        pid = os.path.basename(image_path)
        if image_path.endswith("-image.nii.gz"):
            pid = os.path.basename(image_path).replace("-image.nii.gz", "")
        elif image_path.endswith("_0000.nii.gz"):
            pid = os.path.basename(image_path).replace("_0000.nii.gz", "")
        elif image_path.endswith(".nii.gz"):
            pid = os.path.basename(image_path).replace(".nii.gz", "")

        image = sitk.ReadImage(image_path, sitk.sitkInt16)
        image = sitk.Cast(image, sitk.sitkFloat32)
        image = normalize_sitk_im(image, self.min_value, self.max_value, self.clip, True)

        meta_data = {"pid": pid, "img_res0": image}

        data = {"meta_data": meta_data}

        return data

    def collate_fn(self, batch):
        return {"meta_data": [item["meta_data"] for item in batch]}
