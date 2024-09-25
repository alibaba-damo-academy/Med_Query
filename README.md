## Med_Query
Med_Query is an effective and efficient framework for medical image analysis. This repository is the  
official implementation of our paper:   
- "Med-Query: Steerable Parsing of 9-DoF Medical Anatomies with Query Embeding"

<img src=figures/poc.png width=80% />

## Installation
- Python = 3.9 required
```bash
git clone https://github.com/alibaba-damo-academy/Med_Query.git
cd Med_Query
sh install.sh
```

## Quick Start
- Download RibInst masks and models from [Google Drive](https://drive.google.com/file/d/1vEzyjaUpXu9VPxPG8QXmLlGsI82ZA92P/view?usp=drive_link).
- Testing the rib parsing pipeline using your own data (NIFTI format required).
```
$ med_query_test --snapshot_det /path_to_models/det.pt --snapshot_seg /path_to_models/seg.pt --snapshot_roi /path_to_models/roi.pt -i /your_data_directory/ -o /output_directory/
```

## Data Preparation For Training From Scratch
We elaborate on the instructions based on the rib parsing task, other experiments can follow  
the similar settings.
- Set an environment variable
```
$ echo "export WORK_DIR=/path_to_work_dir" >> ~/.bashrc
$ source ~/.bashrc
```
- Prepare images and masks:
    * Download images from RibFrac dataset: [Trainset P1](https://zenodo.org/record/3893508), [Trainset P2](https://zenodo.org/record/3893498), [Validset](https://zenodo.org/record/3893496), [Testset](https://zenodo.org/record/3993380).
    * Download RibInst masks as mentioned above.

- Extract raw data and labeled masks into:
    * $WORK_DIR/rib_experiment/images/
    * $WORK_DIR/rib_experiment/masks/

- Copy dataset filelist and split files from RibInst directory into:
    * $WORK_DIR/rib_experiment/filelist.csv
    * $WORK_DIR/rib_experiment/trainset.csv
    * $WORK_DIR/rib_experiment/validset.csv
    * $WORK_DIR/rib_experiment/testset.csv

- Data preprocessing offline:
```
$ python med_query/scripts/preprocess_rib.py -f $WORK_DIR/rib_experiment/filelist.csv -i $WORK_DIR/rib_experiment/images/ -m $WORK_DIR/rib_experiment/masks/ -o $WORK_DIR/rib_experiment/ --crop -n 32
```
Generated data by the preprocessing script are listed as follows:  
- Cropped data (Optional):
    * $WORK_DIR/rib_experiment/images_cropped/
    * $WORK_DIR/rib_experiment/masks_cropped/

- Resampled data (isotropic 2mm data of raw data, also including cropped data if it exists):
    * $WORK_DIR/rib_experiment/images_2mm
    * $WORK_DIR/rib_experiment/masks_2mm
    * $WORK_DIR/rib_experiment/masks_2mm_dilated (dilated 12th pair of ribs)

- Detailed information for each rib/target:
    * $WORK_DIR/rib_experiment/cases_info.csv

The file structure looks like:
```
$WORK_DIR
├── rib_experiment
│   ├── cases_info.csv
│   ├── filelist.csv
│   ├── trainset.csv
│   ├── validset.csv
│   ├── testset.csv
│   ├── images
│   │   ├── RibFrac1-image.nii.gz
│   │   ├── RibFrac2-image.nii.gz
│   │
│   ├── masks
│   │   ├── RibFrac1-ribmask_labelled.nii.gz
│   │   ├── RibFrac2-ribmask_labelled.nii.gz
│   │
│   ├── images_cropped
│   │   ├── RibFrac1-crop-1-image.nii.gz
│   │   ├── RibFrac1-crop-2-image.nii.gz
│   │
│   ├── masks_cropped
│   │   ├── RibFrac1-crop-1-ribmask_labelled.nii.gz
│   │   ├── RibFrac1-crop-2-ribmask_labelled.nii.gz
│   │
│   ├── images_2mm
│   │   ├── RibFrac1-image.nii.gz
│   │   ├── RibFrac2-image.nii.gz
│   │
│   ├── masks_2mm
│   │   ├── RibFrac1-ribmask_labelled.nii.gz
│   │   ├── RibFrac2-ribmask_labelled.nii.gz
│   │
│   ├── masks_2mm_dilated
│   │   ├── RibFrac1-ribmask_labelled.nii.gz
│   │   ├── RibFrac2-ribmask_labelled.nii.gz
│   
├── spine_experiment
└── organ_experiment
```

## Training
- Start training of detection model
```
# modify `xxx_config.py` as needed
# `tag` is a string to distinguish each trial
$ med_query_train_det -c det/configs/xxx_config.py -g 0,1,2,3,4,5,6,7 -t tag -p 12345
```

- Start training of segmentation model
```
$ med_query_train_seg -c seg/configs/xxx_config.py -g 0,1,2,3,4,5,6,7 -t tag -p 12346
```

- Start training of roi extractor
```
$ med_query_train_roi -c seg/configs/xxx_config_roi.py -g 0,1,2,3,4,5,6,7 -t tag -p 12347
```

- Stop distributed training
```
$ stop_distributed_training [-t tag]
```
or
```
$ kill $(ps aux | grep train.py | grep -v grep | awk '{print $2}')
```

## Validation
- We support online and offline validation of detection model, the offline validation command is:
```
$ med_query_valid --snapshot_det /path_to_det_model/ -d validset
```
*Validation results will be saved at $WORK_DIR/rib_experiment/results/MedDetSeg_ValRes.csv*

## Testing
- Only testing detection model given test images directory
```
$ med_query_test --snapshot_det /path_to_det_model/ -i /input_directory/ -o /output_directory/
```

- Testing the whole pipeline
```
$ med_query_test --snapshot_det /path_to_det_model/ --snapshot_seg /path_to_seg_model/ --snapshot_roi   
/path_to_roi_model/ -i /input_directory/ -o /output_directory/
```

- Query certain ribs in the whole pipeline
```
$ med_query_test --snapshot_det /path_to_det_model/ --snapshot_seg /path_to_seg_model/ --snapshot_roi  
/path_to_roi_model/ -i /input_directory/ -o /output_directory/ -q 1,3,5,7,9
```

## 3D Detection Visualization
```
med_query_vis -m /loal_path_to_experiment/masks_2mm --info_csv /local_path_to_experiment/MedDetSeg_ValRes.csv -c {case_id}
```
Here is a video demo:

<img src=figures/3D_detection_visualization.gif width=80% />

## License
Med_Query is released under the Apache 2.0 license.

## Citation
If you find this project useful in your research, please cite the following paper:
```
@misc{guo2024med,
      title={Med-Query: Steerable Parsing of 9-DoF Medical Anatomies with Query Embedding}, 
      author={Heng Guo and Jianfeng Zhang and Ke Yan and Le Lu and Minfeng Xu},
      journal={IEEE Journal of Biomedical and Health Informatics},
      year={2024},
      publisher={IEEE}
}
```

