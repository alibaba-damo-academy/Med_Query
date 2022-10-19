## Med_Query
Med_Query is an effective and efficient framework for medical image analysis. The whole project is under construction.

## Overview
Currently the project is organized as follows:
- flare (competition solution for FLARE22)

## Installation
```bash
git clone https://github.com/DAMO-Health/Med_Query.git
cd Med_Query
python setup.py install
```

## License
Med_Query is released under the Apache 2.0 license.

## Usage
```
flare_test --snapshot /path_to_det/det_0.pt,/path_to_det/det_1.pt  --snapshot_seg /path_to_seg/ 
--snapshot_roi /path_to_roi/roi.pt --image_dir /path_to_test_images/ -n 1
```