# Data preprocess

## 1. COCO-stuff
Download the dataset
```
chmod +x download_coco.sh
./download_coco.sh
```
standard dataset splits: 25,000 for training, 1,024 for validation, and 2,048 for test.


## 2. Visual Genome
Simply unzip the `vg_lite.zip` file
```
unzip vg_lite.zip
```

### Or 
Download from the Visual Genome dataset
```
chmod +x download_vg.sh
./download_vg.sh
```
After downloading the Visual Genome dataset, use the python script `preprocess_vg.py` to pre-process the dataset and create the standard dataset splits: 62,565 pairs for training, 5,506 for validation, and 5,088 for test.
```
python preprocess_vg.py
```
