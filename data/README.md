# Data preprocess

## 1. COCO-stuff
Download required files `captions_train2017.json`, `captions_val2017.json`, `instances_train2017.json`, `instances_val2017.json`:
```
pip install gdown
gdown "https://drive.google.com/uc?id=17s3VEarlsA6LXZ8I-UJIQ4aQUN1Rr8JD"
```

### Alternatively, download from source
```
chmod +x download_coco.sh
./download_coco.sh
```
standard dataset splits: 25,000 for training, 1,024 for validation, and 2,048 for test.
Only the `captions_train2017.json`, `captions_val2017.json`, `instances_train2017.json`, `instances_val2017.json` files are required to attract the object scene-graph.

## 2. Visual Genome
Simply unzip the `vg_lite.zip` file
```
unzip vg_lite.zip
```

### Alternatively, download from source
```
chmod +x download_vg.sh
./download_vg.sh
```
After downloading the Visual Genome dataset, use the python script `preprocess_vg.py` to pre-process the dataset and create the standard dataset splits: 62,565 pairs for training, 5,506 for validation, and 5,088 for test.
```
python preprocess_vg.py
```

## 3. BLIP laion cc 558k
Install `huggingface-cli` following the [tutorial](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli) or below,
```
pip install -U "huggingface_hub[cli]"
```
Login using an Access [Tokens](https://huggingface.co/settings/tokens):
```
huggingface-cli login
```

Download from huggingface
```
huggingface-cli download liuhaotian/LLaVA-Pretrain --repo-type dataset --local-dir .
```
