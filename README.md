# EZ-FastChat
This is a study note on finetunning LLM using [FastChat](https://github.com/lm-sys/FastChat).

## 1. Installation
```
conda create -n EZFastChat python=3.9
conda activate EZFastChat
pip install -r requirements.txt
pip install torch torchvision torchaudio
cd fastchat
pip install "fschat[model_worker,webui]"
```
Newer version of cuda might be required:
[cuda12.2](https://developer.nvidia.com/cuda-12-2-0-download-archive) and [cuDNN](https://developer.nvidia.com/rdp/cudnn-download). 
If needed, install the cuDNN following its [installation guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-linux)


## 2. Data
Refer to the [guide](https://github.com/puar-playground/EZ-FastChat/tree/main/data).


## 3. Finetune LLM
Run the `finetune.sh` script:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=20003 fastchat/train/train.py \
    --model_name_or_path lmsys/vicuna-7b-v1.5  \
    --data_path data/finetune/sg_val.json \
    --bf16 True \
    --output_dir experiment_result \
    --num_train_epochs 6 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True
```
We use `fastchat/train/train.py` instead of `fastchat/train/train_mem.py` to avoid an unresolved bug in saving checkpoints, caused by the [Flash-Attention](https://github.com/Dao-AILab/flash-attention). 

The script will do model parallel on 4 GPUs, thus use less number of GPUs will result in increased per-device memory cost. The above setting will results in ~45GB per-device memory cost. Each GPU will have a batch size of 2. The gradient will be accumulated for 32 steps before parameter update. checkpoints will be saved after every 50 updates. And only the most recent 5 checkpoints are retained to save disk storage. 

