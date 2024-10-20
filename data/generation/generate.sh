
MODEL_DIR="/aifs4su/gov/models/baichuan-moe-hf-128k-chat-guhao"
# MODEL_DIR="/aifs4su/gov/models/Llama-2-7b-chat-hf-guhao/"
# DATASET="/aifs4su/gov/models/HK_yue_qa"
DATASET="wikitext"

# OUTPUT="/aifs4su/gov/models/baichuan-pri-hf-16B-HK_yue_qa-dataset-output"
OUTPUT="/aifs4su/gov/models/baichuan-moe-hf-128k-chat-guhao-repajama-dataset-output"

BATCH_SIZE=1
MAX_SAMPLE=5000
# MAX_SAMPLE=4

# CUDA_VISIBLE_DEVICES=1,2 \
# torchrun --nproc_per_node 2 --master_port 7831 /home/lilujun/workspace/BitDistiller/data/generation/generate.py \
#                         --base_model $MODEL_DIR \
#                         --dataset_name $DATASET \
#                         --out_path $OUTPUT \
#                         --batch_size $BATCH_SIZE \
#                         --max_sample $MAX_SAMPLE

# Single Generate
CUDA_VISIBLE_DEVICES=2,1,0 torchrun /home/lilujun/workspace/BitDistiller/data/generation/single_generate.py \
                        --base_model $MODEL_DIR \
                        --dataset_name $DATASET \
                        --out_path $OUTPUT \
                        --batch_size $BATCH_SIZE \
                        --max_sample $MAX_SAMPLE   
