CUDA_VISIBLE_DEVICES=1 python /home/lilujun/workspace/BitDistiller/quantization/autoclip.py \
                        --model_path /aifs4su/gov/models/Qwen2-1.5B-Instruct --calib_dataset pile \
                        --quant_type int --w_bit 3 --q_group_size 128 --run_clip \
                        --batch_size 1 \
                        --dump_clip /home/lilujun/workspace/BitDistiller/quantization/clip_cache/Qwen2-1.5B-Instruct/int3-g128.pt