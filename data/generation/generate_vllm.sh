CUDA_VISIBLE_DEVICES=0,1,2,3 python generate_vllm.py --base_model /aifs4su/gov/models/baichuan-moe-hf-128k-chat-guhao --dataset_name wikitext \
                        --out_path /aifs4su/gov/models/baichuan-moe-hf-128k-chat-guhao-alpaca-dataset-output \
                        --max_sample 5000


# python3 -m vllm.entrypoints.openai.api_server --model /aifs4su/yaodong/baichuan_model_moe_9_10 --port 8000 \
#         --tensor-parallel-size 8 --served-model-name baichuan_model_moe_9_10 --trust-remote-code \
#         --chat-template /aifs4su/zhujunqi/code/baichuan-moe/moe_baichuan_pku_chat.jinja --max-model-len 24208

# docker run -d -it --gpus all --network=host --ipc=host --privileged --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=1024g -v /aifs4su/:/aifs4su -v /etc/network:/etc/network -v /etc:/host/etc -v /lib/udev:/host/lib/udev -e PATH=/miniconda/envs/opencompass/bin:$PATH -e PYTHONPATH=/workspace  --entrypoint sleep registry.hkustllm.com/gov/opencompass-vllm:v1.0  inf