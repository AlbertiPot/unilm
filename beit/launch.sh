# ## pre-train on ImageNet-1K
# Set the path to save checkpoints
OUTPUT_DIR=./save
# Download and extract ImageNet-1k
DATA_PATH=./dataset/in1k/train
# Download the tokenizer weight from OpenAI's DALL-E
TOKENIZER_PATH=./dall_e_tokenizer_weight
# mkdir -p $TOKENIZER_PATH
# wget -o $TOKENIZER_PATH/encoder.pkl https://cdn.openai.com/dall-e/encoder.pkl
# wget -o $TOKENIZER_PATH/decoder.pkl https://cdn.openai.com/dall-e/decoder.pkl

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 run_beit_pretraining.py \
        --data_path ${DATA_PATH} --output_dir ${OUTPUT_DIR} --num_mask_patches 75 \
        --model beit_base_patch16_224_8k_vocab --discrete_vae_weight_path ${TOKENIZER_PATH} \
        --batch_size 64 --lr 9.375e-5 --warmup_epochs 10 --epochs 800 \
        --clip_grad 3.0 --drop_path 0 --layer_scale_init_value 0.1 \
        --imagenet_default_mean_and_std