#!/bin/bash
# Run SFT for Qwen3-1.7B using config file

CONFIG=sft_config.yaml

# Parse YAML config (requires yq: https://github.com/mikefarah/yq)
if ! command -v yq &> /dev/null; then
  echo "yq is required to parse YAML config. Install with: brew install yq or pip install yq."
  exit 1
fi

MODEL_NAME=$(yq '.model_name' $CONFIG)
DATA_PATH=$(yq '.data_path' $CONFIG)
OUTPUT_DIR=$(yq '.output_dir' $CONFIG)
BATCH_SIZE=$(yq '.batch_size' $CONFIG)
EPOCHS=$(yq '.epochs' $CONFIG)
LR=$(yq '.lr' $CONFIG)
MAX_LENGTH=$(yq '.max_length' $CONFIG)
USE_LORA=$(yq '.use_lora' $CONFIG)
LORA_R=$(yq '.lora_r' $CONFIG)
LORA_ALPHA=$(yq '.lora_alpha' $CONFIG)
LORA_DROPOUT=$(yq '.lora_dropout' $CONFIG)
DEVICE=$(yq '.device' $CONFIG)

CMD="python training/sft_qwen.py --model_name $MODEL_NAME --data_path $DATA_PATH --output_dir $OUTPUT_DIR --batch_size $BATCH_SIZE --epochs $EPOCHS --lr $LR --max_length $MAX_LENGTH --device $DEVICE"
if [ "$USE_LORA" = "true" ]; then
  CMD="$CMD --use_lora --lora_r $LORA_R --lora_alpha $LORA_ALPHA --lora_dropout $LORA_DROPOUT"
fi

echo "Running: $CMD"
eval $CMD
