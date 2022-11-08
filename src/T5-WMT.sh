#!/bin/bash
trans_models=( t5-small )
for m in "${trans_models[@]}"
do
  python huggingface/transformers/examples/pytorch/translation/run_translation.py \
    --model_name_or_path local-models/"${m}"-"$1" \
    --tokenizer_name local-models/"${m}"-tokenizer \
    --do_eval \
    --source_lang en \
    --target_lang ro \
    --source_prefix "translate English to Romanian: " \
    --dataset_name wmt16 \
    --dataset_config_name ro-en \
    --output_dir results/translation/"${m}"-"$1" \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
done