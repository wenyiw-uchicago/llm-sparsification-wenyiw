#!/bin/bash

mlm_models=( microsoft/deberta-v2-xlarge )
for m in "${mlm_models[@]}"
do
  python huggingface/transformers/examples/pytorch/language-modeling/run_mlm.py \
    --model_name_or_path local-models/"${m}"-"$1" \
    --tokenizer_name local-models/"${m}-tokenizer"\
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_eval \
    --output_dir results/mlm/"${m}"-"$1"
done