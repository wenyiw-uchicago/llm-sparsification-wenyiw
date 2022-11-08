#!/bin/bash

s=$1
glue_models=( gpt2 microsoft/deberta-v2-xlarge )

# GLUE
for m in "${glue_models[@]}"
do
  python huggingface/transformers/examples/pytorch/text-classification/run_glue.py \
  --model_name_or_path local-models/"${m}"-"${s}"\
  --tokenizer_name local-models/"${m}"-tokenizer\
  --task_name mnli \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir results/GLUE/"${m}"-"${s}"
done