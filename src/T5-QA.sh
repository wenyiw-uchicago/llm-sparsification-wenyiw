#!/bin/bash

qa_models=( t5-small )
for m in "${qa_models[@]}"
do
python huggingface/transformers/examples/pytorch/question-answering/run_seq2seq_qa.py \
  --model_name_or_path local-models/"${m}"-"$1" \
  --tokenizer_name local-models/"${m}"-tokenizer \
  --dataset_name squad_v2 \
  --context_column context \
  --question_column question \
  --answer_column answers \
  --do_eval \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir results/QA/"${m}"-"$1"
done
