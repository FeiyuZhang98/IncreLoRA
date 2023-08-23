export CUDA_VISIBLE_DEVICES=0

experiment_name=$1

target_rank=2
dropout=0.
top_h=1
# target_rank=4
# dropout=0.
# top_h=1
# target_rank=8
# dropout=0.1
# top_h=1


python \
examples/question-answering/run_qa.py \
--advance_learn True \
--multi_lr True \
--experiment_name ${experiment_name}\
--model_name_or_path microsoft/deberta-v3-base \
--dataset_name squad_v2 \
--apply_lora --apply_increlora \
--lora_type svd --target_rank ${target_rank} --lora_r 1 \
--reg_orth_coef 0.1 \
--init_warmup 1000 --incre_interval 1000 \
--top_h ${top_h} \
--incre_rank_num 1 \
--beta1 0.85 --beta2 0.85 \
--lora_module query,key,value,intermediate,layer.output,attention.output \
--lora_alpha 16 \
--lora_dropout ${dropout} \
--do_train --do_eval --version_2_with_negative \
--max_seq_length 384 --doc_stride 128 \
--per_device_train_batch_size 16 \
--learning_rate 1e-3 \
--num_train_epochs 14 \
--warmup_steps 1000 --per_device_eval_batch_size 128 \
--evaluation_strategy steps --eval_steps 1000 \
--save_strategy steps --save_steps 100000 \
--logging_steps 300 \
--tb_writter_loginterval 300 \
--report_to tensorboard \
--seed 9 \
--root_output_dir ./output/debertav3-base/squadv2 \
--overwrite_output_dir 

