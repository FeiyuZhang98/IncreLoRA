
export CUDA_VISIBLE_DEVICES=0

experiment_name=$1

python \
examples/text-classification/run_glue.py \
--experiment_name ${experiment_name} \
--model_name_or_path microsoft/deberta-v3-base \
--task_name qnli \
--apply_lora --apply_increlora --lora_type svd \
--target_rank 8  --lora_r 1   \
--reg_orth_coef 0.1 \
--init_warmup 500 --incre_interval 500 \
--top_h 22 \
--beta1 0.85 --beta2 0.85 \
--lora_module query,key,value,intermediate,layer.output,attention.output \
--lora_alpha 32 \
--lora_dropout 0. \
--do_train --do_eval --max_seq_length 512 \
--per_device_train_batch_size 32 --learning_rate 9e-4  \
--num_train_epochs 5 --warmup_steps 500 \
--cls_dropout 0.1 --weight_decay 0.01 \
--evaluation_strategy steps --eval_steps 500 \
--save_strategy steps --save_steps 16000 \
--logging_steps 300 \
--tb_writter_loginterval 300 \
--report_to tensorboard \
--seed 41 \
--root_output_dir ./output/glue/qnli \
--overwrite_output_dir