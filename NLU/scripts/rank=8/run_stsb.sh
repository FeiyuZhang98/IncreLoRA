
export CUDA_VISIBLE_DEVICES=0

experiment_name=$1

seed=41
target_rank=2
top_h=4

python \
examples/text-classification/run_glue.py \
--experiment_name ${experiment_name}_rank=${target_rank} \
--model_name_or_path microsoft/deberta-v3-base \
--task_name stsb \
--apply_lora --apply_increlora --lora_type svd \
--target_rank ${target_rank}  --lora_r 1    \
--reg_orth_coef 0.3 \
--init_warmup 100 --incre_interval 100 \
--top_h ${top_h} \
--beta1 0.85 --beta2 0.85 \
--lora_module query,key,value,intermediate,layer.output,attention.output \
--lora_alpha 32 \
--do_train --do_eval --max_seq_length 128 \
--per_device_train_batch_size 32 --learning_rate 2.2e-3 \
--num_train_epochs 25 --warmup_steps 100 \
--cls_dropout 0.2 --weight_decay 0.1 \
--evaluation_strategy steps --eval_steps 100 \
--save_strategy steps --save_steps 1000 \
--logging_steps 50 \
--tb_writter_loginterval 50 \
--report_to tensorboard \
--seed ${seed} \
--root_output_dir ./output/glue/stsb \
--overwrite_output_dir