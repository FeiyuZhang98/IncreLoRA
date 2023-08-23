
export CUDA_VISIBLE_DEVICES=0

experiment_name=$1


seed=41 
lr=4e-4
top_h=3

python \
examples/text-classification/run_glue.py \
--experiment_name ${experiment_name}_lr=${lr} \
--model_name_or_path microsoft/deberta-v3-base \
--task_name qqp \
--apply_increlora --apply_lora --lora_type svd \
--target_rank 2  --lora_r 1  \
--reg_orth_coef 0.1 \
--init_warmup 1000 --incre_interval 1000 \
--top_h ${top_h} \
--beta1 0.85 --beta2 0.85 \
--lora_module query,key,value,intermediate,layer.output,attention.output \
--lora_alpha 16 \
--lora_dropout 0. \
--do_train --do_eval \
--max_seq_length 320 \
--per_device_train_batch_size 32 --learning_rate ${lr} \
--num_train_epochs 7 --warmup_steps 1000 \
--cls_dropout 0.15 --weight_decay 0.01 \
--evaluation_strategy steps --eval_steps 1000 \
--save_strategy steps --save_steps 10000 \
--logging_steps 500 \
--tb_writter_loginterval 500 \
--report_to tensorboard \
--seed ${seed} \
--root_output_dir ./output/glue/qqp \
--overwrite_output_dir