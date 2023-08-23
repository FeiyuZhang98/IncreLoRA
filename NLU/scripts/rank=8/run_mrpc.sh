export CUDA_VISIBLE_DEVICES=0

experiment_name=$1

python \
examples/text-classification/run_glue.py \
--advance_learn True \
--multi_lr True \
--lora_dropout 0.3 \
--experiment_name ${experiment_name} \
--model_name_or_path microsoft/deberta-v3-base \
--task_name mrpc \
--apply_lora --apply_increlora --lora_type svd \
--target_rank 8   --lora_r 1   \
--reg_orth_coef 0.1 \
--init_warmup 100 --incre_interval 100 \
--top_h 10 \
--beta1 0.85 --beta2 0.85 \
--lora_module query,key,value,intermediate,layer.output,attention.output \
--lora_alpha 32 \
--do_train --do_eval --max_seq_length 320 \
--per_device_train_batch_size 32 --learning_rate 1e-3 \
--num_train_epochs 30 --warmup_steps 100 \
--cls_dropout 0.0 --weight_decay 0.01 \
--evaluation_strategy steps --eval_steps 100 \
--save_strategy steps --save_steps 3000 \
--logging_steps 100 \
--report_to tensorboard \
--seed 41 \
--root_output_dir ./output/glue/mrpc \
--overwrite_output_dir