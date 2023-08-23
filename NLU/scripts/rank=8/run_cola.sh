export CUDA_VISIBLE_DEVICES=0

experiment_name=$1

seeds=(41) 
top_h=3

for seed in ${seeds[@]}
do
python \
examples/text-classification/run_glue.py \
--advance_learn True \
--multi_lr True \
--experiment_name ${experiment_name}_seed_${seed} \
--model_name_or_path microsoft/deberta-v3-base \
--task_name cola \
--apply_lora --apply_increlora --lora_type svd \
--target_rank 8   --lora_r 1   \
--reg_orth_coef 0.1 \
--init_warmup 100 --incre_interval 100 \
--top_h ${top_h} \
--beta1 0.85 --beta2 0.85 \
--lora_module query,key,value,intermediate,layer.output,attention.output \
--lora_alpha 32 \
--do_train --do_eval --max_seq_length 64 \
--per_device_train_batch_size 32 --learning_rate 1e-3 \
--num_train_epochs 25 --warmup_steps 100 \
--cls_dropout 0.10 --weight_decay 0.00 \
--evaluation_strategy steps --eval_steps 100 \
--save_strategy steps --save_steps 10000 \
--logging_steps 10 \
--tb_writter_loginterval 100 \
--report_to tensorboard \
--seed ${seed} \
--root_output_dir ./output/glue/cola \
--overwrite_output_dir
done