
export CUDA_VISIBLE_DEVICES=0

experiment_name=$1

seeds=(41 42 43 44 45)
top_h=2

for seed in ${seeds[@]}
do
    python \
    examples/text-classification/run_glue.py \
    --experiment_name ${experiment_name} \
    --model_name_or_path microsoft/deberta-v3-base \
    --task_name mnli \
    --apply_increlora --apply_lora --lora_type svd \
    --target_rank 2  --lora_r 1  \
    --reg_orth_coef 0.1 \
    --init_warmup 1000 --incre_interval 1000 \
    --top_h ${top_h} \
    --beta1 0.85 --beta2 0.85 \
    --lora_module query,key,value,intermediate,layer.output,attention.output \
    --lora_alpha 16 \
    --lora_dropout 0.2 \
    --do_train --do_eval \
    --max_seq_length 256 \
    --per_device_train_batch_size 32 --learning_rate 3.5e-4 --num_train_epochs 9 \
    --warmup_steps 1000 \
    --cls_dropout 0.15 --weight_decay 0 \
    --evaluation_strategy steps --eval_steps 1000 \
    --save_strategy steps --save_steps 10000 \
    --logging_steps 500 \
    --seed ${seed} \
    --root_output_dir ./output/glue/mnli \
    --overwrite_output_dir
done