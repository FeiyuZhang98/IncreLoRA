export CUDA_VISIBLE_DEVICES=0
experiment_name=$1


top_h=13

python \
examples/summarization/run_summarization_no_trainer.py \
--model_name_or_path facebook/bart-large \
--dataset_name xsum \
--experiment_name ${experiment_name} \
--apply_lora --apply_increlora \
--lora_type svd --target_rank 2 --lora_r 1 \
--lora_alpha 32 \
--lora_dropout 0. \
--reg_orth_coef 0.1 \
--schedule_strategy epoch \
--incre_epoch 15 \
--init_warmup 3000  --incre_interval 3000 \
--top_h ${top_h} \
--beta1 0.85 --beta2 0.85 \
--lora_module q_proj,k_proj,v_proj,out_proj,fc1,fc2 \
--per_device_train_batch_size 8 --learning_rate 5e-4 \
--num_train_epochs 35 --num_warmup_steps 3000 \
--max_source_length 768 --max_target_length 64 --max_length 768 \
--pad_to_max_length --num_beams 8 \
--per_device_eval_batch_size 8 \
--seed 9 \
--with_tracking \
--tb_writter_loginterval 500 \
--output_dir ./output/bart-large/xsum 
