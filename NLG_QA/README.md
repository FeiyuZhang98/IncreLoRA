## Setup Environment

### Create and activate the conda env
```bash
conda create -n NLG python=3.7
conda activate NLG 
```

### Install Pytorch
```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

### Install the pre-requisites
Install dependencies: 
```bash
pip install -r requirements.txt
```

Install `transformers`: (here we build our examples based on the latest `transformers` at the time we conduct experiments, which is `v4.21.0` and has better support for summarization tasks.)
```bash
pip install -e . 
```

Install the updated `loralib`:
```bash
pip install -e ../loralib/
```


## Adapt BART on summarization tasks

### The example to reproduce the XSum results

```bash
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
--top_h 13 \
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
```


### Instructions

#### Hyperparameter Setup

+ `apply_lora`: Apply LoRA to the target model. 
+ `lora_type`: Config the low-rank parameterization, `frd` for low-rank decomposition and `svd` for SVD decomposition. Use `svd` for IncreLoRA and `frd` for LoRA. 
+ `apply_increlora`: Further apply IncreLoRA for the model that have been modified by LoRA. 
+ `lora_module`: The types of modules updated by LoRA. 
+ `lora_r`: The initial rank of each incremental matrix. 
+ `top_h`: The number of modules selected. 
+ `target_rank`: The average target rank of final incremental matrices, i.e. the average number of singular values per matrix. 
+ `init_warmup`: The steps of initial warmup for budget scheduler.
+ `incre_interval`: The time internval between two budget allocations.
+ `beta1` and `beta2`: The coefficient of exponentional moving average when updating importance scores. 
+ `reg_orth_coef`: The weight of orthongonal regularization. 


### Other examples

The floder `scripts` contains more examples of adapting BAET-large and DeBERTaV3-base with IncreLoRA on summarization and question-answering tasks.  

