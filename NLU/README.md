# Adapating DeBERTaV3 with IncreLoRA

The folder contains the implementation of IncreLoRA in DeBERTaV3 using the updated package of `loralib`, which contains the implementation of IncreLoRA. 


## Setup Environment

### Create and activate the conda env
```bash
conda create -n NLU python=3.7
conda activate NLU 
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

Install `transformers`: (here we fork NLU examples from [microsoft/LoRA](https://github.com/microsoft/LoRA/tree/main/examples/NLU) and build our examples based on their `transformers` version, which is `v4.4.2`.)
```bash
pip install -e . 
```

Install the updated `loralib`:
```bash
pip install -e ../loralib/
```


## Fine-tune DeBERTaV3 on GLUE benchmark

### The example to reproduce the QNLI results

```bash
python \
examples/text-classification/run_glue.py \
--experiment_name ${experiment_name} \
--model_name_or_path microsoft/deberta-v3-base \
--task_name qnli \
--apply_lora --apply_increlora --lora_type svd \
--target_rank 2  --lora_r 1   \
--reg_orth_coef 0.1 \
--init_warmup 500 --incre_interval 500 \
--top_h 12 \
--beta1 0.85 --beta2 0.85 \
--lora_module query,key,value,intermediate,layer.output,attention.output \
--lora_alpha 32 \
--lora_dropout 0. \
--do_train --do_eval --max_seq_length 512 \
--per_device_train_batch_size 32 --learning_rate 7e-4  \
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

The floder `scripts` contains more examples of adapting DeBERTaV3-base with IncreLoRA on GLUE datasets. 

