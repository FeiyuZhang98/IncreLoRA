# IncreLoRA: Incremental Parameter Allocation Method for Parameter-Efficient Fine-tuning

## Repository Overview

There are several directories in this repo:

* [loralib/](loralib) contains the source code of the updated package `loralib`, which include our implementation of IncreLoRA ([loralib/increlora.py](loralib/loralib/increlora.py)) and needs to be installed to run the examples;
* [NLU/](NLU) contains an example implementation of IncreLoRA in DeBERTaV3-base, which produces the results on the GLUE benchmark;
* [NLG_QA/](NLG_QA) contains an example implementation of IncreLoRA in BART-large and DeBERTaV3-base, which can be used to reproduce the results of summarization and question-answering tasks. 


## Quickstart of IncreLoRA

1. Install the updated `loralib`:

  ```bash 
  pip install -e loralib/ 
  ```


2. Then we apply SVD-based adaptation of IncreLoRA. Here is an example (For more examples, please see [modeling_debertav2.py](NLU/src/transformers/models/deberta_v2/modeling_deberta_v2.py) for how we adapte DeBERTa): 

  ```python
  # ===== Before =====
  # layer = nn.Linear(in_features, out_features)
  
  # ===== After ======
  import loralib 
  # Add a SVD-based adaptation matrices with rank r=12
  layer = loralib.SVDLinear(in_features, out_features, r=12)
  ```

   Also, before the training loop begins, mark only LoRA parameters as trainable.
  ```python
  model = BigModel()
  # This sets requires_grad to False for all parameters without the string "lora_" in their names
  loralib.mark_only_lora_as_trainable(model)
  ```

3. During the training loop, we apply RankAllocator of IncreLoRA to update importance scores of incremental matrices and allocate budget accordingly. 
  ```python
  from loralib import RankAllocator
  from loralib import compute_orth_regu 
  # Initialize the RankAllocator 
  rankallocator = RankAllocator(
      model, lora_r=1, target_rank=2,
      init_warmup=1000, incre_interval=1000, 
      top_h=2, beta1=0.85, beta2=0.85, 
  )
  ```
+ `lora_r`: The initial rank of each incremental matrix. 
+ `target_rank`: The average target rank of final incremental matrices, i.e. the average number of singular values per matrix. 
+ `init_warmup`: The steps of initial warmup for budget scheduler.
+ `incre_interval`: The time internval between two budget allocations.
+ `top_h`: The number of selected modules per allocation.
+ `beta1` and `beta2`: The coefficient of exponentional moving average when updating importance scores. 

  At each step of back-propagation, we apply an additional regularization to enforce the orthongonality of `SVDLinear` modules by `compute_orth_regu(model)`. Before each step of `optimizer.step()`, we call `RankAllocator` to update importance estimation and allocate the budget accordingly: 
  ```python
  # ===== Before =====
  # loss.backward() 
  # optimizer.step() 
  # global_step += 1 
  
  # ===== After ======
  (loss+compute_orth_regu(model, regu_weight=0.1)).backward
  rankallocator.update_and_increase(model, global_step)
  optimizer.step()
  global_step += 1
  ```


## GLUE benchmark

Check the folder `NLU` for more details about reproducing the GLUE results. 
An example of adapting DeBERTaV3-base on MNLI: 

```bash
python \
examples/text-classification/run_glue.py \
--model_name_or_path microsoft/deberta-v3-base \
--task_name mnli \
--apply_increlora --apply_lora --lora_type svd \
--target_rank 2  --lora_r 1  \
--reg_orth_coef 0.1 \
--init_warmup 1000 --incre_interval 1000 \
--top_h 2 \
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
--seed 41 \
--root_output_dir ./output/glue/mnli \
--overwrite_output_dir
```

Please see [`NLU/scripts`](NLU/scripts/) for more examples of GLUE. 


## Summarization and Question Answering Task

Check the folder [`NLG_QA`](NLG_QA/) for more details about reproducing the results of summarization and question-answering tasks.  
An example of adapting DeBERTaV3-base on SQuADv2: 

```bash
python \
examples/question-answering/run_qa.py \
--advance_learn True \
--multi_lr True \
--model_name_or_path microsoft/deberta-v3-base \
--dataset_name squad_v2 \
--apply_lora --apply_increlora \
--lora_type svd --target_rank 2 --lora_r 1 \
--reg_orth_coef 0.1 \
--init_warmup 1000 --incre_interval 1000 \
--top_h 1 \
--incre_rank_num 1 \
--beta1 0.85 --beta2 0.85 \
--lora_module query,key,value,intermediate,layer.output,attention.output \
--lora_alpha 16 \
--lora_dropout 0. \
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
```
