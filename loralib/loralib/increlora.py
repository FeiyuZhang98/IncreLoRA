#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import ipdb
import re
import numpy as np

from .layers import LoRALayer 
from typing import Optional, List 

class loraW(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, A, E, B, scaling, ranknum):
        return torch.cat([b for b in B], 1) @                                 \
                (torch.cat([a for a in A], 0) * torch.cat([e for e in E], 0)) \
                    * scaling / (ranknum+1e-5)
    
class SVDLinear(nn.Linear, LoRALayer):
    # SVD-based adaptation implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, 
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.module_name = ""
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.ParameterList([nn.Parameter(
                self.weight.new_zeros((r, in_features))
            )])
            self.lora_E = nn.ParameterList([nn.Parameter(
                self.weight.new_zeros(r, 1)
            )])
            self.lora_B = nn.ParameterList([nn.Parameter(
                self.weight.new_zeros((out_features, r))
            )])
            self.W = loraW()
            self.hook_handle = self.W.register_full_backward_hook(self.backward_hook)
            
            self.score = 0
            self.gradMatrix_trace = 0
            self.ranknum = nn.Parameter(
                self.weight.new_zeros(1), requires_grad=False
            )
            self.ranknum.data.fill_(float(self.r))
            self.scaling = self.lora_alpha if self.lora_alpha>0 else float(self.r)   
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            self.ranknum.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def backward_hook(self, module, grad_input, grad_output):
        # print("Output_Grad:", grad_output)
        grad_Matrix = grad_output[0]
        try:
            W = (
                
                 self.W(self.lora_A, self.lora_E, self.lora_B, self.scaling, self.ranknum)
                 ).abs()
            # scale_W = torch.mean(W)
            scale_W=1
            self.score = torch.sum(((W / scale_W) * grad_Matrix).abs().detach()) / math.sqrt(W.numel())
            # self.score = torch.mean((grad_Matrix ** 2).detach())
        except:
            ipdb.set_trace()
        
    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A,B the same way as the default for nn.Linear 
            # and E (singular values) for zero 
            nn.init.zeros_(self.lora_E[0])
            nn.init.normal_(self.lora_A[0], mean=0.0, std=0.02)
            nn.init.normal_(self.lora_B[0], mean=0.0, std=0.02)

    def add_reserve_param(self, add_r, advance_learn=True):
        for _ in range(add_r):
            e = nn.Parameter(self.weight.new_zeros(1, 1), requires_grad=False)
            a = nn.Parameter(self.weight.new_zeros((1, self.in_features)), requires_grad=advance_learn)
            b = nn.Parameter(self.weight.new_zeros((self.out_features, 1)), requires_grad=advance_learn)
            e[0][0] = 1e-5 if advance_learn else 0.
            nn.init.normal_(a, mean=0.0, std=0.02)
            nn.init.normal_(b, mean=0.0, std=0.02)
            self.lora_E.append(e)
            self.lora_A.append(a)
            self.lora_B.append(b)
    
    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode == True:
            self.lora_A.requires_grad = True
            self.lora_E.requires_grad = True
            self.lora_B.requires_grad = True
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(
                        self.W(self.lora_A, self.lora_E, self.lora_B, self.scaling, self.ranknum)
                    )
                self.merged = False
        else:
            self.lora_A.requires_grad = False
            self.lora_E.requires_grad = False
            self.lora_B.requires_grad = False
    
    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += T(
                    self.W(self.lora_A, self.lora_E, self.lora_B, self.scaling, self.ranknum)
                )
            self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                try:
                    result += (
                        self.lora_dropout(x) @ self.W(self.lora_A, self.lora_E, self.lora_B, self.scaling, self.ranknum).T
                    )
                except:
                    ipdb.set_trace()
                    print(self.W)
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class RankAllocator(object):
    """
    The RankAllocator for IncreLoRA Model that will be called every training step. 

    Args:
        model: the model that we apply IncreLoRA to.
        lora_r (`int`): The initial rank for each incremental matrix.
        target_rank (`int`): The target average rank of incremental matrix.
        init_warmup (`int`): The steps of initial fine-tuning warmup.
        incre_interval (`int`): The time internval between two budget allocations.
        top_h (`int`): The number of modules selected.
        advance_learn (`bool`): With or without advance learning.
        beta1 (`float`): The hyperparameter of EMA for sensitivity smoothing.
        beta2 (`float`): The hyperparameter of EMA for undertainty quantification.
        total_step (`int`): The total training steps, correctly configured before training.
        target_total_rank (`Optinal[int]`): The speficified final total rank. 
        tb_writter (`SummaryWriter`): Tensorboard SummaryWriter. 
        tb_writter_loginterval (`int`): The logging interval of SummaryWriter. 
    """
    def __init__(
        self, model, 
        lora_r:int,
        target_rank:int, 
        init_warmup:int, 
        incre_interval:int,
        top_h:int,
        advance_learn:bool,
        beta1:float, 
        beta2:float, 
        total_step:Optional[int]=None, 
        target_total_rank:Optional[int]=None,
        weight_decay=None,
        incre_rank_num=None,
        tb_writter=None,
        tb_writter_loginterval:int=500, 
    ):
        self.ave_target_rank = target_rank
        self.target_rank = target_total_rank
        self.lora_init_rank = lora_r 

        self.init_warmup = init_warmup
        self.incre_interval = incre_interval
        self.advance_learn = advance_learn
        self.top_h = top_h
        if incre_rank_num:
            self.incre_rank_num = incre_rank_num
        else:
            rank_dic = {2:1, 4:2, 6:3, 8:4}
            self.incre_rank_num = rank_dic[self.ave_target_rank]
        
        self.beta1 = beta1
        self.beta2 = beta2
        self.total_step = total_step

        self.model = model
        self.weight_decay = weight_decay
        self.ipt = {} 
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}
        self.cat_ipt = {}
        self.rank_pattern = {} 
        self.get_lora_param_name()
        self.total_rank = self.initial_total_rank 

        self.tb_writter = tb_writter
        self.log_interval = tb_writter_loginterval 
        
        assert (self.beta1<1 and self.beta1>0)
        assert (self.beta2<1 and self.beta2>0)
        
            
    def set_total_step(self, total_step:int): 
        # Set total step number 
        self.total_step = total_step
        rank_per_round = self.top_h * self.incre_rank_num
        total_round = math.ceil((self.target_rank - self.initial_total_rank) / rank_per_round)
        total_incre_step = self.incre_interval * total_round
                            
        print("Total incremental step: total_incre_step: {}, of total steps: {:.0%}"
              .format(total_incre_step, total_incre_step / total_step))

    def get_rank_pattern(self):
        # Return rank pattern 
        return self.rank_pattern

    def get_lora_param_name(self):
        # Prepare the budget scheduler 
        self.name_set = set() 
        self.initial_total_rank = 0 
        self.shape_dict = {}
        for n, layer in self.model.named_modules():
            if isinstance(layer, SVDLinear):
                self.name_set.add(n)
                self.initial_total_rank += layer.lora_A[0].size(0) 
                self.shape_dict[n+'.lora_A'] = layer.lora_A[0].shape
                self.shape_dict[n+'.lora_B'] = layer.lora_B[0].shape
                
        self.name_set = list(sorted(self.name_set))
        if self.target_rank is None:
            self.target_rank = self.ave_target_rank * len(self.name_set) 


    def update_ipt(self, model): 
        for n, layer in model.named_modules():
            if isinstance(layer, SVDLinear):
                if n not in self.ipt:
                    self.ipt[n] = 0
                    self.exp_avg_ipt[n] = 0
                    self.exp_avg_unc[n] = 0
                
                # self.tb_writter.add_scalar("GradMatrix_Rank/%s"%(n[:-7],), layer.gradMatrix_rank, global_step)
                try:
                    self.ipt[n] = layer.score
                
                    # Update sensitivity 
                    self.exp_avg_ipt[n] = self.beta1 * self.exp_avg_ipt[n] + \
                                        (1-self.beta1)*self.ipt[n]
                    # Update uncertainty 
                    self.exp_avg_unc[n] = self.beta2 * self.exp_avg_unc[n] + \
                                        (1-self.beta2)*(self.ipt[n]-self.exp_avg_ipt[n]).abs()
                except:
                    ipdb.set_trace()
                    print(layer)

    def calculate_score(self, n, layer, metric="ipt"):
        if metric == "ipt":
            # Combine the senstivity and uncertainty 
            ipt_score = self.exp_avg_ipt[n] * self.exp_avg_unc[n]
        elif metric == "mag":
            ipt_score = 0.
            for n,p in layer.named_parameters():
                ipt_score += p.abs().detach().clone() 
        else:
            raise ValueError("Unexcptected Metric: %s"%metric)
        return ipt_score 

    def increase_to_target_rank(self, model, optimizer): 
        is_dict = {}
        all_is = []
        # Calculate the importance score for each sub matrix 
        for n, layer in model.named_modules():
            if isinstance(layer, SVDLinear):
                ipt_score = self.calculate_score(n, layer, metric="ipt")                
                is_dict[n] = ipt_score
                all_is.append(ipt_score)

        # Calculate the increasing threshold 
        k = min(self.top_h, self.target_rank - self.total_rank)
        increase_threshold = torch.topk(torch.tensor(all_is), k)[0][-1].item() 
        with torch.no_grad():
            curr_sum_rank = 0
            sum_param = 0
            new_param_list = []
            add_r = self.incre_rank_num
            for n, layer in model.named_modules():
                if isinstance(layer, SVDLinear):
                    if is_dict[n] >= increase_threshold:
                        # rank increase 1
                        layer.ranknum += add_r
                        self.total_rank += add_r
                        
                        # add lora_E
                        for param in layer.lora_E[ -add_r: ]:
                            param.requires_grad = True
                            new_param_list.append(param)
                        
                        if self.advance_learn:
                            layer.add_reserve_param(add_r, True)
                            new_param_list.extend(layer.lora_A[ -add_r: ])
                            new_param_list.extend(layer.lora_B[ -add_r: ])
                        else:
                            for param in layer.lora_A[ -add_r: ]:
                                param.requires_grad = True
                                new_param_list.append(param)
                            for param in layer.lora_B[ -add_r: ]:
                                param.requires_grad = True
                                new_param_list.append(param)
                            layer.add_reserve_param(add_r, False)
                            
                        print("The lora parameters rank of {} increased by {}".format(n, add_r))
                    
                    ranknum = layer.ranknum
                    if self.tb_writter is not None:
                        self.tb_writter.add_scalar("Ranknum/%s"%(n,), ranknum, self.global_step) 
                        self.rank_pattern[n] = int(ranknum.item())
                        curr_sum_rank += ranknum
                        sum_param += ranknum*self.shape_dict[n+".lora_A"][1]  
                        sum_param += ranknum*self.shape_dict[n+".lora_B"][0]  

            optimizer.add_param_group({'params': new_param_list, "weight_decay": self.weight_decay,})
            
            if self.total_rank == self.target_rank:
                for name, module in model.named_modules():
                    if isinstance(module, SVDLinear):
                        module.hook_handle.remove()
                        for param in module.lora_E[ -add_r: ]:
                            param.fill_(0.)
                            
            if self.tb_writter is not None:
                self.tb_writter.add_scalar("Budget/total_rank", curr_sum_rank, self.global_step)
                self.tb_writter.add_scalar("Budget/increase_threshold", increase_threshold, self.global_step)
                self.tb_writter.add_scalar("Budget/sum_param", sum_param, self.global_step)

        return increase_threshold


    def update_and_increase(self, model, global_step, optimizer):
        self.global_step = global_step
        increase_threshold=None
        add_r = self.incre_rank_num    
        # 为模型添加初始的储备参数
        if global_step == 0:
            new_param_list = []
            for name, module in model.named_modules():
                    if isinstance(module, SVDLinear):
                        module.add_reserve_param(add_r, self.advance_learn)
                        new_param_list.extend(module.lora_A[ -add_r: ])
                        new_param_list.extend(module.lora_B[ -add_r: ])
            if self.advance_learn:
                optimizer.add_param_group({'params': new_param_list, "weight_decay": self.weight_decay,})
        
        if self.total_rank < self.target_rank:
            self.update_ipt(model)
            if global_step > self.init_warmup and global_step % self.incre_interval == 0:
                increase_threshold = self.increase_to_target_rank(model, optimizer) 
        
        self._maybe_tb_writter_log(model)
        return self.top_h, increase_threshold

    def _maybe_tb_writter_log(self, model):
        def compute_and_log(mat_cov, name):
            I = torch.eye(*mat_cov.size(), out=torch.empty_like(mat_cov))
            I.requires_grad = False
            orth_regu = torch.norm(mat_cov-I, p="fro")
            regu_loss.append(orth_regu.item())
            self.tb_writter.add_scalar(
                "Orth_regu_loss/%s"%name, orth_regu.item(), self.global_step
            )
            
        if self.tb_writter is not None and self.global_step%self.log_interval==0:
            with torch.no_grad():
                regu_loss = []
                for n, layer in model.named_modules():
                    if isinstance(layer, SVDLinear):
                        wA = torch.cat([a for a in layer.lora_A], 0) 
                        wB = torch.cat([b for b in layer.lora_B], 1)
                        mat_cov_A = wA @ wA.T
                        mat_cov_B = wB.T @ wB 
                        compute_and_log(mat_cov_A, n+'.lora_A')
                        compute_and_log(mat_cov_B, n+'.lora_B')

                self.tb_writter.add_scalar(
                    "train/orth_regu_loss", sum(regu_loss)/len(regu_loss), self.global_step
                )


def compute_orth_regu(model, regu_weight=0.1):
    # The function to compute orthongonal regularization for SVDLinear in `model`. 
    regu_loss, num_param = 0., 0
    for n,p in model.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            para_cov = p @ p.T if "lora_A" in n else p.T @ p 
            I = torch.eye(*para_cov.size(), out=torch.empty_like(para_cov))
            I.requires_grad = False
            regu_loss += torch.norm(para_cov-I, p="fro")
            num_param += 1
    return regu_weight*regu_loss/num_param