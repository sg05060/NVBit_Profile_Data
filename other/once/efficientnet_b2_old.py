#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import torch, gc
import torchvision.models as models
import torchvision.datasets as dsets
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Iterable, Callable
from bn_fold import fuse_bn_recursively
import torch.fft
import math

from tqdm import tqdm
from openpyxl import Workbook

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cudnn.enabled = True
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

def Quant(x, n) :

    N = 2 ** n
    N_MIN, N_MAX = -N//2 + 1 , N//2 - 1
    x_max, x_min = torch.max(x) , torch.min(x)
    
    # Symmetric
    x_max_abs = torch.abs(x_max)
    x_min_abs = torch.abs(x_min)
    
    x_abs_flag = torch.ge(x_max_abs,x_min_abs)
    
    x_max = x_abs_flag*x_max_abs + (~x_abs_flag)*x_min_abs
    x_min = -x_max

    scale = (x_max - x_min) / (N-2)
    scale += (x_max * (scale == 0))
    zero_n = x_max * N_MIN - x_min * N_MAX
    zero_d = x_max - x_min
    zero_p =  torch.round(zero_n / (zero_d + 1e-30)) * (zero_d != 0) # [(x_max+x_min) / [(x_max-x_min)*(2^15)+1e-30] ]* (zero_d !=0)

    x_hat = torch.round(x / scale + zero_p) 
    x_q   = torch.clip(x_hat, N_MIN, N_MAX).type(torch.int16)

    return x_q, scale, zero_p
    
def DeQuant(    x_q,
                scale,
                zero_p):
    return scale  * (x_q.to('cuda') - zero_p)

def value_aware_classify(x_torch,length):
    x_shape = x_torch.shape
    x_2D = x_torch.reshape(-1,length)
    
    # value_offset (-15 ~ 15) 31
    value_index     = torch.tensor([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15], dtype=torch.int16)
    
    # value_classify (~15 ~ 15) 31
    value_mask_flag = torch.zeros(31,x_2D.shape[0],x_2D.shape[1], dtype=torch.int16)    
    value_mask_flag[0] = (x_2D==0) 

    entropy_value_mask_flag = torch.zeros(x_2D.shape[0],x_2D.shape[1], dtype=torch.int16)       

    # 양수 (1~14)     # 음수 (-14~-1)  
    for n in range(16,31):
        value_mask_flag[n] = ((x_2D <= -(2**(n-16))) & (x_2D >= -(2**(n-15))+1))
        entropy_value_mask_flag = ((x_2D <= -(2**(n-16))) & (x_2D >= -(2**(n-15))+1))
    #    print("range :",n, shannon_entropy_except_zero(entropy_value_mask_flag*x_2D))
    for n in range(1,16): 
        value_mask_flag[n] = ((x_2D>=2**(n-1)) & (x_2D <= 2**(n)-1)) 
        entropy_value_mask_flag = ((x_2D>=2**(n-1)) & (x_2D <= 2**(n)-1))
    #    print("range :",n, shannon_entropy_except_zero(entropy_value_mask_flag*x_2D))
    
    value_mask = torch.sum(value_mask_flag*value_index.reshape(-1,1,1), dim=0)
    # (31,x_2D.shape[0],x_2D.shape[1])
    return value_mask.type(torch.int16)

def Quantize_for_mask(x_torch, n, mask_value, default_value):
    x_shape = x_torch.shape

    mask_flag = torch.zeros((16, x_shape[0], x_shape[1]), dtype=torch.int16)
    for idx in range(16):
        mask_flag[idx] = (torch.abs(x_torch) == (idx)) # flag[1] = 
    
    # [0xffff, 0xfffe, 0xfffc, 0xfff8, 0xfff0, 0xffe0, 0xffc0, 0xff80, 
    #  0xff00, 0xfe00, 0xfc00, 0xf800, 0xf000, 0xe000, 0xc000, 0x8000]    
    
    # mask            = torch.tensor([mask_value if(i==abs(n)) else default_value  for i in range(16)], dtype=torch.int16)
    # mask            = torch.tensor([-1,-2,-4,-8,-16,-32,-64,-128,-256,-256,-256,-256,-256,-512,-512,-1024]) # mobilenet_V2
    # mask            = torch.tensor([-1,-2,-4,-8,-16,-32,-64,-128,-256,-256,-256,-256,-512,-512,-1024,-4096]) # resnet_50, VGG 19
    mask            = torch.tensor([-1,-2,-4,-8,-16,-32,-64,-128,-256,-256,-256,-256,-512,-512,-1024,-2048]) # Efficientnet_b2
    
    
    # print("mask :",mask)
    offset_mapped   = torch.tensor([int(abs(mask[i]/2)) if mask[i]!= -1 else 0 for i in range(16)], dtype=torch.int16)
    offset_max      = torch.tensor([(16384 >> (15 - i)) for i in range(16)], dtype=torch.int16)   
    # print("offset_max",offset_max)
    # [0x0000, 0x0001, 0x0002, 0x0004, 0x0008, 0x0010, 0x0020, 0x0040, 
    #  0x0080, 0x0100, 0x0200, 0x0400, 0x0800, 0x1000, 0x2000, 0x4000]
        
    offset = torch.tensor([offset_mapped[i] if (offset_mapped[i]<=offset_max[i]) else offset_max[i] for i in range(16)], dtype=torch.int16)      
    # print("offset :",offset)   


    mask_block = torch.sum(mask_flag * mask.reshape(-1,1,1), dim=0)
    offset_block = torch.sum(mask_flag * offset.reshape(-1,1,1), dim=0)

    return mask_block.type(torch.int16).cuda(), offset_block.type(torch.int16).cuda() # for lossless+lossy

def shannon_entropy(x_torch):
    x_reshape = x_torch.reshape(-1).type(torch.int16) #################### FP 시 수정 필요.
    x_count = torch.zeros((65535), dtype=torch.int64)
    for x in x_reshape:        # 2백만개 , 인덱스
        x_count[x] += 1
        # for value in range(-32767,32768): # 65535
        #     if x == value:                
        #         x_count[value] += 1
        #     #    x_count[idx] = (x_torch == idx)
        
    x_entropy = torch.zeros((65535))
    for idx in range(-32767,32768):
        x_entropy[idx] = torch.round(x_count[idx]/torch.sum(x_count),decimals=5)
        # if x_count[idx] != 0:
        write_ws.append(["{0}".format(idx),"{0}".format(x_entropy[idx]), "{0}".format(x_count[idx])])
        print("symbol : {0}, percent : {1}, freq : {2}".format(idx, x_entropy[idx], x_count[idx]))
    return symbol_frequency(x_entropy)


def symbol_frequency(symbol_set):
    bit_set = torch.zeros((65535))
    for idx in range(-32767,32768):
        # print("symbol_set[idx] type : ",symbol_set[idx].dtype, symbol_set[idx])
        if(symbol_set[idx] !=0):
            #print(idx, symbol_set[idx])
            bit_set[idx] = torch.round((symbol_set[idx] * math.log2(symbol_set[idx])), decimals=5)
        else :
            bit_set[idx] = 0
    entropy = -1 * (torch.round(torch.sum(bit_set), decimals=5))
    return entropy

write_workbook = Workbook()
write_ws = write_workbook.active
write_ws['C3'] = 'value_index_loss'
            
result = []

length = 32
count = 1
dtype = 10
value_list = [-8]#[-(1 << i) for i in range(1)] # -2 # 잘라내는 값 ex, -2 = 0xfe
# value_list = [0,-4] # -2 # 잘라내는 값 ex, -2 = 0xfe

index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
write_ws.append(index)

# mask_value_list = [-2,-4,-8,-16,-32,-64,-128,-256,-512,-1024,-2048,-4096]        
# other_value_list = [-1,-2,-4,-8,-16,-32,-64,-128,-256,-512]

mask_value = -16384
other_value = -1
n = 15
# _model = fuse_bn_recursively(models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)).to('cuda')
# _model = fuse_bn_recursively(models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)).to('cuda')
_model = fuse_bn_recursively(models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)).to('cuda')
model = nn.DataParallel(_model).to('cuda')
#parameters_index = np.array([])
fp_parameters = torch.tensor([])
parameters = torch.tensor([],dtype=torch.int16)
quantize_parameters = torch.tensor([],dtype=torch.int16)
param_list = torch.tensor([])
for name, param in tqdm(model.named_parameters()):
    with torch.no_grad():
        Data_shape = param.shape
        shapes = 1

        fp_param_save = np.save("/home/kkh/pytorch/data/efficientnet_b2_original_fp/"+ name, param.cpu().numpy()) # per layer
        fp_parameters = torch.cat([fp_parameters.reshape(-1,),param.reshape(-1).cpu()], dim=0)

        for i in Data_shape:
            shapes *= i
        if shapes%64 != 0 :
            Quant_input, scale, zero_p = Quant(param,16)#quantize_channel(param,16)
            param[:] = DeQuant(Quant_input, scale, zero_p)#dequantize_channel(Comp_output, scale, zero_p)
            continue
        if 'bn' in name:
            Quant_input, scale, zero_p = Quant(param,16)#quantize_channel(param,16)
            param[:] = DeQuant(Quant_input, scale, zero_p)#dequantize_channel(Comp_output, scale, zero_p)
            continue
        if 'bias' in name:
            Quant_input, scale, zero_p = Quant(param,16)#quantize_channel(param,16)
            param[:] = DeQuant(Quant_input, scale, zero_p)#dequantize_channel(Comp_output, scale, zero_p)
            continue
        

        # # # # # # Step 1 : Quantization (INT16)
        Quant_input, scale, zero_p = Quant(param,16)#quantize_channel(param,16)         # input (294*32) INT16
        Quant_input_shape = Quant_input.shape
        
        param_save = np.save("/home/kkh/pytorch/data/efficientnet_b2_original/"+ name, Quant_input.cpu().numpy()) # per layer
        parameters = torch.cat([parameters.reshape(-1,),Quant_input.reshape(-1).cpu()], dim=0)
        
        # # # # # Step 2 : classify (value 기준으로)
        value_aware_block = value_aware_classify(Quant_input,length)        # shift (294*32) 4bit        
        mask_block, offset_block = Quantize_for_mask(value_aware_block,n,mask_value,other_value)        # mask (294*32) INT16 
        Quant_input = ((Quant_input.reshape(mask_block.shape) & mask_block) + offset_block).reshape(Quant_input_shape)#
        #print("Quant_input :",Quant_input.dtype)

        quant_param_save = np.save("/home/kkh/pytorch/data/efficientnet_b2_quantize/"+ name, Quant_input.cpu().numpy()) # per layer
        quantize_parameters = torch.cat([quantize_parameters.reshape(-1,),Quant_input.reshape(-1).cpu()], dim=0)

        # parameters_index = np.append(parameters_index.reshape(-1,).tolist(), Quant_input.tolist())
        #print("parameters_index :", parameters_index.dtype)
        
        # # # Step 5 : DeQuantization (INT16)
        param[:] = DeQuant(Quant_input, scale, zero_p)#dequantize_channel(Comp_output, scale, zero_p)
        # param[:] = DeQuant(Comp_output, scale, zero_p)#dequantize_channel(Comp_output, scale, zero_p)

#@ Test 2 : total param

# print("total_fp32_param_entropy",shannon_entropy(fp_parameters).cpu())
# print("total_int16_param_entropy",shannon_entropy(parameters).cpu())
print("total_quant_int_param_entropy",shannon_entropy(quantize_parameters).cpu())

print("total_original_fp_parameter_size : ",fp_parameters.shape)
print("total_original_parameter_size : ",parameters.shape)
print("total_quantize_parameter_size : ",quantize_parameters.shape)

fp_total_param_save = np.save("/home/kkh/pytorch/data/efficientnet_b2_original_fp/total.weight",fp_parameters)
total_param_save = np.save("/home/kkh/pytorch/data/efficientnet_b2_original/total.weight",parameters)
total_quant_param_save = np.save("/home/kkh/pytorch/data/efficientnet_b2_quantize/total.weight",quantize_parameters)

##################################################
param_size = 0
buffer_size = 0

for param in model.parameters():
    param_size += param.nelement() * param.element_size()

for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**2
print('Size: {:.3f} MB'.format(size_all_mb))

####################################################
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("params",params)


        
dataset = dsets.ImageFolder("/media/imagenet/val", models.EfficientNet_B2_Weights.IMAGENET1K_V1.transforms()) ### 2번째 인자, transform
loader = DataLoader(dataset= dataset, # dataset
                batch_size= 128,   # batch size power to 2
                shuffle = False, # false
                num_workers = 8, # num_workers
                pin_memory=True) # pin_memory

correct = 0
total = 50000
accum = 0
model.eval()
# torch.no_grad()
with torch.no_grad():
    for idx, (input, label) in enumerate(tqdm(loader)):
        input = input.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)
        output = model(input)
        # print(output)
        pred = torch.argmax(output, 1)
        correct += (pred == label).int().sum()
        accum += 4
        if idx % 1000 == 0:
            print(idx, correct /accum * 100, correct, accum)
        acc1 = correct / total * 100
    print(acc1)
mask            = torch.tensor([mask_value if(i==n) else other_value  for i in range(16)])
print("n : ",n,"mask_value : ",mask_value,"other_value : ",other_value)
result += mask.tolist()
result.append(round(acc1.tolist(),3))
write_ws.append(result)
write_workbook.save('/home/kkh/pytorch/data/efficientnet_b2_Quantize.xlsx')
result = []
# if abs(mask_value) == 2**(n+8):
#     break
# if(round(acc1.tolist(),3) < 50):
#     break

# write_workbook.save('/home/kkh/pytorch/data/23.01.20_param_all_loss.xlsx')

# %%
