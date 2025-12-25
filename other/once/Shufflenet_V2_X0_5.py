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

import heapq
import math
from collections import defaultdict

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
    value_mask_flag = torch.zeros(31, x_2D.shape[0], x_2D.shape[1], dtype=torch.int16)    
    value_mask_flag[0] = (x_2D==0) 

    entropy_value_mask_flag = torch.zeros(x_2D.shape[0], x_2D.shape[1], dtype=torch.int16)       

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

def Quantize_for_mask(x_torch, mask):
    x_shape = x_torch.shape

    mask_flag = torch.zeros((16, x_shape[0], x_shape[1]), dtype=torch.int16)
    for idx in range(16):
        mask_flag[idx] = (torch.abs(x_torch) == (idx)) # flag[1] = 
    
    # [0xffff, 0xfffe, 0xfffc, 0xfff8, 0xfff0, 0xffe0, 0xffc0, 0xff80, 
    #  0xff00, 0xfe00, 0xfc00, 0xf800, 0xf000, 0xe000, 0xc000, 0x8000]    
    
    # mask            = torch.tensor([-1,-2,-4,-8,-16,-32,-64,-128,-256,-256,-256,-256,-256,-512,-512,-1024])
    
    offset_mapped   = torch.tensor([int(abs(mask[i]/2)) if mask[i]!= -1 else 0 for i in range(16)], dtype=torch.int16)
    offset_max      = torch.tensor([(16384 >> (15 - i)) for i in range(16)], dtype=torch.int16)   
    # [0x0000, 0x0001, 0x0002, 0x0004, 0x0008, 0x0010, 0x0020, 0x0040, 
    #  0x0080, 0x0100, 0x0200, 0x0400, 0x0800, 0x1000, 0x2000, 0x4000]
        
    offset = torch.tensor([offset_mapped[i] if (offset_mapped[i]<=offset_max[i]) else offset_max[i] for i in range(16)], dtype=torch.int16)      

    mask_block = torch.sum(mask_flag * mask.reshape(-1,1,1), dim=0)
    offset_block = torch.sum(mask_flag * offset.reshape(-1,1,1), dim=0)

    return mask_block.type(torch.int16).cuda(), offset_block.type(torch.int16).cuda() # for lossless+lossy

def huffman_code(symbols, probabilities):
    """Huffman coding algorithm to calculate the optimal prefix code."""
    # Create a priority queue to store nodes of the Huffman tree.
    heap = [[probability, [symbol, ""]] for symbol, probability in zip(symbols, probabilities)]
    heapq.heapify(heap)
    
    # Combine nodes until there is only one node left in the heap.
    while len(heap) > 1:
        # Get the two nodes with the lowest probabilities.
        low_prob_node_1 = heapq.heappop(heap)
        low_prob_node_2 = heapq.heappop(heap)
        
        # Combine the nodes and add the result to the heap.
        for pair in low_prob_node_1[1:]:
            pair[1] = '0' + pair[1]
        for pair in low_prob_node_2[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [low_prob_node_1[0] + low_prob_node_2[0]] + low_prob_node_1[1:] + low_prob_node_2[1:])
    
    # Return the Huffman code as a dictionary.
    huffman_dict = dict(heapq.heappop(heap)[1:])
    # print("huffman_dict",huffman_dict)
    return huffman_dict

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
    probability = []
    for idx in range(-32767,32768):
        x_entropy[idx] = torch.round(x_count[idx]/torch.sum(x_count),decimals=5)
        if x_count[idx] != 0:
            # write_ws.append(["{0}".format(idx),"{0}".format(x_entropy[idx]), "{0}".format(x_count[idx])])
            # print("symbol : {0}, percent : {1}, freq : {2}".format(idx, x_entropy[idx], x_count[idx]))
            probability.append(x_entropy[idx].item())
            # print("in for",probability)
    # print(probability)
    huffman_coding_results(probability)
    return symbol_frequency(x_entropy)

def symbol_frequency(symbol_set):
    bit_set = torch.zeros((65535))
    for idx in range(-32767,32768):
        # print("symbol_set[idx] type : ",symbol_set[idx].dtype, symbol_set[idx])
        if(symbol_set[idx] !=0):
            # print(idx, symbol_set[idx])
            bit_set[idx] = torch.round((symbol_set[idx] * math.log2(symbol_set[idx])), decimals=5)
        else :
            bit_set[idx] = 0
    entropy = -1 * (torch.round(torch.sum(bit_set), decimals=5))
    return entropy

def huffman_coding_results(probability):
    # Define the entropy values and corresponding symbols.
    # input (probability)    
    
    #entropy_values = [0.4, 0.25, 0.2, 0.1, 0.05]
    cnt = len(probability)
    symbols = [ f"{i}" for i in range(cnt)]
  
    # Calculate the Huffman code and the number of bits required for each symbol.
    huffman_dict = huffman_code(symbols, probability)
    symbol_bits = {symbol: len(huffman_dict[symbol]) for symbol in symbols}
    
    bit_total=0
    code_length = 0
    
    # Print the results.
    for symbol, bits in symbol_bits.items():
        # print(symbol, "\t", bits)
        bit_total += bits
    for symbol, probability in zip(symbols, probability):
        code_length += symbol_bits[symbol]*probability
    print("symbols counts :",cnt, "huffmanc coding code_length",code_length)
    return cnt,code_length
    


# Write_workbook
write_workbook = Workbook()
write_ws = write_workbook.active
write_ws['C3'] = 'value_index_loss'
index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
write_ws.append(index)

# parameter print
fp_parameters = torch.tensor([])
parameters = torch.tensor([],dtype=torch.int16)
quantize_parameters = torch.tensor([],dtype=torch.int16)
param_list = torch.tensor([])
#parameters_index = np.array([])

# initialize variables
length          = 32 # set block size 
result          = []
# mask_value_list = [-2,-4,-8,-16,-32,-64,-128,-256,-512,-1024,-2048,-4096]        
# other_value_list = [-1,-2,-4,-8,-16,-32,-64,-128,-256,-512]

# mask              = torch.tensor([-1,-2,-4,-8,-16,-32,-64,-128,-256,-256,-256,-256,-256,-256,-256,-512])      # Case 1
# mask              = torch.tensor([-1,-2,-4,-8,-16,-32,-64,-128,-256,-512,-512,-512,-256,-512,-512,-1024])      # Case 2
# mask              = torch.tensor([-1,-2,-4,-8,-16,-32,-64,-128,-256,-512,-512,-512,-512,-512,-1024,-1024])      # Case 3

mask              = torch.tensor([-1,-2,-4,-8,-16,-32,-64,-128,-256,-512,-1024,-512,-1024,-1024,-1024,-2048])      # Case 5

print(mask)    
_model = fuse_bn_recursively(models.shufflenet_v2_x0_5(weights=models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1)).to('cuda')
# _model = fuse_bn_recursively(models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)).to('cuda')
model = nn.DataParallel(_model).to('cuda')
for name, param in tqdm(model.named_parameters()):
    with torch.no_grad():
        Data_shape = param.shape
        shapes = 1

        # fp_param_save = np.save("/home/kkh/pytorch/data/MobileNet_V2_original_fp/"+ name, param.cpu().numpy()) # per layer
        # fp_parameters = torch.cat([fp_parameters.reshape(-1,),param.reshape(-1).cpu()], dim=0)

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
        
        # param_save = np.save("/home/kkh/pytorch/data/MobileNet_V2_original/"+ name, Quant_input.cpu().numpy()) # per layer
        parameters = torch.cat([parameters.reshape(-1,),Quant_input.reshape(-1).cpu()], dim=0)
        
        # # # # # Step 2 : classify (value 기준으로)
        value_aware_block = value_aware_classify(Quant_input,length)        # shift (294*32) 4bit        
        mask_block, offset_block = Quantize_for_mask(value_aware_block,mask)        # mask (294*32) INT16 
        Quant_input = ((Quant_input.reshape(mask_block.shape) & mask_block) + offset_block).reshape(Quant_input_shape)#
        #print("Quant_input :",Quant_input.dtype)

        # quant_param_save = np.save("/home/kkh/pytorch/data/MobileNet_V2_quantize/"+ name, Quant_input.cpu().numpy()) # per layer
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


# fp_total_param_save = np.save("/home/kkh/pytorch/data/MobileNet_V2_original_fp/total.weight",fp_parameters)
# total_param_save = np.save("/home/kkh/pytorch/data/MobileNet_V2_original/total.weight",parameters)
# total_quant_param_save = np.save("/home/kkh/pytorch/data/MobileNet_V2_quantize/total.weight",quantize_parameters)
        
dataset = dsets.ImageFolder("/media/imagenet/val", models.SqueezeNet1_0_Weights.IMAGENET1K_V1.transforms()) ### 2번째 인자, transform
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
# mask            = torch.tensor([mask_value if(i==n) else other_value  for i in range(16)])
# print("n : ",n,"mask_value : ",mask_value,"other_value : ",other_value)
result += mask.tolist()
result.append(round(acc1.tolist(),3))
write_ws.append(result)
# write_workbook.save('/home/kkh/pytorch/data/Squeeze.xlsx')
result = []
# if abs(mask_value) == 2**(n+8):
#     break
# if(round(acc1.tolist(),3) < 50):
#     break

# write_workbook.save('/home/kkh/pytorch/data/23.01.20_param_all_loss.xlsx')


# %%
