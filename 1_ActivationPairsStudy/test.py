# --------------------------------
# test phase
# --------------------------------

import numpy as np
import torch
import torch.nn as nn
import os
import cv2

from torch.utils.data import DataLoader
from PIL import Image

from models import *
from dataset_gray import *
from utils_model import *
from utils_image import *

datadir_test = 'testset/Classic5'
qf_candidates = [30]
batchsize_test = 1
base_dir = './result_test'
ckpt_dir = 'model_zoo/RRRNetwork_epoch1000.pth'
device = 'cpu'

model = RRRNetwork()
    
ckpt_name = os.path.basename(ckpt_dir)
model_name = os.path.splitext(ckpt_name)[0]
base_dir = os.path.join(base_dir, model_name)

dataset_test = DatasetJPEG(n_channels=1, patch_size=0, dataset_path=datadir_test, phase='test', qf_candidates=qf_candidates)
test_loader = DataLoader(dataset_test, batch_size=batchsize_test, shuffle=False, num_workers=0, drop_last=False)

num_data_test = len(dataset_test)
num_batch_test = np.ceil(num_data_test / batchsize_test)

blocked_dir = os.path.join(base_dir, 'image_before')
if not os.path.exists(blocked_dir):
    os.makedirs(blocked_dir)

output_dir = os.path.join(base_dir, 'image_after')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(ckpt_dir):
    raise ValueError(f"Model file '{ckpt_dir}' does not exist.")

dict_model = torch.load(ckpt_dir, map_location=device)
model.load_state_dict(dict_model['net'])

device = torch.device(device if torch.cuda.is_available() else 'cpu')

print('#' * 50)
print(f"{'INITIALIZE TEST ENVIRONMENT':^50s}")
print('#' * 50)

print(f'Testing on device: {device}')
print(f"datadir_test: {datadir_test}")
print(f"qf_candidates: {qf_candidates}")
print(f"batchsize_test: {batchsize_test}")
print(f"base_dir: {base_dir}")
print(f"ckpt_dir: {ckpt_dir}")
print(f"device: {device}")
print(f'\n\nRESULTS : ')


fn_tonumpy = lambda x: x.to('cpu').detach().numpy()
mean = lambda arr: sum(arr) / len(arr)

mse_blocked_arr = []
psnr_blocked_arr = []
psnrb_blocked_arr = []
ssim_blocked_arr = []
mse_output_arr = []
psnr_output_arr = []
psnrb_output_arr = []
ssim_output_arr = []

with torch.no_grad():
    
    model.eval()
    
    for order, data in enumerate(test_loader):

        # forward pass
        blocked = data['L'].to(device)
        ground_truth = data['H'].to(device)

        output = model(blocked)

        # 손실함수 계산
        
        output_cpu = output.detach().cpu()
        ground_truth_cpu = ground_truth.detach().cpu()
        
        loss = nn.MSELoss()
        
        mse_blocked = loss(blocked, ground_truth).item()
        psnr_blocked = calculate_psnr_batch(blocked, ground_truth_cpu)[0]
        psnrb_blocked = calculate_psnrb_batch(blocked, ground_truth_cpu)[0]
        ssim_blocked = calculate_ssim_batch(blocked, ground_truth_cpu)[0]
        
        mse_output = loss(output, ground_truth).item()
        psnr_output = calculate_psnr_batch(output_cpu, ground_truth_cpu)[0]
        psnrb_output = calculate_psnrb_batch(output_cpu, ground_truth_cpu)[0]
        ssim_output = calculate_ssim_batch(output_cpu, ground_truth_cpu)[0]
        
        mse_blocked_arr.append(mse_blocked)
        psnr_blocked_arr.append(psnr_blocked)
        psnrb_blocked_arr.append(psnrb_blocked)
        ssim_blocked_arr.append(ssim_blocked)
        mse_output_arr.append(mse_output)
        psnr_output_arr.append(psnr_output)
        psnrb_output_arr.append(psnrb_output)
        ssim_output_arr.append(ssim_output)
        
        # In Validation, MSE, PSNR, PSNR-B, SSIM Is Calculated
        # print last value of each array : because each element in array represents avg value of batch
    
        blocked = fn_tonumpy(blocked)
        ground_truth = fn_tonumpy(ground_truth)
        output = fn_tonumpy(output)
        
        blocked_image = np.squeeze(blocked, axis=0)
        blocked_image = np.squeeze(blocked_image, axis=0)
        blocked_image = (blocked_image * 255).astype(np.uint8)
        blocked_image = Image.fromarray(blocked_image)
        
        output_image = np.squeeze(output, axis=0)
        output_image = np.squeeze(output_image, axis=0)
        output_image = (output_image * 255).astype(np.uint8)
        output_image = Image.fromarray(output_image)
        
        blocked_path = os.path.join(blocked_dir, f"{dataset_test.paths_H[order].split('/')[-1].split('.')[0]}_blocked.png")
        blocked_image.save(blocked_path)
        
        output_path = os.path.join(output_dir, f"{dataset_test.paths_H[order].split('/')[-1].split('.')[0]}_output.png")
        output_image.save(output_path)
        
        print(f"DEBLOCKING PERFORMANCE OF {dataset_test.paths_H[order].split('/')[-1]}: MSE {mse_blocked:.6f} -> {mse_output:.6f}  | PSNR {psnr_blocked:.6f} -> {psnr_output:.6f} | PSNR-B {psnrb_blocked:.6f} -> {psnrb_output:.6f} | SSIM {ssim_blocked:.6f} -> {ssim_output:.6f}")
    
# calculate average values of the performance arrays
avg_mse_blocked = sum(mse_blocked_arr) / len(mse_blocked_arr)
avg_psnr_blocked = sum(psnr_blocked_arr) / len(psnr_blocked_arr)
avg_psnrb_blocked = sum(psnrb_blocked_arr) / len(psnrb_blocked_arr)
avg_ssim_blocked = sum(ssim_blocked_arr) / len(ssim_blocked_arr)
avg_mse_output = sum(mse_output_arr) / len(mse_output_arr)
avg_psnr_output = sum(psnr_output_arr) / len(psnr_output_arr)
avg_psnrb_output = sum(psnrb_output_arr) / len(psnrb_output_arr)
avg_ssim_output = sum(ssim_output_arr) / len(ssim_output_arr)

# print average performance values
print(f"\nAVERAGE PERFORMANCE: MSE {avg_mse_blocked:.6f} -> {avg_mse_output:.6f}  | PSNR {avg_psnr_blocked:.6f} -> {avg_psnr_output:.6f} | PSNR-B {avg_psnrb_blocked:.6f} -> {avg_psnrb_output:.6f} | SSIM {avg_ssim_blocked:.6f} -> {avg_ssim_output:.6f}")
