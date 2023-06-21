# --------------------------------
# test phase
# --------------------------------

import numpy as np
import torch
import torch.nn as nn
import os
import sys
import cv2
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from PIL import Image

from models import *
from dataset_gray import *
from utils_model import *
from utils_image import *

def save_histogram_of_results(results, dismiss_zeros = False, save_path='images_temp', bins=511, dpi=500):
    plt.draw()
    plt.style.use('ggplot')

    fig, axes = plt.subplots(1, len(results), sharex=False, sharey=False, figsize=(40, 5))

    for i, checkpoint in enumerate(results.keys()):
        # Extract the tensor values and flatten them
        values = results[checkpoint].numpy().flatten()
        
        if dismiss_zeros == True:
            indices = np.nonzero(np.abs(values) > 1e-6)[0]
            values = values[indices]

        # Plot the histogram
        axes[i].hist(values, bins=bins, density=True)
        axes[i].set_title(f'Checkpoint {checkpoint}')

    # Save the figure to the desired path
    fig.savefig(f"{save_path}/histogram_{dataset_test.paths_H[order].split('/')[-1].split('.')[0]}.png", dpi=dpi)
    plt.close()


datadir_test = '../Dataset/testset/LIVE1_gray'
qf_candidates = [30]
batchsize_test = 1
base_dir = './result_test'
ckpt_dir = 'model_zoo/SSSNetwork_200.pth'
device = 'cpu'

# model = PlainSSSNetwork(omega = 1)
# model = RRRNetwork()
# model = RRSNetwork(omega = 1)
# model = CustomNetwork('R,S,R/R,S,R/R,S,R/R,S,R/R,S,R/R,S,R', '1/1/1/1/1/1', 'F/T/T/T/T/T')
# model = CustomNetwork('R,S,S/R,S,S/R,S,S/R,S,S/R,S,S/R,S,S', '1/1/1/1/1/1', 'F/T/T/T/T/T')
# model = CustomNetwork('S,S,R/S,S,R/S,S,R/S,S,R/S,S,R/S,S,R', '1/1/1/1/1/1', 'F/T/T/T/T/T')
# model = CustomNetwork('S,R,S/S,R,S/S,R,S/S,R,S/S,R,S/S,R,S', '1/1/1/1/1/1', 'F/T/T/T/T/T')
# model = CustomNetwork('R,S,S/R,S,S/R,S,S/R,S,S/R,S,S/R,S,S', '1/1/1/1/1/1', 'F/T/T/T/T/T', tracking_point_list=['first', '0', '1', '2', '3', '4', '5', 'last', 'output'])
# model = CustomNetwork('S,R,R/S,R,R/S,R,R/S,R,R/S,R,R/S,R,R', '1/1/1/1/1/1', 'F/T/T/T/T/T')
# model = ReLUNetwork(tracking_point_list=['first', '0', '1', '2', '3', '4', '5', 'last', 'output'])
model = SSSNetwork(omega = 1)
# model = PlainRRRNetwork(tracking_point_list=['first', '0', '1', '2', '3', '4', '5', 'last', 'output'])
# model = RRSNetwork(omega = 10, tracking_point_list=['first', '0', '1', '2', '3', '4', '5', 'last', 'output'])
# model = CustomNetwork('S,S,S/S,S,S/S,S,S/S,S,S/S,S,S/S,S,S', '1/1/1/1/1/1', 'F/T/T/T/T/T', tracking_point_list=[''])
# model = CustomNetwork('S,S,S/S,S,S/S,S,S/S,S,S/S,S,S/S,S,S', '1/1/1/1/1/1', 'F/F/F/F/F/F', tracking_point_list=[''])
    
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

histogram_dir = os.path.join(base_dir, 'histograms')
if not os.path.exists(histogram_dir):
    os.makedirs(histogram_dir)

if not os.path.exists(ckpt_dir):
    raise ValueError(f"Model file '{ckpt_dir}' does not exist.")

dict_model = torch.load(ckpt_dir, map_location=device)
model.load_state_dict(dict_model['net'])

device = torch.device(device if torch.cuda.is_available() else 'cpu')

# save the current value of sys.stdout
original_stdout = sys.stdout

with open(f"{base_dir}/PERFORMANCE.txt", "w") as f:
    sys.stdout = f

    print('#' * 50)
    print(f"{'INITIALIZE TEST ENVIRONMENT':^50s}")
    print('#' * 50)

    print(f'Testing on device: {device}')
    print(f"datadir_test: {datadir_test}")
    print(f"qf_candidates: {qf_candidates}")
    print(f"batchsize_test: {batchsize_test}")
    print(f"base_dir: {base_dir}")
    print(f"ckpt_dir: {ckpt_dir}")
    print(f"model_: {model}")
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

            output, result = model(blocked)
            
            # save_histogram_of_results(results=result, save_path=histogram_dir)

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
            
            print(f"DEBLOCKING PERFORMANCE OF {dataset_test.paths_H[order].split('/')[-1]} :\t"
                  f" MSE {mse_blocked:.6f} -> {mse_output:.6f} |"
                  f" PSNR {psnr_blocked:.6f} -> {psnr_output:.6f} |"
                  f" PSNR-B {psnrb_blocked:.6f} -> {psnrb_output:.6f} |"
                  f" SSIM {ssim_blocked:.6f} -> {ssim_output:.6f}")
        
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
    print(f"\nAVERAGE PERFORMANCE:\t\t\t\t\t"
          f" MSE {avg_mse_blocked:.6f} -> {avg_mse_output:.6f} "
          f"| PSNR {avg_psnr_blocked:.6f} -> {avg_psnr_output:.6f} "
          f"| PSNR-B {avg_psnrb_blocked:.6f} -> {avg_psnrb_output:.6f} "
          f"| SSIM {avg_ssim_blocked:.6f} -> {avg_ssim_output:.6f}" )
    
    sys.stdout = original_stdout
    
print('Test Finished')