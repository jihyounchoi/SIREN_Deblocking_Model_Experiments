import numpy as np
import torch
import torch.nn as nn 
import os
import json
import traceback
import time
import sys

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import *
from dataset_gray import *
from utils_model import *
from utils_image import *
from datetime import datetime

torch.autograd.set_detect_anomaly(True)

def main(params_json):
    activation = params_json['activation']
    residual = params_json['residual']
    edge_aware = params_json['edge_aware']
    optimizer_name = params_json['optimizer_function']
    gamma = params_json['gamma_optim']
    loss = params_json['loss']
    learning_rate = params_json['learning_rate']
    num_epoch = params_json['num_epoch']
    milestone = params_json['milestone']
    batchsize_train = params_json['batchsize_train']
    batchsize_val = params_json['batchsize_val']
    base_dir = params_json['base_dir']
    ckpt_dir_to_load_manually = params_json['ckpt_dir_to_load_manually']
    ckpt_dir_save = params_json['ckpt_dir_save']
    log_dir = params_json['log_dir']
    patch_size = params_json['patch_size']
    datadir_train = params_json['datadir_train']
    datadir_val = params_json['datadir_val']
    device = params_json['device']
    step_to_save = params_json['step_to_save']
    qf_candidates = params_json['QF_candidates']
    
    
    residual_text = "Res" if residual == True else ""
    coordinate_text = "Edge" if edge_aware == True else ""
    base_dir = os.path.join(base_dir, f"{activation}{residual_text}{coordinate_text}")


    parameter_printer(params_json=params_json)
    set_dir(params_json=params_json, base_dir=base_dir)
    
    original_out = sys.stdout
    with open(f"{base_dir}/PERFORMANCE.txt", "w") as f:
        sys.stdout = f

        # define train and test dataset objects
        dataset_train = DatasetJPEG(edge_aware = edge_aware, patch_size=patch_size, dataset_path=datadir_train, phase='train', qf_candidates=qf_candidates)
        dataset_val = DatasetJPEG(edge_aware = edge_aware, patch_size=patch_size, dataset_path=datadir_val, phase='val', qf_candidates=qf_candidates)

        # define train and test loader objects
        train_loader = DataLoader(dataset_train, batch_size=batchsize_train, shuffle=True, num_workers=1, drop_last=True)
        val_loader = DataLoader(dataset_val, batch_size=batchsize_val, shuffle=False, num_workers=1, drop_last=False)
        
        # create network
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f'Training on device: {device}')
        
        model = Network(edge_aware = edge_aware, activation = activation, residual = residual)
        model.to(device)

        # define loss function
        if loss == 'L1':
            loss_function = nn.L1Loss()
        elif loss == 'L2':
            loss_function = nn.MSELoss()
        else:
            print(f'{loss} : WRONG LOSS FUNCTION!')
            exit

        # define optimizer
        if optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
        elif optimizer_name == 'ADAM':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        else:
            print(f'{optimizer_name} : WRONG OPTIMIZER NAME!')
            exit
        
        # define the scheduler to decrease the learning rate by a factor of 0.1
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestone, gamma=gamma)

        num_data_train = len(dataset_train)
        num_data_val = len(dataset_val)

        num_batch_train = np.floor(num_data_train / batchsize_train) # np.ceil when Not apply drop-last to trainloader
        num_batch_val = np.ceil(num_data_val / batchsize_val)

        # set up other functions
        fn_tonumpy = lambda x: x.to('cpu').detach().numpy()

        # set up SummaryWriter for Tensorboard
        writer_train = SummaryWriter(log_dir=os.path.join(base_dir, log_dir, 'train'))
        writer_val = SummaryWriter(log_dir=os.path.join(base_dir, log_dir, 'val'))

        # train the network
        st_epoch = 0
        
        # 학습한 모델이 있을 경우 모델 로드하기
        if ckpt_dir_to_load_manually == "":
            model, optimizer, st_epoch = load_latest(ckpt_dir=os.path.join(base_dir, ckpt_dir_save), net=model, optim=optimizer)
        else:
            model, optimizer, st_epoch = load_manually(ckpt_dir=ckpt_dir_to_load_manually, net=model, optim=optimizer)
        
        print(f'Learning Starts on Epoch : {st_epoch}')

        
        for epoch in range(st_epoch + 1, num_epoch + 1):
            
            model.train()
            loss_arr = []
            
            print(f"\nCURRENT LEARNING RATE : {optimizer.param_groups[0]['lr']}\n")
            
            start_epoch = datetime.now()
                    
            for batch, data in enumerate(train_loader, 1):
                
                # forward pass
                blocked = data['L'].to(device)
                ground_truth = data['H'].to(device)
                
                optimizer.zero_grad()
                output = model(blocked)[0]
                
                # backward pass
                loss = loss_function(output, ground_truth)

                loss.backward()
                optimizer.step()

                loss_arr += [loss.item()]
                
                # In Training, only MSE Calculated
                # log를 출력하는 경우에는 batch 단위의 loss를 print 하지만, tensorboard에는 epoch 단위의 avg loss만을 push한다
                # print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | AVG LOSS OF THIS BATCH %.6f " %
                #     (epoch, num_epoch, batch, num_batch_train, loss.item()))
                
                # Tensorboard 저장하기
                # blocked = fn_tonumpy(blocked)
                # ground_truth = fn_tonumpy(fn_denorm(ground_truth, mean=0.5, std=0.5))
                # output = fn_tonumpy(fn_class(output))
                
                # blocked = fn_tonumpy(blocked)
                # ground_truth = fn_tonumpy(ground_truth)
                # output = fn_tonumpy(output)

                # writer_train.add_image('label', ground_truth, num_batch_train * (epoch - 1) + batch, dataformats='NCHW')
                # writer_train.add_image('input', blocked, num_batch_train * (epoch - 1) + batch, dataformats='NCHW')
                # writer_train.add_image('output', output, num_batch_train * (epoch - 1) + batch, dataformats='NCHW')
                
            scheduler.step()

            writer_train.add_scalar('TRAIN_LOSS_EPOCH', np.mean(loss_arr), epoch)
            print(f'LOSS OF THIS EPOCH : {np.mean(loss_arr)}')
            
            start_val = datetime.now()
            print(f'Elapsed Time To Train Epoch {epoch} : {start_val - start_epoch}\n')
            
            
            loss_arr = []
            psnr_arr = []
            psnrb_arr = []
            ssim_arr = []

            with torch.no_grad():
                
                model.eval()
                
                for batch, data in enumerate(val_loader, start=1):

                    # forward pass
                    blocked = data['L'].to(device)
                    ground_truth = data['H'].to(device)

                    output = model(blocked)[0]

                    # 손실함수 계산
                    
                    output_cpu = output.detach().cpu()
                    ground_truth_cpu = ground_truth.detach().cpu()
                    
                    loss = loss_function(output, ground_truth)

                    loss_arr += [loss.item()]
                    psnr_arr += calculate_psnr_batch(output_cpu, ground_truth_cpu)
                    psnrb_arr += calculate_psnrb_batch(output_cpu, ground_truth_cpu)
                    ssim_arr += calculate_ssim_batch(output_cpu, ground_truth_cpu)
                    
                    # In Validation, MSE, PSNR, PSNR-B, SSIM Is Calculated
                    # print last value of each array : because each element in array represents avg value of batch
                    
                blocked = fn_tonumpy(blocked)
                ground_truth = fn_tonumpy(ground_truth)
                output = fn_tonumpy(output)
                
                
                if epoch % 100 == 0: # save validation result in 100 epochs
                    writer_val.add_image('label', ground_truth, num_batch_val * (epoch - 1) + batch, dataformats='NCHW')
                    
                    # only first channel of input is saved because original input channel with edge-direction is 2, which is neither 3, or 1
                    writer_val.add_image('input', blocked[:,0:1,:,:], num_batch_val * (epoch - 1) + batch, dataformats='NCHW')
                    writer_val.add_image('output', output, num_batch_val * (epoch - 1) + batch, dataformats='NCHW')
        
            mean = lambda arr: sum(arr) / len(arr)
            
            writer_val.add_scalar('VAL_LOSS_EPOCH', mean(loss_arr), epoch)
            writer_val.add_scalar('VAL_PSNR_EPOCH', mean(psnr_arr), epoch)
            writer_val.add_scalar('VAL_PSNRB_EPOCH', mean(psnrb_arr), epoch)
            writer_val.add_scalar('VAL_SSIM_EPOCH', mean(ssim_arr), epoch)
            
            print("VALIDATION PERFORMANCE OF THIS EPOCH: LOSS %.6f | PSNR %.6f | PSNR-B %.6f | SSIM %.6f"%
                (mean(loss_arr), mean(psnr_arr), mean(psnrb_arr), mean(ssim_arr)))
            
            # epoch step_to_save 마다 모델 저장하기
            if epoch % step_to_save == 0:
                save(ckpt_dir=os.path.join(base_dir, ckpt_dir_save), net=model, optim=optimizer, epoch=epoch)
                    
            end_val = datetime.now()
            print(f'Elapsed Time To Validate Epoch {epoch} : {end_val - start_val}')
            
            writer_train.close()
            writer_val.close()
            
    sys.stdout = original_out
                

if __name__ == '__main__':
    
    with open('configurations.json') as f:
        config = json.load(f)
        
    
        for i, params_json in enumerate(config['param_settings']):
            
            print('#####################################################')
            print(f'Start {i + 1}th setting at {datetime.now()}')

            try: 
                main(params_json=params_json)

                print(f'Finish {i + 1}th setting at {datetime.now()}')
                print('#####################################################\n\n\n')

                msg_subject = f'Setting {i + 1} finished successfully'
                msg_body = write_successed_msg(params_json=params_json)
                send_email(subject=msg_subject, body=msg_body)


            except Exception as e:
                
                error = traceback.format_exc()
                
                print(f'Terminated {i + 1}th setting with error: {error}')
                print('#####################################################\n\n\n')

                msg_subject = f'Setting {i + 1} terminated by error'
                msg_body = write_error_msg(params_json=params_json, error=error)
                send_email(subject=msg_subject, body=msg_body)
            
        

        