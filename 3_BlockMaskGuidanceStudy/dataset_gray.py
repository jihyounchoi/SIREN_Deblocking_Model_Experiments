import random
import torch
import numpy as np
import torch.utils.data as data
import utils_image as util
import torch.functional as F
import cv2


class DatasetJPEG(data.Dataset):

    def __init__(self, edge_aware, n_channels=1, patch_size=32, dataset_path = None, phase = 'train', qf_candidates : list = [10*i for i in range(1, 10)]):
        super(DatasetJPEG, self).__init__()
        self.n_channels = n_channels if n_channels else 1
        self.patch_size = patch_size if patch_size else 32

        # -------------------------------------
        # get the path of H, return None if input is None
        # -------------------------------------
        self.edge_aware = edge_aware
        self.paths_H = util.get_image_paths(dataset_path)
        self.phase = phase
        self.qf_candidates = qf_candidates

    def __getitem__(self, index):
        # -------------------------------------
        # get H image
        # -------------------------------------
        
        patch_H = np.zeros((self.patch_size, self.patch_size), dtype=np.int8)
        patch_L = np.zeros((self.patch_size, self.patch_size), dtype=np.int8)
        
        H_path = self.paths_H[index]
        L_path = H_path
        
        # read image in single channel
        image_H = util.imread_uint(H_path, self.n_channels)
        H, W = image_H.shape[:2]
        
        if self.phase == 'train':
            """
            # --------------------------------
            # get L/H patch pairs
            # --------------------------------
            """

            # ---------------------------------
            # randomly crop the patch
            # ---------------------------------
            rnd_h = random.randint(1, (H - self.patch_size - 10))
            rnd_w = random.randint(1, (W - self.patch_size - 10))
            
            patch_H = image_H[rnd_h : rnd_h + self.patch_size, rnd_w : rnd_w + self.patch_size, :]

            # ---------------------------------
            # augmentation - flip, rotate
            # ---------------------------------
            mode = random.randint(0, 7)
            patch_H = util.augment_img(patch_H, mode=mode) # apply augmentation randomly

            # ---------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # ---------------------------------
            # img_H = util.uint2tensor3(patch_H)
            # img_L = img_H.clone()

            patch_L = patch_H.copy()

            # --------------------------------- 
            # select Quality Factor (QF)
            # ---------------------------------
            
            quality_factor = self.qf_candidates[random.randrange(len(self.qf_candidates))]
            
            # Generate encoded-decoded images
            result, encimg = cv2.imencode('.jpg', patch_L, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
            patch_L = cv2.imdecode(encimg, 0)
            
            # patch_H = torch.from_numpy(np.ascontiguousarray(np.squeeze(patch_H, axis = 2))).float().div(255.)
            # patch_L = torch.from_numpy(np.ascontiguousarray(patch_L)).float().div(255.)
            
            patch_H = torch.from_numpy(np.ascontiguousarray(patch_H)).permute(2, 0, 1).float().div(255.)
            patch_L = torch.from_numpy(np.ascontiguousarray(patch_L.reshape(self.patch_size, self.patch_size, 1))).permute(2, 0, 1).float().div(255.)
            
            if self.edge_aware == True:
                patch_L = util.add_block_edge(patch_L)
            
            return {'L': patch_L, 'H': patch_H, 'qf': quality_factor, 'L_path': L_path, 'H_path' : H_path}

        else:
            """
            # --------------------------------
            # get L/H image pairs
            # --------------------------------
            """
            
            assert len(self.qf_candidates) == 1
            
            image_L = image_H.copy()
            
            quality_factor = self.qf_candidates[0]

            result, encimg = cv2.imencode('.jpg', image_L, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
            image_L = cv2.imdecode(encimg, 0)
            
            # image_H = torch.from_numpy(np.ascontiguousarray(np.squeeze(image_H, axis = 2))).float().div(255.)
            # image_L = torch.from_numpy(np.ascontiguousarray(image_L)).float().div(255.)

            image_H = torch.from_numpy(np.ascontiguousarray(image_H)).permute(2, 0, 1).float().div(255.)
            image_L = torch.from_numpy(np.ascontiguousarray(image_L.reshape(H, W, 1))).permute(2, 0, 1).float().div(255.)
            
            if self.edge_aware == True:
                image_L = util.add_block_edge(image_L)
            
            return {'L': image_L, 'H': image_H, 'qf': quality_factor, 'L_path': L_path, 'H_path' : H_path}
    
    def __len__(self):
        return len(self.paths_H)
    
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dataset_train = DatasetJPEG(n_channels = 1, patch_size = 64, dataset_path='trainset/small_DIV2K_train_cropped', phase = 'train', qf_candidates=[30])
    dataset_test = DatasetJPEG(n_channels = 1, patch_size = 0, dataset_path='trainset/small_DIV2K_val_cropped', phase = 'val', qf_candidates=[30])
    
    # 데이터 로더 잘 구현되었는지 확인
    data_train = dataset_train.__getitem__(0)
    data_test = dataset_test.__getitem__(0)
    
    blocked_train = data_train['L'].numpy().squeeze()
    ground_truth_train = data_train['H'].numpy().squeeze()
    
    blocked_test = data_test['L'].numpy().squeeze()
    ground_truth_test = data_test['H'].numpy().squeeze()
    
    test_height = blocked_test.shape[0]
    test_width = blocked_test.shape[1]

    # 불러온 이미지 시각화 - train
    plt.subplot(221)
    plt.imshow(ground_truth_train, cmap='gray', extent = [0, 32, 32, 0])
    plt.title('Ground Truth')
    
    plt.subplot(222)
    plt.imshow(blocked_train, cmap='gray', extent = [0, 32, 32, 0])
    plt.title('Blocked')
    
    # 불러온 이미지 시각화 - test
    # warning : Although test images' size are not uniform,
    # just forcely convert them into 32 x 32 shape.
    # so displayed images may distorted
    
    plt.subplot(223)
    plt.imshow(ground_truth_test, cmap='gray', extent = [0, test_width, test_height, 0])
    plt.title('Ground Truth')
    
    plt.subplot(224)
    plt.imshow(blocked_test, cmap='gray', extent = [0, test_width, test_height, 0])
    plt.title('Blocked')

    plt.show()