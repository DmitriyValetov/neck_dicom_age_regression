import cv2
import random
import numpy as np
import SimpleITK as sitk


import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class BeginDetectorDataset(Dataset):
    def __init__(self, df, pos_num, neg_mul, mode='test', size=1000, shape=(512, 512), expand=False):
        self.df = df
        self.mode= mode
        self.size = size
        self.shape = shape
        self.expand = expand
        self.pos_num = pos_num
        self.neg_mul = neg_mul
        self.neg_num = int(pos_num*neg_mul)
        self.batch_size = self.pos_num + self.neg_num
        self.norm = torch.nn.InstanceNorm2d(1)
        
    def __len__(self):
        return self.size

    
    def __getitem__(self, i):
        pos_batch = []
        neg_batch = []

        while len(pos_batch) + len(neg_batch) < self.batch_size:
            row = self.df.loc[random.randint(0, len(self.df)-1)]
            image = sitk.GetArrayFromImage(sitk.ReadImage(row.Name)).astype(np.float32)

            begin_up = row.Begin
            begin_down = row['End1vertebra '] if not isinstance(row['End1vertebra '], str) else 0 
            pos_indices = list(range(begin_down, begin_up+1))
            random.shuffle(pos_indices)

            neg_indices = list(range(0, begin_down)) + list(range(begin_up+1, image.shape[0]))
            random.shuffle(neg_indices)

            while len(pos_batch) < self.pos_num and len(pos_indices)>0:
                i = pos_indices.pop()
                tmp = cv2.resize(image[i], self.shape, interpolation=cv2.INTER_AREA)
                tmp = torch.tensor(tmp).unsqueeze(0)
                if self.mode == 'train':
                    pos_batch.append(TF.rotate(tmp, random.randint(1,270), fill=tmp.min().item()))
                else:
                    pos_batch.append(tmp)

            while len(neg_batch) < self.neg_num and len(neg_indices)>0:
                i = neg_indices.pop()
                tmp = cv2.resize(image[i], self.shape, interpolation=cv2.INTER_AREA)
                tmp = torch.tensor(tmp).unsqueeze(0)
                neg_batch.append(tmp)

        batch = pos_batch + neg_batch
        targets = [1]*len(pos_batch) + [0]*len(neg_batch)

        for_shuffle = list(zip(batch, targets))
        random.shuffle(for_shuffle)
        batch, targets = zip(*for_shuffle)

        batch = torch.stack(batch)
        batch = self.norm(batch)
        if self.expand:
            batch = batch.expand(batch.shape[0], 3, batch.shape[2], batch.shape[3])
        targets = torch.tensor(targets)
        return batch, targets
    

class BeginDetectorDataset_for_infer(Dataset):
    def __init__(self, file_path, shape=(512, 512), expand=False):
        self.shape = shape
        self.expand = expand
        self.file_path = file_path
        self.norm = torch.nn.InstanceNorm2d(1)
        self.image = sitk.GetArrayFromImage(sitk.ReadImage(self.file_path)).astype(np.float32)
        
        
    def __len__(self):
        return self.image.shape[0]

    
    def __getitem__(self, i):
        slide = torch.tensor(cv2.resize(self.image[i, ...], self.shape, interpolation=cv2.INTER_AREA)).unsqueeze(0).unsqueeze(0) # 
        slide = self.norm(slide)
        if self.expand:
            slide = slide.expand(1, 3, slide.shape[2], slide.shape[3])
        return slide, i