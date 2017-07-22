import torch.utils.data as data
import os,sys
import numpy as np
from IPython import embed
import random
from PIL import Image
import torch
import torchvision.transforms as transforms


def get_images_labels(path):
    images,labels=[],[]
    with open(path,'r')as fp:
        for line in fp.readlines():
            images.append(line.strip().split(' ')[0])
            labels.append(int(line.strip().split(' ')[1]))
    return images,labels

def get_images_test(path):
    images=[]
    item='test'
    test_files=os.listdir(os.path.join(path,item))
    test_files.sort()
    for file in test_files:
        images.append(os.path.join(item,file))
    return images

class DatasetFromFolder(data.Dataset):
    def __init__(self,mode,training_size=224,image_type='RGB',mean_std=(128,1),image_source='train'):
        super(DatasetFromFolder,self).__init__()
        self.training_size=training_size
        self.image_type=image_type
        self.image_source=image_source
        self.root='../'
        self.mode=mode
        self.mean,self.std=mean_std
        if self.mode!='test':
            self.images,self.labels=get_images_labels(os.path.join(self.root,'metadata','{}_data_2.txt'.format(mode)))
        else:
            self.images=get_images_test(os.path.join(self.root,self.image_source))
        self.num_class=134
        self.tf=self.get_tf()

    def __getitem__(self, index):
        image_name=self.images[index]
        img=Image.open(os.path.join(self.root,self.image_source,image_name))
        if self.image_source!='raw'or self.mode!='train':
            img=img.resize([max(self.training_size),max(self.training_size)])
        img=img.convert(self.image_type)
        img=self.transform(img)
        #embed()
        if np.random.randint(2)%2==1:
            img[:,:,:]=torch.from_numpy(img.numpy()[:,:,::-1].copy())
        if self.mode!='test':
            labels=self.labels[index]
            return img,labels
        else:
            return img

    def __len__(self):
        return len(self.images)

    def get_tf(self):
        if self.image_source=='raw'and self.mode=='train':
            tf=transforms.Compose([
                transforms.Scale(int(round(max(self.training_size))* 1.2)),
                transforms.RandomCrop(int(max(self.training_size))),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean,
                                 std=self.std)
            ])
        else:
            tf = transforms.Compose([
            #transforms.Scale(int(round(max(self.training_size)))),
            #transforms.CenterCrop(max(self.training_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean,
                                 std=self.std)
            ])
        return tf

    def transform(self,img):
        input_data=img.convert(self.image_type)
        input_data=self.tf(input_data)
        return input_data
