import os,sys
import numpy as np
from dataset import DatasetFromFolder

def get_training_set(training_size=224,image_type='RGB',mean_std=((0.5,0.5,0.5),(0.5,0.5,0.5)),image_source='train'):
    return DatasetFromFolder('train',training_size,image_type,mean_std,image_source=image_source)

def get_testing_set(training_size=224,image_type='RGB',mean_std=((0.5,0.5,0.5),(0.5,0.5,0.5)),image_source='train'):
    return DatasetFromFolder('val',training_size,image_type,mean_std,image_source=image_source)

def get_final_test_set(training_size=224,image_type='RGB',mean_std=((0.5,0.5,0.5),(0.5,0.5,0.5)),image_source='train'):
    return DatasetFromFolder('test',training_size,image_type,mean_std,image_source=image_source)