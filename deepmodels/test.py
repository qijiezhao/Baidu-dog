from PIL import Image
import torch
import torchvision.transforms as transforms

import sys
sys.path.append('yourdir/pretrained-models.pytorch') # if needed
import pretrainedmodels
from IPython import embed
#Load Model
#model_name = 'inceptionresnetv4' #fbresnet152
model_name=sys.argv[1]
model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
model.eval()

#Load One Input Image
path_img = 'data/cat.jpg'
with open(path_img, 'rb') as f:
    with Image.open(f) as img:
        input_data = img.convert(model.input_space)

tf = transforms.Compose([
    transforms.Scale(int(round(max(model.input_size)*1.143))),
    transforms.CenterCrop(max(model.input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=model.mean, std=model.std)
])

input_data = tf(input_data)          # 3x400x225 -> 3x299x299
input_data = input_data.unsqueeze(0) # 3x299x299 -> 1x3x299x299
input = torch.autograd.Variable(input_data)

#Load Imagenet Synsets
with open('data/imagenet_synsets.txt', 'r') as f:
    synsets = f.readlines()

#len(synsets)==1001
#sysnets[0] == background
synsets = [x.strip() for x in synsets]
splits = [line.split(' ') for line in synsets]
key_to_classname = {spl[0]:' '.join(spl[1:]) for spl in splits}

with open('data/imagenet_classes.txt', 'r') as f:
    class_id_to_key = f.readlines()

class_id_to_key = [x.strip() for x in class_id_to_key]

#Make predictions
output = model(input) # size(1, 1000)
max, argmax = output.data.squeeze().max(0)
class_id = argmax[0]
class_key = class_id_to_key[class_id]
classname = key_to_classname[class_key]

print(path_img, 'is a', classname) 
