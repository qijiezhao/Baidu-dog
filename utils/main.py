import argparse
import os,sys,math,random
import torch,torchvision
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from IPython import embed
from data import get_training_set, get_testing_set,get_final_test_set
import argparse
from sklearn.metrics import confusion_matrix
import numpy as np
sys.path.append('../deepmodels')
import pretrainedmodels

def get_top1(label_sum,out_sum):
    label_sum=label_sum.numpy()
    out_sum=out_sum.numpy().argmax(1)
    assert len(label_sum)==len(out_sum)
    len_right=len(label_sum[label_sum==out_sum])
    return len_right/float(len(label_sum))


def exp_lr_scheduler(optimizer,epoch,init_lr=0.01,lr_decay_num0=1):
    lr=init_lr*(0.1**(epoch//lr_decay_num0))
    if epoch%lr_decay_num0==0:
        print 'lr is set to {}'.format(lr)
    # optimizer.param_groups[0]['lr']=lr*0.1
    # optimizer.param_groups[1]['lr']=lr
    for param_group in optimizer.param_groups:
        param_group['lr']=lr
    return optimizer

def data_parallel(module, input, device_ids, output_device=None):
    if not device_ids:
        return module(input)
    if output_device is None:
        output_device = device_ids[0]
    replicas = nn.parallel.replicate(module, device_ids)
    inputs = nn.parallel.scatter(input, device_ids)
    replicas = replicas[:len(inputs)]
    outputs = nn.parallel.parallel_apply(replicas, inputs)
    return nn.parallel.gather(outputs, output_device)

def fine_tune(model,drop_ratio=0.5):
    mod=[nn.Dropout(p=drop_ratio)]
    mod.append(nn.Linear(2048,134))
    new_fc=nn.Sequential(*mod)
    model.fc=new_fc
    return model

model_names = sorted(name for name in pretrainedmodels.__dict__
    if not name.startswith("__")
    and callable(pretrainedmodels.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch dogs Training')
parser.add_argument('--model_path',default='None',type=str)
parser.add_argument('--model_name', '-m', metavar='MODEL_NAME', default='fbresnet152',
                    #choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: fbresnet152)')
parser.add_argument('--parallel',default=True,type=bool)
parser.add_argument('--gpu_ids',default='012',type=str)
parser.add_argument('--num_workers',default=1,type=int)
parser.add_argument('--batch_size',default=20,type=int)
parser.add_argument('--base_lr',default=0.01,type=float)
parser.add_argument('--prefix',default='',type=str)
parser.add_argument('--image_source',default='raw',type=str)
parser.add_argument('--finetune',default=True,type=bool)
parser.add_argument('--freeze',default=False,type=bool)
parser.add_argument('--drop_ratio',default=0.5,type=float)
args = parser.parse_args()

model_name=args.model_name
if not model_name in model_names:
    is_model=torchvision.models.__dict__[model_name]()
    if not is_model:
        print 'No model matched! '
        exit()
model_path=args.model_path
parallel=args.parallel
num_workers=args.num_workers
batch_size=args.batch_size
base_lr=args.base_lr
prefix=args.prefix
image_source=args.image_source
finetune=args.finetune
drop_ratio=args.drop_ratio

finetune= 'imagenet' if finetune==True else False

gpu_ids=[int(_) for _ in args.gpu_ids]
training_size_dic={'resnet50':(3,224,224),
                   'resnet101':(3,224,224),
                   'resnet152':(3,224,224),
                   'inception_v3':(3,299,299)
                   # to be added
                   }
if model_path =='None' and model_name in model_names:
    model=pretrainedmodels.__dict__[model_name](num_classes=1000,pretrained=finetune)
    #embed()
    if 'resnext' in model_name:
        model=fine_tune(model,drop_ratio=drop_ratio)
    else:#not 'resnext'
        model.classif=nn.Linear(1536,134)
    training_size=model.input_size
    image_type=model.input_space
    mean,std=model.mean,model.std
    mean_std=(mean,std)
elif model_path=='None' and model_name not in model_names:
    model=torchvision.models.__dict__[model_name](pretrained=True)#all of the shallow models are setted as pretrained mode
    model=fine_tune(model,drop_ratio=drop_ratio)
    training_size=training_size_dic[model_name] if model_name in training_size_dic else (3,224,224)
    image_type='RGB'
    mean_std=((0.5,0.5,0.5),(0.5,0.5,0.5))
else:
    model=torch.load(model_path)
if torch.cuda.is_available():
    model=model.cuda()

train_set=get_training_set(training_size=training_size,image_type=image_type,mean_std=mean_std,image_source=image_source)
test_set=get_testing_set(training_size=training_size,image_type=image_type,mean_std=mean_std,image_source=image_source)
training_data_loader=DataLoader(dataset=train_set,num_workers=num_workers,batch_size=batch_size,shuffle=True)
testing_data_loader=DataLoader(dataset=test_set,num_workers=num_workers,batch_size=batch_size,shuffle=True)


loss_function=nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_function=loss_function.cuda(gpu_ids[0])
#embed()
freeze=args.freeze
#tune the different layers' learning rate: different model has different layer names
if freeze:
    optimizer=optim.SGD([#{'params':model.conv2d_7b.parameters()},
                         {'params':model.classif.parameters()}
                         ],lr=base_lr,momentum=0.9,weight_decay=0.0005)
else:
    optimizer=optim.SGD(model.parameters(),lr=base_lr,momentum=0.9,weight_decay=0.0005)
if parallel and model_path=='None':
        model=nn.DataParallel(model,device_ids=gpu_ids)
#embed()
def train(epoch):
    model.train()
    label_sum,out_sum=torch.zeros(batch_size),torch.zeros(batch_size)
    for iteration,(inputs,labels) in enumerate(training_data_loader,1):
        #embed()
        inputs,labels=Variable(inputs),Variable(labels)
        if torch.cuda.is_available():
            inputs=inputs.cuda(gpu_ids[0])
            labels=labels.cuda(gpu_ids[0])
        optimizer.zero_grad()
        out=model(inputs)
        loss=loss_function(out,labels)
        loss.backward()
        optimizer.step()
        if label_sum.sum()==0:
            label_sum=labels.cpu().data
            out_sum=out.cpu().data
        else:
            label_sum=torch.cat([label_sum,labels.cpu().data])
            out_sum=torch.cat([out_sum,out.cpu().data])
        if iteration%40==0:
            hit_one=get_top1(label_sum,out_sum)
            print '=====>Epoch[{}]({}/{}): loss:{:.4f}, hit_one_accuracy:{:.4f}'.format(epoch,iteration,len(training_data_loader),loss.data[0],hit_one)
            label_sum,out_sum=torch.zeros(batch_size),torch.zeros(batch_size)
        #else:
        #    print '===> Epoch[{}]({}/{}): loss:{:.4f}'.format(epoch,iteration,len(training_data_loader),loss.data[0])
        del inputs,labels

def get_top1_val(gt_labels,predictions):
    gt,pre=np.array(gt_labels),np.array(predictions)
    len_acc=len(gt[gt==pre])
    hit_acc=len_acc/float(len(gt))
    return hit_acc

def test(epoch):
    model.eval()
    predictions=[]
    gt_labels=[]

    for iteration,(inputs,labels) in enumerate(testing_data_loader,1):
        inputs,labels=Variable(inputs),Variable(labels)
        if torch.cuda.is_available():
            inputs=inputs.cuda(gpu_ids[0])
        pred=model(inputs)
        pred_inds=pred.cpu().data.numpy().argmax(1)
        predictions.extend(list(pred_inds))
        gt_labels.extend(list(labels.data.numpy()))
    hit1_acc=get_top1_val(gt_labels,predictions)
    print 'tested result: hit_1 accuracy={}'.format(hit1_acc)

def checkpoint(epoch):
    model_path=os.path.join('../','tmp_models','{}_model_epoch_{}.pth'.format(prefix,epoch))
    #embed()
    torch.save(model,model_path)
    print 'Checkpoint saved to {}'.format(model_path)

for epoch in range(0,10):
    optimizer=exp_lr_scheduler(optimizer,epoch,base_lr,8)

    train(epoch)
    #checkpoint(epoch)
    test(epoch)


'''final test: give the result'''

final_test_set=get_final_test_set(training_size=training_size,image_type=image_type,mean_std=mean_std,image_source=image_source)
final_test_data_loader=DataLoader(dataset=final_test_set,num_workers=1,batch_size=10,shuffle=False)
def get_test_names():
    path='../{}/test'.format(image_source)
    test_files=os.listdir(path)
    test_files.sort()
    test_files=[_.split('.')[0] for _ in test_files]
    return test_files

def final_test():
    model.eval()
    predictions=[]
    predictions_max=np.zeros([batch_size,134])
    for iteration,inputs in enumerate(final_test_data_loader,1):
        inputs=Variable(inputs)
        if torch.cuda.is_available():
            inputs=inputs.cuda(gpu_ids[0])
        pred=model(inputs).cpu().data.numpy()
        pred_inds=pred.argmax(1)
        predictions.extend(list(pred_inds))
        if iteration==0:
            predictions_max=pred
            continue
        predictions_max=np.vstack([predictions_max,pred])
        if iteration%50==0:
            print '=====> test done over {} images '.format(iteration*10)
    test_files=get_test_names()
    with open('../results/{}_{}_{}_{}_result.txt'.format(prefix,model_name,image_source,finetune),'wb')as fw:
        fw.write('\n'.join([str(_)+'\t'+_x for _,_x in zip(predictions,test_files)]))
    np.save('../results/{}_{}_{}_{}_result_mat.npy'.format(prefix,model_name,image_source,finetune),predictions_max)
    print 'saving the result done.'

final_test()

