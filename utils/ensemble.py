import os,sys,numpy as np
import argparse,time

parser = argparse.ArgumentParser(description='employ the ensemble')
parser.add_argument('--files',default='',type=str)
parser.add_argument('--s',default=0,type=int)
parser.add_argument('--weights',default='0',type=str)
parser.add_argument('--out',default='ensemble',type=str)
args=parser.parse_args()
s=args.s
weights=args.weights

root='../results/'
files=args.files.split(',')
len_files=len(files)
weights=[float(_) for _ in weights.split(',')] if weights!='0' else [1.0]*len_files
assert len_files==len(weights),'weights not matched!'
final_result=np.zeros([len_files,10593,134])
for i in range(len_files):
    tmp=np.load(os.path.join(root,files[i]+'.npy'))
    if len(tmp)>10593:
        gap=len(tmp)-10593
        tmp=tmp[gap:]
    final_result[i]=tmp
#final_result=final_result.mean(0).reshape(10593,134)
weights=np.array(weights).reshape(1,len_files)
final_result=final_result.reshape(len_files,-1)
final_result=np.dot(weights,final_result).reshape(10593,134)
final_result=final_result.argmax(1).reshape(10593)
exa_file='../results/ensemble_1.txt'
filenames=[]
for line in open(exa_file,'r').readlines():
    filenames.append(line.strip().split('\t')[1])
with open('../results/'+args.out+'-'.join(time.ctime().split(' ')[:2])+'.txt','wb')as fw:
    fw.write('\n'.join(str(_)+'\t'+_x for _,_x in zip(final_result,filenames)))
