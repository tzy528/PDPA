# generate attack data
import argparse
from ast import parse
import os
import time
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import models.gaussian_diffusion as gd
from models.DNN import DNN
import evaluate_utils
import data_utils
from copy import deepcopy
import random
import re
import pandas

random_seed = 1
torch.manual_seed(random_seed) # cpu
torch.cuda.manual_seed(random_seed) # gpu
np.random.seed(random_seed) # numpy
random.seed(random_seed) # random and transforms
torch.backends.cudnn.deterministic=True # cudnn
def worker_init_fn(worker_id):
    np.random.seed(random_seed + worker_id)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='qos/', help='choose the dataset')
parser.add_argument('--data_path', type=str, default='../datasets/', help='load data path')
parser.add_argument('--batch_size', type=int, default=400)
parser.add_argument('--topN', type=str, default='[10, 20, 50, 100]')
parser.add_argument('--tst_w_val', action='store_true', help='test with validation')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--gpu', type=str, default='0', help='gpu card ID')
parser.add_argument('--log_name', type=str, default='log', help='the log name')
parser.add_argument('--save_path', type=str, default='./saved_models/', help='save model path')
# params for diffusion
parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
parser.add_argument('--steps', type=int, default=10, help='diffusion steps')
parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
parser.add_argument('--noise_scale', type=float, default=0.1, help='noise scale for noise generating')
parser.add_argument('--noise_min', type=float, default=0.0001, help='noise lower bound for noise generating')
parser.add_argument('--noise_max', type=float, default=0.02, help='noise upper bound for noise generating')
parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
parser.add_argument('--sampling_steps', type=int, default=10, help='steps of the forward process during inference')

args = parser.parse_args()

print("args:", args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda:0" if args.cuda else "cpu")

print("Starting time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

### DATA LOAD ###
train_path = args.data_path +args.dataset + 'train_all1.txt'
valid_path = args.data_path +args.dataset + 'test_list.txt'
test_path = args.data_path +args.dataset + 'train_list.txt'


train_data, valid_y_data, test_y_data, n_user, n_item = data_utils.data_load(train_path, valid_path, test_path)
train_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.A))           #A:[user,[ item]]
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)
print('train:',len(train_dataset))
test_dataset = data_utils.DataDiffusion(torch.FloatTensor(test_y_data.A))           #A:[user,[ item]]
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
print("test:",len(test_dataset))
# if args.tst_w_val:
#     tv_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.A) + torch.FloatTensor(valid_y_data.A))
#     test_twv_loader = DataLoader(tv_dataset, batch_size=args.batch_size, shuffle=False)
# mask_tv = train_data + valid_y_data

print('data ready.')

rting=1
#### load model
if rting:
    params_filename='40steps100_0.pth'                                                  #40steps100_0.pth
    model_path = args.save_path+ 'data/' +params_filename
    args.sampling_noise = True
else:
    params_filename='_400generate_steps20_50_0.1_0.0001_0.02.pth'                                           #20/50    5/35   10/30
    model_path = args.save_path + 'ws/' + params_filename
model = torch.load(model_path)
print("models ready.")
matchstep = re.search(r'steps(\d+)_', params_filename)
steps11 = int(matchstep.group(1))
# matchstep = re.search(r'epoch(\d+)_lr', params_filename)
# modelepoch = str(matchstep.group(1))
args.steps = steps11
args.sampling_steps = steps11
### CREATE DIFFUISON ###
if args.mean_type == 'x0':
    mean_type = gd.ModelMeanType.START_X
elif args.mean_type == 'eps':
    mean_type = gd.ModelMeanType.EPSILON
else:
    raise ValueError("Unimplemented mean type %s" % args.mean_type)

diffusion = gd.GaussianDiffusion(mean_type, args.noise_schedule, \
        args.noise_scale, args.noise_min, args.noise_max, args.steps, device).to(device)
print("diffusion ready.")
scale=1
def ungetFile(data_loader):
    model.eval()
    attack_items = []
    savelist = []

    temp = 339
    count = -1
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            batch = batch.to(device)
            prediction = diffusion.getAttack(model, batch, args.sampling_steps, args.sampling_noise)
            attack_items.extend(prediction)
    selectuser=[i for i in range(len(test_dataset))]
    # selected_indices = random.sample(range(len(selectuser)), scale*34)
    attfile = pandas.read_csv('./saved_models/att_item_10.txt', names=['user'])
    attfile = attfile['user'].tolist()
    tempdict={}
    targercou=[0 for _ in range(339)]
    for j in range(len(test_dataset)):
        for i in range(5825):
           if i in attfile and test_dataset[j][i].item()!=0 :
                  targercou[j]+=1
    selected=sorted(range(len(targercou)), key=lambda i: targercou[i], reverse=True)[:34*scale]
    print(selected)
    for i in selected:
        tempdict[i]=temp
        temp+=1
    for i in selected:
        for j in range(5825):
            if test_dataset[i][j].item()!=0 or j in attfile:
                if j in attfile:
                    savelist.append([int(tempdict[i]),int(j),15])
                else:
                    round_value = round(attack_items[i][j].item(), 3)
                    savelist.append([int(tempdict[i]),int(j),abs(round_value)])


    savelist=pandas.DataFrame(savelist)
    savelist.to_csv(f'./saved_models/intensity/unsein10sc10.txt', sep='\t', header=None, index=None)
    print("done.")
def getFile(data_loader):
    model.eval()
    attack_items = []
    sid = []
    cout=0
    temp=0
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            batch = batch.to(device)
            prediction = diffusion.getAttack(model, batch, args.sampling_steps, args.sampling_noise)
            attack_items.extend(prediction)
    print(len(attack_items))
    dir_path = os.path.join(args.save_path, 'xiaorong')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    step = '20'
    file_name = f"wsuncpin10sc10.txt"
    full_path = os.path.join(dir_path, file_name)
    usermax = 338
    alpha = 1  # 控制左侧的形状
    beta = 2  # 控制右侧的形状
    attack_items_dict = {}
    userdict = {}
    maskuser = pandas.read_csv(f'./saved_models/xiaorong/wsuncpin10sc10_1.txt', sep='\t', header=None, names=['user', 'item','rt'], dtype={'user': int, 'item': int, 'rt': float})
    attuser=maskuser['user'].tolist()
    attdict={}
    for uid,iid,rt in maskuser.values:
        if uid not in attdict:
            attdict[uid] = []
        attdict[uid].append(iid)
    attfile=pandas.read_csv('./saved_models/att_item_10.txt',names=['user'])
    attfile=attfile['user'].tolist()
    print(attfile)
    savelist = []
    for i in range(len(attack_items)):
        if i in attuser:
            if i==0:
                cout = 1
            if temp < i:
                cout += 1
            for iid in range(5825):
                round_value = round(attack_items[int(i)][int(iid)].item(), 3)
                if iid in attfile:
                    savelist.append([int(usermax+cout),int(iid),15])
                elif iid in attdict[i]:
                    if round_value >= -0.002 and round_value <= 0.003:
                        round_value = -1
                    savelist.append([int(usermax+cout),int(iid),abs(round_value)])
            temp=i
    # for i in range(len(attack_items)):
    #     for j in range(5825):
    #         rounded_value = round(attack_items[i][j].item(),3)
    #         savelist.append([int(usermax+i),int(j),rounded_value])

    savelist1 = pandas.DataFrame(savelist)
    savelist1.to_csv(full_path,sep='\t',header=None,index=False,)



    print('generate done')
def getFileAll1(data_loader):
    model.eval()
    attack_items = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            batch = batch.to(device)
            prediction = diffusion.getAttack(model, batch, args.sampling_steps, args.sampling_noise)
            attack_items.extend(prediction)
    print(len(attack_items))
    usermask=pandas.read_csv('./saved_models/att_item_20.txt',  header=None,names=['user'])
    usermask = usermask['user'].tolist()
    print(usermask)

    uidict={}
    # attdict={}
    # for i in range(len(attack_items)):
    #     for j in range(5825):
    #         round_value=round(attack_items[i][j].item(),3)
    #         if round_value >= 0.66 :
    #         #     if i not in uidict.keys():
    #         #         uidict[i]={}
    #         #     uidict[i][j]=round_value
    #         # if round_value >= 0.3:
    #             if i not in attdict.keys():
    #                 attdict[i]=[]
    #             attdict[i].append(j)
    #
    # counter=[0 for _ in range(339)]
    # for i in attdict:
    #     for j in usermask:
    #         for k in attdict[i]:
    #             if j == k:
    #                 counter[i]+=1
    # print(counter)
    # selected=sorted(range(len(counter)), key=lambda i: counter[i], reverse=True)[:34*scale]
    filler1 = []
    selected = random.sample(range(339), 1 * 34)
    print(selected)
    for i in selected:
        for j in range(5825):
            if j in usermask or attack_items[i][j].item() >= 0.675:
                filler1.append([int(i),int(j),1])

    # filler1=[]
    # for i in attdict:
    #     if i in selected:
    #         for j in range(5825):
    #             if j in attdict[i] or j in usermask:
    #                 filler1.append([int(i),int(j),1])

    print(len(filler1))
    filler1=pandas.DataFrame(filler1)
    filler1.to_csv(f'./saved_models/xiaorong/wsuncpin10sc10_1.txt',sep='\t',header=None,index=False)
    # data = []
    # for i in attdict:
    #     if i in selected:
    #         for j in attdict[i]:
    #             data.append([i, j, attdict[i][j]])
    # data1=pandas.DataFrame(data)
    # data1.to_csv('./saved_models/att_item_5.txt',sep='\t',header=None,index=False)
    # with open (full_path, 'w' )as f:
    #     for i in range(len(attack_items)):
    #         for j in range(5825):
    #             rounded_value = round(attack_items[i][j].item(), 3)
    #             if (rounded_value >= 0.5 and rounded_value < 1.5) :
    #                 rounded_value = 0
    #                 f.write(str(usermax + i) + '\t' + str(j) + '\t' + str(rounded_value) + '\n')
    #             if rounded_value >= 1.5 and rounded_value <= 2.5:
    #                 rounded_value =1
    #                 f.write(str(usermax + i) + '\t' + str(j) + '\t' + str(rounded_value) + '\n')

    print("generate done")
if rting:
    getFile(test_loader)
    # ungetFile(test_loader)
else:
    getFileAll1(train_loader)