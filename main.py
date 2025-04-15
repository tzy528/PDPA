"""
Train a diffusion model for recommendation
"""

import argparse
from ast import parse
import os
import time
import numpy as np
import copy

import pandas
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
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=320)
parser.add_argument('--epochs', type=int, default=800, help='upper epoch limit')
parser.add_argument('--topN', type=str, default='[10, 20, 50, 100]')
parser.add_argument('--tst_w_val', action='store_true', help='test with validation')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--gpu', type=str, default='0', help='gpu card ID')
parser.add_argument('--save_path', type=str, default='./saved_models/', help='save model path')
parser.add_argument('--log_name', type=str, default='log', help='the log name')
parser.add_argument('--round', type=int, default=1, help='record the experiment')

# params for the model
parser.add_argument('--time_type', type=str, default='cat', help='cat or add')
parser.add_argument('--dims', type=str, default='[1000]', help='the dims for the DNN')
parser.add_argument('--norm', type=bool, default=False, help='Normalize the input or not')
parser.add_argument('--emb_size', type=int, default=10, help='timestep embedding size')

# params for diffusion
parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
parser.add_argument('--steps', type=int, default=1000, help='diffusion steps')  #
parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
parser.add_argument('--noise_scale', type=float, default=0.01, help='noise scale for noise generating')
parser.add_argument('--noise_min', type=float, default=0.0001, help='noise lower bound for noise generating')
parser.add_argument('--noise_max', type=float, default=0.02, help='noise upper bound for noise generating')
parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
parser.add_argument('--sampling_steps', type=int, default=50, help='steps of the forward process during inference')
parser.add_argument('--reweight', type=bool, default=True, help='assign different weight to different timestep or not')

args = parser.parse_args()
print("args:", args)
args.sampling_steps = args.steps
print("sam_step",args.sampling_steps)
print("step",args.steps)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda:0" if args.cuda else "cpu")

print("Starting time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

### DATA LOAD ###
train_path = args.data_path +args.dataset + 'train_list.txt'
valid_path = args.data_path +args.dataset + 'train_list.txt'
test_path = args.data_path +args.dataset + 'train_list.txt'

train_data, valid_y_data, test_y_data, n_user, n_item = data_utils.data_load(train_path, valid_path, test_path)
train_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.A))           #A:[user,[ item]]
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, worker_init_fn=worker_init_fn)
print('train:',len(train_dataset))
testA=test_y_data.A
# testA[ testA ==0 ]=-1
# testSave=pandas.DataFrame(testA)
# testSave.to_csv(args.data_path +args.dataset + '-1test.txt',index=False,header=False)
# print(testA)
test_dataset = data_utils.DataDiffusion(torch.FloatTensor(testA))           #A:[user,[ item]]
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=False, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)
print("test:",len(test_dataset))
rting=0
if rting:
    loader=test_loader
else:
    loader=train_loader

print('data ready.')


### Build Gaussian Diffusion ###
if args.mean_type == 'x0':
    mean_type = gd.ModelMeanType.START_X
elif args.mean_type == 'eps':
    mean_type = gd.ModelMeanType.EPSILON
else:
    raise ValueError("Unimplemented mean type %s" % args.mean_type)

diffusion = gd.GaussianDiffusion(mean_type, args.noise_schedule, \
        args.noise_scale, args.noise_min, args.noise_max, args.steps, device).to(device)

### Build MLP ###
out_dims = eval(args.dims) + [n_item]
print(out_dims)
in_dims = out_dims[::-1]
model = DNN(in_dims, out_dims, args.emb_size, time_type="cat", norm=args.norm).to(device)

optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
print("models ready.")

param_num = 0
mlp_num = sum([param.nelement() for param in model.parameters()])
diff_num = sum([param.nelement() for param in diffusion.parameters()])  # 0
param_num = mlp_num + diff_num
print("Number of all parameters:", param_num)

def evaluate(data_loader, data_te, mask_his, topN):
    model.eval()
    e_idxlist = list(range(mask_his.shape[0]))
    e_N = mask_his.shape[0]

    predict_items = []
    target_items = []
    for i in range(e_N):
        target_items.append(data_te[i, :].nonzero()[1].tolist())
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            his_data = mask_his[e_idxlist[batch_idx*args.batch_size:batch_idx*args.batch_size+len(batch)]]
            batch = batch.to(device)
            prediction = diffusion.p_sample(model, batch, args.sampling_steps, args.sampling_noise)
            prediction[his_data.nonzero()] = -np.inf

            _, indices = torch.topk(prediction, topN[-1])
            indices = indices.cpu().numpy().tolist()
            predict_items.extend(indices)

    test_results = evaluate_utils.computeTopNAccuracy(target_items, predict_items, topN)

    return test_results

def getFile(data_loader,epoch):
    model.eval()
    attack_items = []
    sid = []
    cout = 0
    temp = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            batch = batch.to(device)
            prediction = diffusion.getAttack(model, batch, args.sampling_steps, args.sampling_noise)
            attack_items.extend(prediction)
    print(len(attack_items))
    dir_path = os.path.join(args.save_path, '50')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    step = '10'
    file_name = f"attfilestep{step}_{epoch}.txt"
    full_path = os.path.join(dir_path, file_name)
    usermax = 338
    for i in range(len(attack_items)):
        for j in range(1644):
            if attack_items[i][j].item() > 0:
                sid.append([int(i), int(j), round(attack_items[i][j].item(), 0)])
    savelist = pandas.DataFrame(sid)
    savelist.to_csv(full_path, sep='\t', index=False, header=False)
    print("generate done")

def getFileAll1(data_loader,epoch):
    model.eval()
    attack_items = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            batch = batch.to(device)
            prediction = diffusion.getAttack(model, batch, args.sampling_steps, args.sampling_noise)
            attack_items.extend(prediction)
    print(len(attack_items))

    dir_path = os.path.join(args.save_path,'m1')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_name = f"{args.steps}.txt"
    full_path = os.path.join(dir_path, file_name)
    savelist=[]
    usermask = pandas.read_csv('./saved_models/att_item_5.txt', header=None, names=['user'])
    usermask = usermask['user'].tolist()
    print(usermask)

    uidict = {}
    attdict = {}
    for i in range(len(attack_items)):
        for j in range(3900):
            round_value = round(attack_items[i][j].item(), 3)
            if round_value >= 0.5:
                if i not in uidict.keys():
                    uidict[i] = {}
                uidict[i][j] = round_value
            if round_value >= 0.5 and round_value <= 3:
                if i not in attdict.keys():
                    attdict[i] = []
                attdict[i].append(j)
                if round_value > 1.5:
                    attdict[i].append(j)
    counter = [0 for i in range(339)]
    for i in attdict:
        for j in usermask:
            for k in attdict[i]:
                if j == k:
                    counter[i] += 1
    print(counter)
    selected = sorted(range(len(counter)), key=lambda i: counter[i], reverse=True)[:34]
    print(selected)
    filler1 = []
    filler = {}
    for i in uidict:
        if i in selected:
            if i not in filler.keys():
                filler[i] = {}
            for j in range(5825):
                if j in uidict[i] or j in usermask:
                    if j in uidict[i]:
                        filler[i][j] = uidict[i][j]
                    else:
                        filler[i][j] = 1
            # for j in uidict[i]:
            #     filler[i][j]=uidict[i][j]
            # for att in usermask:
            #     if att not in filler[i].keys():
            #         filler[i][att]= 1
    for uid, val in filler.items():
        for iid, rt in val.items():
            filler1.append([uid, iid, rt])
    filler1 = pandas.DataFrame(filler1)
    filler1.to_csv('./saved_models/mask/attack_step10.txt', sep='\t', header=None, index=False)

    print("generate done")



best_recall, best_epoch = -100, 0
best_test_result = None
stopepoch=[0 for i in range(5)]
print("Start training...")
for epoch in range(1, args.epochs + 1):
    '''
    
    if epoch - best_epoch >= 20:
        print('-'*18)
        print('Exiting from training early')
        break

    '''
    model.train()
    start_time = time.time()

    batch_count = 0
    total_loss = 0.0
    
    for batch_idx, batch in enumerate(loader):

        batch = batch.to(device)
        batch_count += 1
        optimizer.zero_grad()
        losses = diffusion.training_losses(model, batch, args.reweight)
        loss = losses["loss"].mean()

        total_loss += loss
        loss.backward()
        optimizer.step()
    modelsave=args.save_path + 'data/'
    m1path=args.save_path + 'm1model/'
    if not os.path.exists(m1path):
        os.makedirs(m1path)
    if epoch >5:
        if epoch % 5 == 0:
          if not rting:
            torch.save(model, '{}_{}generate_steps{}_{}_{}_{}_{}.pth'.format(m1path, args.batch_size,args.steps,epoch,args.noise_scale,args.noise_min,args.noise_max))
          else:
            torch.save(model, '{}_{}_{}steps{}_{}_{}_{}.pth'.format(m1path,args.batch_size,epoch, args.steps,args.noise_scale,args.noise_min,args.noise_max))
            # getFile(loader,epoch)
          # getFileAll1(train_loader,epoch)
    #
    # if epoch>60:
    #     break


    '''
     if epoch % 5 == 0:
        valid_results = evaluate(test_loader, valid_y_data, train_data, eval(args.topN))
        if args.tst_w_val:
            test_results = evaluate(test_twv_loader, test_y_data, mask_tv, eval(args.topN))
        else:
            test_results = evaluate(test_loader, test_y_data, mask_tv, eval(args.topN))
        evaluate_utils.print_results(None, valid_results, test_results)

        if valid_results[1][1] > best_recall: # recall@20 as selection
            best_recall, best_epoch = valid_results[1][1], epoch
            best_results = valid_results
            best_test_results = test_results

            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            torch.save(model, '{}{}_lr{}_wd{}_bs{}_dims{}_emb{}_{}_steps{}_scale{}_min{}_max{}_sample{}_reweight{}_{}.pth' \
                .format(args.save_path, args.dataset, args.lr, args.weight_decay, args.batch_size, args.dims, args.emb_size, args.mean_type, \
                args.steps, args.noise_scale, args.noise_min, args.noise_max, args.sampling_steps, args.reweight, args.log_name))
    
    '''

    print("Runing Epoch {:03d} ".format(epoch) + 'train loss {:.4f}'.format(total_loss) + " costs " + time.strftime(
                        "%H: %M: %S", time.gmtime(time.time()-start_time)))
    print('---'*18)

print('==='*18)
#print("End. Best Epoch {:03d} ".format(best_epoch))
#evaluate_utils.print_results(None, best_results, best_test_results)
print("End time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))





