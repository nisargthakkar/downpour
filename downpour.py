import os
from PIL import Image
import time
import csv
import random
import pandas as pd
import numpy as np
import enum

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

import argparse
from random import Random

TAG_HANDSHAKE_INIT = 101
TAG_GRADIENT_SEND = 102
TAG_PARAMETER_RECV = 103

HANDSHAKE_SERVER_EXIT = 0
HANDSHAKE_RECV_PARAMS = 1
HANDSHAKE_SEND_GRADIENTS = 2

NUM_EPOCH = 5

def get_log_prefix():
    rank = dist.get_rank()
    if rank == 0:
        return "SERVER"
    
    return "WORKER %s" % rank

class ModelDataType(enum.Enum):
    grad = 'grad'
    param = 'param'

class KaggleAmazonDataset(Dataset):

    def __init__(self, csv_path, img_path, img_ext, transform=None):

        tmp_df = pd.read_csv(csv_path)

        self.img_path = img_path
        self.img_ext = img_ext
        self.transform = transform

        self.X_train = tmp_df['image_name']
        self.y_train = tmp_df['tags']

        self.num_labels = 17

    def __getitem__(self, index):
        img = Image.open(self.img_path + self.X_train[index] + self.img_ext)
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label_ids = self.y_train[index].split()
        label_ids = [ int(s) for s in label_ids ]
        label=torch.zeros(self.num_labels)
        label[label_ids] = 1
        return img, label

    def __len__(self):
        return len(self.X_train.index)

class Inception_Module(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Inception_Module, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        conv1x1 = self.conv1x1(x)
        conv3x3 = self.conv3x3(x)
        conv5x5 = self.conv5x5(x)
        out = [conv1x1, conv3x3, conv5x5]
        out = torch.cat(out, 1)
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.module1 = Inception_Module(3, 10)
        self.module2 = Inception_Module(30, 10)
        self.fc1 = nn.Linear(1920, 256)
        self.fc2 = nn.Linear(256, 17)

    def forward(self, x):
        x=self.module1(x)
        x = F.relu(F.max_pool2d(x, 2))
        x=self.module2(x)
        x = F.relu(F.max_pool2d(x , 2))
        x = x.view(x.size(0), -1) # Flatten layer
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return torch.sigmoid(x)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    

def accrue_gradients(model, accruedgradients):
    for paramIdx, param in enumerate(model.parameters()):
        accruedgradients[paramIdx] += param.grad.data
    return accruedgradients

def train_worker(epoch, train_loader, model, criterion, optimizer, step, accruedgradients, worker_group=None, device=torch.device('cpu'), n_steps=1):
    losses = AverageMeter()
    precisions_1 = AverageMeter()
    precisions_k = AverageMeter()

    datasize = pack_params(model).size()

    model.train()

    log_prefix = get_log_prefix()

    print('%s starting training' % log_prefix)

    t_train = time.monotonic()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device=device)
        target = target.to(device=device)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        accruedgradients = accrue_gradients(model, accruedgradients)

        print('%s optimizing parameters' % log_prefix)
        optimizer.step()

        if step % n_steps == 0:
            # Start Asynchronously pushing gradients and fetch parameters
            send_gradients(model, accruedgradients)
            accruedgradients = reset_accrued_gradients(model)
        
        topk=3
        _, predicted = output.topk(topk, 1, True, True)
        batch_size = target.size(0)
        prec_k=0
        prec_1=0
        count_k=0
        for i in range(batch_size):
            prec_k += target[i][predicted[i]].sum()
            prec_1 += target[i][predicted[i][0]]
            count_k+=topk #min(target[i].sum(), topk)
        prec_k/=count_k
        prec_1/=batch_size
        #print ('prec_1',prec_1)

        #Update of averaged metrics
        losses.update(loss.item(), 1)
        precisions_1.update(prec_1, 1)
        precisions_k.update(prec_k, 1)

        if (batch_idx+1) % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.3f} ({:.3f}),\tPrec@1: {:.3f} ({:.3f}),\tPrec@3: {:.3f} ({:.3f}),\tTimes: Batch: {:.4f} ({:.4f}),\tDataLoader: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), losses.val, losses.avg, precisions_1.val, precisions_1.avg , precisions_k.val, precisions_k.avg ,
                batch_times.val, batch_times.avg, loader_times.avg))

        step += 1

        t_batch = time.monotonic()

    # Workers synchronize to fetch latest parameters
    if worker_group is not None:
        print('%s hit barrier' % log_prefix)
        dist.barrier(worker_group)

    fetch_parameters(model, datasize)
    print('%s received parameters at end of epoch' % log_prefix)

    train_time = time.monotonic() - t_train
    print('{} Training Epoch: {} done. \tLoss: {:.3f},\tPrec@1: {:.3f},\tPrec@3: {:.3f}\tTimes: Total: {:.3f}\n'.format(log_prefix, epoch, losses.avg, precisions_1.avg, precisions_k.avg, train_time))
    return step, accruedgradients, losses.sum, precisions_1.sum, precisions_k.sum

def train_server(model, optimizer):
    num_workers = dist.get_world_size() - 1
    max_worker_exit_messages = num_workers

    log_prefix = get_log_prefix()

    datasize = pack_params(model).size()

    while True:
        print('%s waiting for handshake' % log_prefix)
        worker = torch.zeros(1)
        src = dist.recv(tensor=worker, tag=TAG_HANDSHAKE_INIT)
        msg = worker[0]
        print('%s received handshake %s' % (log_prefix, msg))
        if msg == HANDSHAKE_SERVER_EXIT:
            # All epochs completed. Time to finish
            break
        elif msg == HANDSHAKE_RECV_PARAMS:
            paramData = pack_params(model)

            print('%s propagating parameters' % log_prefix)
            dist.send(tensor=paramData, dst=src, tag=TAG_PARAMETER_RECV)
        elif msg == HANDSHAKE_SEND_GRADIENTS:
            optimizer.zero_grad()
            print('%s waiting to receive gradients' % log_prefix)
            # receive gradient data from worker at rank 'src'
            gradientData = torch.zeros(datasize, dtype=torch.float32)
            dist.recv(tensor=gradientData, src=src, tag=TAG_GRADIENT_SEND)

            unpack_data(gradientData, model, ModelDataType.grad)

            print('%s optimizing parameters' % log_prefix)
            # Now that we have the gradients from the workers, we can perform an optimizer step
            optimizer.step()

            paramData = pack_params(model)

            print('%s propagating parameters' % log_prefix)
            dist.send(tensor=paramData, dst=src, tag=TAG_PARAMETER_RECV)

# Helper functions from PyTorch tutorials: https://pytorch.org/tutorials/intermediate/dist_tuto.html
class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

class DataPartitioner(object):

    # Seed is needed to ensure all workers partition in the same way 
    def __init__(self, data, sizes, seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

def partition_dataset(dataset):
    # Number of workers
    size = dist.get_world_size() - 1
    batch_size = 250
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank() - 1)
    train_loader = DataLoader(partition,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=args.workers # 1 for CUDA
                             )

    return train_loader

def pack_data(data):
    packedData = torch.Tensor([-1])
    for layer_data in data:
        layer_data = layer_data.view(-1)
        packedData = torch.cat([packedData, layer_data])

    return packedData[1:]

def pack_params(model):
    packedData = torch.Tensor([-1])
    for param in model.parameters():
        if param.requires_grad:
            packedData = torch.cat([packedData, param.data.view(-1)])        
    
    return packedData[1:]

def unpack_data(data, model, modelDataType):
    start_idx = 0
    for param in model.parameters():
        if param.requires_grad:
            paramLength = param.data.numel()
            if modelDataType == ModelDataType.param:
                param.data.copy_(data[start_idx:start_idx + paramLength].view(param.size()))
            else:
                param.grad.data.copy_(data[start_idx:start_idx + paramLength].view(param.size()))

            start_idx += paramLength

def send_gradients(model, gradients):
    # Start Asynchronously pushing gradients and then fetch parameters
    rank = dist.get_rank()

    log_prefix = get_log_prefix()

    print('%s initiating handshake to send gradients and receive parameters' % log_prefix)
    # Send handshake message to inform of worker rank
    worker = torch.tensor([HANDSHAKE_SEND_GRADIENTS], dtype=torch.float32)
    dist.send(tensor=worker, dst=0, tag=TAG_HANDSHAKE_INIT)

    gradientData = pack_data(gradients)
    print('%s sending gradients' % log_prefix)
    dist.send(tensor=gradientData, dst=0, tag=TAG_GRADIENT_SEND)

    print('%s waiting to receive parameters' % log_prefix)
    data = torch.zeros(gradientData.size(), dtype=torch.float32)
    dist.recv(tensor=data, src=0, tag=TAG_PARAMETER_RECV)
    unpack_data(data, model, ModelDataType.param)

def fetch_parameters(model, datasize):
    # Start Asynchronously pushing gradients and then fetch parameters
    rank = dist.get_rank()

    log_prefix = get_log_prefix()

    print('%s initiating handshake to receive parameters' % log_prefix)
    # Send handshake message to inform of worker rank
    worker = torch.tensor([HANDSHAKE_RECV_PARAMS], dtype=torch.float32)
    dist.send(tensor=worker, dst=0, tag=TAG_HANDSHAKE_INIT)

    print('%s waiting to receive parameters' % log_prefix)
    data = torch.zeros(datasize, dtype=torch.float32)
    dist.recv(tensor=data, src=0, tag=TAG_PARAMETER_RECV)
    unpack_data(data, model, ModelDataType.param)

def reset_accrued_gradients(model):
    accruedgradients = []
    for param in model.parameters():
        if param.requires_grad:
            accruedgradients.append(torch.zeros(param.size(), requires_grad=False))

    return accruedgradients

def init_program(rank, size, args):
    device = None
    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print ('device:', device)
    DATA_PATH=args.data_path
    IMG_PATH = DATA_PATH+'train-jpg/'
    IMG_EXT = '.jpg'
    TRAIN_DATA = DATA_PATH+'train.csv'
    print ('dataloader_workers',args.workers)

    model = Net().to(device=device)

    print ('Optimizer:',args.opt)
    if args.opt=='adam':
        # optimizer = optim.Adam(model.parameters()) # Using default Adam values
        optimizer = optim.Adam(model.parameters(), lr=0.01)
    elif args.opt=='adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=0.01)
    elif args.opt=='adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    elif args.opt=='nesterov':
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=False)

    # Initializing gradients to 0 
    for param in model.parameters():
        param.grad = torch.zeros(param.size(), requires_grad=True)
        param.grad.data.zero_()
        # optimizer.zero_grad()

    log_prefix = get_log_prefix()

    world_size = dist.get_world_size()
    worker_list = list(range(1, world_size))
    print('%s creating group list. Waiting for %s' % (log_prefix, worker_list))
    worker_group = dist.new_group(worker_list)

    if rank == 0:
        train_server(model, optimizer)
        print('%s exiting' % log_prefix)
    else:
        batch_size=250

        transformations = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
        
        dset_train = KaggleAmazonDataset(TRAIN_DATA,IMG_PATH,IMG_EXT,transformations)
        print ('%s dataset loaded. Size = %s' %(log_prefix, len(dset_train)))
        
        train_loader = partition_dataset(dset_train)
        print ('%s dataset partitioned. Number of batches = %s' %(log_prefix, len(train_loader)))

        criterion = nn.BCELoss().to(device=device)
        
        accruedgradients = reset_accrued_gradients(model)

        step = 0
        loss_sum = 0
        loss = 0
        train_start_time = time.monotonic()
        for epoch in range(NUM_EPOCH):
            step, accruedgradients, loss, prec_1, prec_3 = train_worker(epoch, train_loader, model, criterion, optimizer, step, accruedgradients, worker_group=worker_group, device=device, n_steps=args.n_steps)
            
        # All reduce to get loss values
        if worker_group is not None:
            # t_samples = torch.tensor(len(train_loader) * NUM_EPOCH)
            num_samples = len(train_loader)
            
            t_report = torch.tensor([loss, prec_1, prec_3, num_samples])
            print('%s collecting loss, prec_1, prec_3, num_samples values' % log_prefix)
            dist.all_reduce(t_report, op=dist.ReduceOp.SUM, group=worker_group)

            print('%s Weighted Loss: %s' % (log_prefix, t_report[0].item()/t_report[3].item()))
            print('%s Weighted Prec@1: %s' % (log_prefix, t_report[1].item()/t_report[3].item()))
            print('%s Weighted Prec@3: %s' % (log_prefix, t_report[2].item()/t_report[3].item()))

        if rank == 1:
            # Send server exit message
            print('%s initiating handshake to exit server' % log_prefix)
            worker = torch.tensor([HANDSHAKE_SERVER_EXIT], dtype=torch.float32)
            dist.send(tensor=worker, dst=0, tag=TAG_HANDSHAKE_INIT)

        total_time = time.monotonic() - train_start_time

        # print('{} Final Average Times: Total: {:.3f}, Avg-Batch: {:.4f}, Avg-Loader: {:.4f}, Weighted-Loss: {}\n'.format(log_prefix, np.average(train_times), np.average(batch_times), np.average(loader_times), weightedLoss))
        print('{} \tLoss: {:.3f},\tPrec@1: {:.3f},\tPrec@3: {:.3f}\tTimes: Total: {:.3f}\n'.format(log_prefix, loss/num_samples, prec_1/num_samples, prec_3/num_samples, total_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--disable_cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--data_path', type=str, default='/scratch/gd66/spring2019/lab4/kaggleamazon/',
                        help='Data path')
    parser.add_argument('--opt', type=str, default='adam',
                        help='NN optimizer (Examples: adam, rmsprop, adadelta, ...)')
    parser.add_argument('--n_steps', type=int, default=1,
                        help='Steps after which to sync with server')
    args = parser.parse_args()


    dist.init_process_group(backend='mpi')

    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    init_program(dist.get_rank(), dist.get_world_size(), args)