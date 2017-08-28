import argparse
import os
import time
import numpy as np
import data
from importlib import import_module
import shutil
from utils import *
import sys
sys.path.append('../')
from split_combine import SplitComb

import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
from config_training import config as config_training

from layers import acc
import csv

from logger import Logger as LOGGER
parser = argparse.ArgumentParser(description='PyTorch DataBowl3 Detector')
parser.add_argument('--model', '-m', metavar='MODEL', default='base',
                    help='model')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--save-freq', default='1', type=int, metavar='S',
                    help='save frequency')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save-dir', default='', type=str, metavar='SAVE',
                    help='directory to save checkpoint (default: none)')
parser.add_argument('--test', default=0, type=int, metavar='TEST',
                    help='1 do test evaluation, 0 not')
parser.add_argument('--split', default=8, type=int, metavar='SPLIT',
                    help='In the test phase, split the image to 8 parts')
parser.add_argument('--gpu', default='all', type=str, metavar='N',
                    help='use gpu')
parser.add_argument('--n_test', default=8, type=int, metavar='N',
                    help='number of gpu for test')

def main():
        
    global args
    args = parser.parse_args()
    
    
    torch.manual_seed(0)
    torch.cuda.set_device(0)

    model = import_module(args.model)
    config, net, loss, get_pbb = model.get_model()
    start_epoch = args.start_epoch
    save_dir = args.save_dir
    
    if args.resume:
        checkpoint = torch.load(args.resume)
        if start_epoch == 0:
            start_epoch = checkpoint['epoch'] + 1
        if not save_dir:
            save_dir = checkpoint['save_dir']
        else:
            save_dir = os.path.join('results',save_dir)
        net.load_state_dict(checkpoint['state_dict'])
    else:
        if start_epoch == 0:
            start_epoch = 1
        if not save_dir:
            exp_id = time.strftime('%Y%m%d-%H%M%S', time.localtime())
            save_dir = os.path.join('results', args.model + '-' + exp_id)
        else:
            save_dir = os.path.join('results',save_dir)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logfile = os.path.join(save_dir,'log')
    if args.test!=1:
        sys.stdout = Logger(logfile)
        pyfiles = [f for f in os.listdir('./') if f.endswith('.py')]
        for f in pyfiles:
            shutil.copy(f,os.path.join(save_dir,f))
    n_gpu = setgpu(args.gpu)
    args.n_gpu = n_gpu
    net = net.cuda()
    loss = loss.cuda()
    cudnn.benchmark = True
    net = DataParallel(net)
    datadir = config_training['preprocess_result_path']
    
    if args.test == 1:
        margin = 32
        sidelen = 144

        split_comber = SplitComb(sidelen,config['max_stride'],config['stride'],margin,config['pad_value'])
        dataset = data.DataBowl3Detector(
            datadir,
            'full.npy',
            config,
            phase='test',
            split_comber=split_comber)
        test_loader = DataLoader(
            dataset,
            batch_size = 1,
            shuffle = False,
            num_workers = args.workers,
            collate_fn = data.collate,
            pin_memory=False)
        
        test(test_loader, net, get_pbb, save_dir,config)
        return

    #net = DataParallel(net)
    
    dataset = data.DataBowl3Detector(
        datadir,
        'kaggleluna_full.npy',
        config,
        phase = 'train')
    train_loader = DataLoader(
        dataset,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.workers,
        pin_memory=True)
    dataset = data.DataBowl3Detector(
        datadir,
        'valsplit.npy',
        config,
        phase = 'val')
    val_loader = DataLoader(
        dataset,
        batch_size = 2,
        shuffle = False,
        num_workers = args.workers,
        pin_memory=True)
    print(args.batch_size)
    optimizer = torch.optim.SGD(
        net.parameters(),
        args.lr,
        momentum = 0.9,
        weight_decay = args.weight_decay)
    
    def get_lr(epoch):
        if epoch <= args.epochs * 0.5:
            lr = args.lr
        elif epoch <= args.epochs * 0.8:
            lr = 0.1 * args.lr
        else:
            lr = 0.01 * args.lr
        return lr
    
    #train_loss = []
    #val_loss = []
    logger = LOGGER('./board')
    train_ct = 0
    val_ct = 0
    for epoch in range(start_epoch, args.epochs + 1):
        train_ct = train(train_loader, net, loss, epoch, optimizer, get_lr, args.save_freq, save_dir, train_ct, logger)
        val_ct = validate(val_loader, net, loss, val_ct, logger)
    '''    
    for i in range(len(train_loss) - len(val_loss)):
        val_loss.append(0)
    writer = csv.writer(file('loss.csv', 'wb'))
    writer.writerow(['iter', 'train_loss', 'val_loss'])
    for i in range(len(train_loss)):
        writer.writerow([i, train_loss[i], val_loss[i]])
    '''
def train(data_loader, net, loss, epoch, optimizer, get_lr, save_freq, save_dir, count, logger):
    start_time = time.time()
    
    net.train()
    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    metrics = []
    #print('hhhhhh')
    for i, (data, target0, target1, target2, target3, coord) in enumerate(data_loader):
        #count += args.batch_size
        #print('Counting: %d'%count)

        data = Variable(data.cuda(async = True))
        target0 = Variable(target0.cuda(async = True))
        target1 = Variable(target1.cuda(async = True))
        target2 = Variable(target2.cuda(async = True))
        target3 = Variable(target3.cuda(async = True))
        coord = Variable(coord.cuda(async = True))
        output0, output1, output2, output3 = net(data, coord)
        loss_output = loss(output0, target0, output1, target1, output2, target2, output3, target3)

        optimizer.zero_grad()
        loss_output[0].backward()
        optimizer.step()
        #print('bkwd done.')
        loss_output[0] = loss_output[0].data[0]
        metrics.append(loss_output)
        info = {
            'train_loss': loss_output[0],
            'train_cls_loss':loss_output[1],
            'train_reg_loss':loss_output[2] + loss_output[3] + loss_output[4] + loss_output[5],
            'train_reg_loss_z': loss_output[2],
            'train_reg_loss_x': loss_output[3],
            'train_reg_loss_y': loss_output[4],
            'train_reg_loss_r': loss_output[5],
        }

        for tag, value in info.items():
            #print(tag, value, count)
            logger.scalar_summary(tag, value, count)

        count += 1
    if epoch % args.save_freq == 0:         
        print('epoch: %d'%(epoch))    
        state_dict = net.module.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
            
        torch.save({
            'epoch': epoch,
            'save_dir': save_dir,
            'state_dict': state_dict,
            'args': args},
            os.path.join(save_dir, '%03d.ckpt' % epoch))

    end_time = time.time()
    metrics = np.asarray(metrics, np.float32)
    #print(metrics0.shape)
    print('Epoch %03d (lr %.5f)' % (epoch, lr))
    print('Train:      tpr %3.2f, tnr %3.2f, total pos %d, total neg %d, time %3.2f' % (
        100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),
        100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]),
        np.sum(metrics[:, 7]),
        np.sum(metrics[:, 9]),
        end_time - start_time))
    print('loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f' % (
        np.mean(metrics[:, 0]),
        np.mean(metrics[:, 1]),
        np.mean(metrics[:, 2]),
        np.mean(metrics[:, 3]),
        np.mean(metrics[:, 4]),
        np.mean(metrics[:, 5])))
    print
    return count
def validate(data_loader, net, loss, count, logger):
    start_time = time.time()
    
    net.eval()
    metrics = []
    for i, (data, target0, target1, target2, target3, coord) in enumerate(data_loader):
        data = Variable(data.cuda(async = True))
        target0 = Variable(target0.cuda(async = True))
        target1 = Variable(target1.cuda(async = True))
        target2 = Variable(target2.cuda(async = True))
        target3 = Variable(target3.cuda(async = True))
        coord = Variable(coord.cuda(async = True))
        #loss_output = Variable(loss_output.cuda(async = True))

        output0, output1, output2, output3 = net(data, coord)

        loss_output = loss(output0, target0, output1, target1, output2, target2, output3, target3)
        #loss_output[0] = loss_output[0].data[0]
        loss_output[0] = loss_output[0].data[0]
        metrics.append(loss_output)
        info = {
            'val_loss': loss_output[0],
            'val_cls_loss':loss_output[1],
            'val_reg_loss':loss_output[2] + loss_output[3] + loss_output[4] + loss_output[5],
            'val_reg_loss_z': loss_output[2],
            'val_reg_loss_x': loss_output[3],
            'val_reg_loss_y': loss_output[4],
            'val_reg_loss_r': loss_output[5],
        }
        for tag, value in info.items():
            logger.scalar_summary(tag, value, count)
        count += 1
        #val_loss.append(loss_output0[0] + loss_output1[0] + loss_output2[0] + loss_output3[0])
        #print('Total val loss iter %d: %.4f'%(i, loss_output0[0] + loss_output1[0] + loss_output2[0] + loss_output3[0]))

    end_time = time.time()

    metrics = np.asarray(metrics, np.float32)


    #print('Epoch %03d (lr %.5f)' % (epoch, lr))
    
    print('Validation:      tpr %3.2f, tnr %3.2f, total pos %d, total neg %d, time %3.2f' % (
        100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),
        100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]),
        np.sum(metrics[:, 7]),
        np.sum(metrics[:, 9]),
        end_time - start_time))
    print('loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f' % (
        np.mean(metrics[:, 0]),
        np.mean(metrics[:, 1]),
        np.mean(metrics[:, 2]),
        np.mean(metrics[:, 3]),
        np.mean(metrics[:, 4]),
        np.mean(metrics[:, 5])))
    print
    return count

def test(data_loader, net, get_pbb, save_dir, config):
    start_time = time.time()
    save_dir = os.path.join(save_dir,'bbox')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(save_dir)
    net.eval()
    namelist = []
    split_comber = data_loader.dataset.split_comber
    for i_name, (data, target, coord, nzhw) in enumerate(data_loader):
        s = time.time()
        target = [np.asarray(t, np.float32) for t in target]
        lbb = target[0]
        nzhw = nzhw[0]
        name = data_loader.dataset.filenames[i_name].split('-')[0].split('/')[-1].split('_clean')[0]
        data = data[0][0]
        coord = coord[0][0]
        isfeat = False
        if 'output_feature' in config:
            if config['output_feature']:
                isfeat = True
        n_per_run = args.n_test
        print(data.size())
        splitlist = range(0,len(data)+1,n_per_run)
        if splitlist[-1]!=len(data):
            splitlist.append(len(data))
        outputlist0 = []
        outputlist1 = []
        outputlist2 = []
        outputlist3 = []
        featurelist = []

        for i in range(len(splitlist)-1):
            input = Variable(data[splitlist[i]:splitlist[i+1]], volatile = True).cuda()
            inputcoord = Variable(coord[splitlist[i]:splitlist[i+1]], volatile = True).cuda()
            if isfeat:
                output,feature = net(input,inputcoord)
                featurelist.append(feature.data.cpu().numpy())
            else:
                output0, output1, output2, output3 = net(input,inputcoord)

            outputlist0.append(output0.data.cpu().numpy())
            outputlist1.append(output1.data.cpu().numpy())
            outputlist2.append(output2.data.cpu().numpy())
            outputlist3.append(output3.data.cpu().numpy())

        output0 = np.concatenate(outputlist0,0)
        output1 = np.concatenate(outputlist1,0)
        output2 = np.concatenate(outputlist1,0)
        output3 = np.concatenate(outputlist1,0)

        output0 = split_comber.combine(output0, stride = 2, nzhw=nzhw)
        output1 = split_comber.combine(output1, stride = 4, nzhw=nzhw)
        output2 = split_comber.combine(output2, stride = 8, nzhw=nzhw)
        output3 = split_comber.combine(output3, stride = 16,nzhw=nzhw)

        if isfeat:
            feature = np.concatenate(featurelist,0).transpose([0,2,3,4,1])[:,:,:,:,:,np.newaxis]
            feature = split_comber.combine(feature,sidelen)[...,0]

        thresh = 0.8
        outputs={}
        pbb0,mask0 = get_pbb(output0, stride = 0, thresh = thresh, ismask=True)
        pbb1,mask1 = get_pbb(output1, stride = 1, thresh = thresh, ismask=True)
        pbb2,mask2 = get_pbb(output2, stride = 2, thresh = thresh, ismask=True)
        pbb3,mask3 = get_pbb(output3, stride = 3, thresh = thresh, ismask=True)
        if isfeat:
            feature_selected = feature[mask[0],mask[1],mask[2]]
            np.save(os.path.join(save_dir, name+'_feature.npy'), feature_selected)
        #tp,fp,fn,_ = acc(pbb,lbb,0,0.1,0.1)
        #print([len(tp),len(fp),len(fn)])
        print([i_name,name])
        e = time.time()
        np.save(os.path.join(save_dir, name+'_pbb.npy'), pbb)
        np.save(os.path.join(save_dir, name+'_lbb.npy'), lbb)
    np.save(os.path.join(save_dir, 'namelist.npy'), namelist)
    end_time = time.time()


    print('elapsed time is %3.2f seconds' % (end_time - start_time))
    print
    print

def singletest(data,net,config,splitfun,combinefun,n_per_run,margin = 64,isfeat=False):
    z, h, w = data.size(2), data.size(3), data.size(4)
    print(data.size())
    data = splitfun(data,config['max_stride'],margin)
    data = Variable(data.cuda(async = True), volatile = True,requires_grad=False)
    splitlist = range(0,args.split+1,n_per_run)
    outputlist = []
    featurelist = []
    for i in range(len(splitlist)-1):
        if isfeat:
            output,feature = net(data[splitlist[i]:splitlist[i+1]])
            featurelist.append(feature)
        else:
            output = net(data[splitlist[i]:splitlist[i+1]])
        output = output.data.cpu().numpy()
        outputlist.append(output)
        
    output = np.concatenate(outputlist,0)
    output = combinefun(output, z / config['stride'], h / config['stride'], w / config['stride'])
    if isfeat:
        feature = np.concatenate(featurelist,0).transpose([0,2,3,4,1])
        feature = combinefun(feature, z / config['stride'], h / config['stride'], w / config['stride'])
        return output,feature
    else:
        return output
if __name__ == '__main__':
    main()

