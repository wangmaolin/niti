#!/usr/bin/python3
import os
import sys
import logging
import argparse
import time
import torch
import torch.nn as nn
from datetime import datetime
from torchvision import datasets, transforms
from log import setup_logging, ResultsLog, save_checkpoint
from meters import AverageMeter, accuracy
from preprocess import get_transform, get_int8_transform
from lenet import lenet
from vgg import VGG_cifar 
from optim import OptimRegime
from data import get_dataset
from ti_lenet import TiLenet
from ti_vgg import TiVGG_cifar
import torch.optim as optim
import torch.nn.functional as F
import ti_torch
from torch.utils.tensorboard import SummaryWriter
import shutil

model_names = ['lenet','vgg']
parser = argparse.ArgumentParser(description='PyTorch integer MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--seed', type=int, default=-1, metavar='S',
                    help='random seed (default: None)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-all', action='store_true', default=False,
                    help='Save all checkpoints along training')
parser.add_argument('--weight-frac', action='store_true', default=False,
                    help='add another 8 bits fraction for gradient accumulation')
parser.add_argument('--weight-decay', action='store_true', default=False,
                    help='integer training weight decay')
parser.add_argument('--weight-hist', action='store_true', default=False,
                    help='record weight histogram after each epoch finishes')
parser.add_argument('--grad-hist', action='store_true', default=False,
                    help='record gradient histogram during training')
parser.add_argument('--download', action='store_true', default=False,
                    help='Download dataset')
parser.add_argument('--results-dir', metavar='RESULTS_DIR',
                    help='results dir')
parser.add_argument('--data-dir',
                    help='dataset dir')
parser.add_argument('--save', metavar='SAVE', default='',
                    help='saved folder')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                help='number of data loading workers (default: 8)')
parser.add_argument('--dataset', metavar='DATASET', default='cifar10',
                    help='dataset name or folder')
parser.add_argument('--depth',action="store",dest='depth',default=18,type=int,
                    help='resnet depth')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--model-type', default='int',choices=['int','float','hybrid'],
                    help='choose to train int model or float model')
parser.add_argument('--model', '-a', metavar='MODEL', default='vgg',choices=model_names,
                    help='model architecture: ' +' | '.join(model_names))
parser.add_argument('-e', '--evaluate', type=str, metavar='FILE',
                    help='evaluate model FILE on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--init', default='', type=str, metavar='PATH',
                    help='path to weight init checkpoint (default: none)')
def main():
    #check gpu exist
    if not torch.cuda.is_available():
        print('Require nvidia gpu with tensor core to run')
        return

    global args
    args = parser.parse_args()
    #set up log and training results save path
    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if args.save is '':
        args.save = time_stamp
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    setup_logging(os.path.join(save_path, 'results.log'),
                  resume=args.resume is not '')
    results_path = os.path.join(save_path, 'results')
    results= ResultsLog(results_path,
                        title=args.model_type+' training results - %s' % args.save)

    logging.info("saving to %s", save_path)
    logging.info("run arguments: %s", args)

    if args.weight_hist:
       writer_path = os.path.join(save_path,'runs')
       logging.info("writing weight histogram to %s", writer_path)
       if os.path.exists(writer_path) and os.path.isdir(writer_path):
            shutil.rmtree(writer_path)
       writer_weight = SummaryWriter(writer_path)
    else:
       writer_weight = None

    # random seed config
    if args.seed > 0:
        torch.manual_seed(args.seed)
        logging.info("random seed: %s", args.seed)
    else:
        logging.info("random seed: None")

    logging.info("act rounding scheme: %s", ti_torch.ACT_ROUND_METHOD.__name__)
    logging.info("err rounding scheme: %s", ti_torch.ERROR_ROUND_METHOD.__name__)
    logging.info("gradient rounding scheme: %s", ti_torch.GRAD_ROUND_METHOD.__name__)
    if args.weight_frac:
        ti_torch.UPDATE_WITH_FRAC = True
        logging.info("Update WITH Fraction")
    else:
        ti_torch.UPDATE_WITH_FRAC = False

    if args.weight_decay: 
        ti_torch.WEIGHT_DECAY = True
        logging.info("Update WITH WEIGHT DECAY")
    else:
        ti_torch.WEIGHT_DECAY = False
    # logging.info("ACC bitwidth: %d", ti_torch.ACC_BITWIDTH)

    # Create Network
    if args.model_type =='int':
        logging.info('Create integer model')
        optimizer = None
        if args.dataset =='mnist':
            model = TiLenet()
        elif args.dataset =='cifar10':
            if args.model == 'vgg':
                model = TiVGG_cifar(args.depth, 10)

        if args.weight_frac:
            regime = model.regime_frac
        else:
            regime = model.regime
    else:
        if args.dataset == 'mnist' and args.model == 'lenet':
            model= lenet().to('cuda:0')

        elif args.dataset == 'cifar10':
            if args.model == 'vgg':
                model= VGG_cifar(args.depth,10).to('cuda:0')

        num_parameters = sum([l.nelement() for l in model.parameters()])
        logging.info("created float network on %s", args.dataset)
        logging.info("number of parameters: %d", num_parameters)
        regime = getattr(model, 'regime')
        optimizer = OptimRegime(model.parameters(), regime)

    best_prec1 = 0
    if args.evaluate:
        if not os.path.isfile(args.evaluate):
            parser.error('invalid checkpoint: {}'.format(args.evaluate))
        checkpoint = torch.load(args.evaluate)
        model.load_state_dict(checkpoint['state_dict'])
        logging.info("loaded checkpoint '%s' (epoch %s)",
                     args.evaluate, checkpoint['epoch'])
    elif args.resume:
        checkpoint_file = args.resume
        if os.path.isdir(checkpoint_file):
            results.load(os.path.join(checkpoint_file, 'results.csv'))
            checkpoint_file = os.path.join(
                checkpoint_file, 'model_best.pth.tar')
        if os.path.isfile(checkpoint_file):
            logging.info("loading checkpoint '%s'", args.resume)
            checkpoint = torch.load(checkpoint_file)
            args.start_epoch = checkpoint['epoch']+1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("loaded checkpoint '%s' (epoch %s), best_prec1 %f",
                         checkpoint_file, checkpoint['epoch'],checkpoint['best_prec1'])
        else:
            logging.error("no checkpoint found at '%s'", args.resume)
    elif args.init:
        if not os.path.isfile(args.init):
            parser.error('invalid checkpoint: {}'.format(args.init))
        checkpoint = torch.load(args.init)
        model.load_state_dict(checkpoint['state_dict'])
        logging.info("initial weights from checkpoint '%s' ",args.init)

    # dataset loading code
    default_transform = {
        'train': get_transform(args.dataset, augment=True),
        'eval': get_transform(args.dataset, augment=False)
    }
    criterion = nn.CrossEntropyLoss()
    criterion.to('cuda:0')

    test_data = get_dataset(name=args.dataset,
                            split='val',
                            transform=default_transform['eval'],
                            download = args.download,
                            datasets_path=args.data_dir)

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    if args.evaluate:
        val_loss, val_prec1, val_prec5= forward(
            test_loader, model, criterion, 0, training=False, model_type=args.model_type)
        logging.info('Validation Prec@1 {val_prec1:.3f} '
                     'Validation Prec@5 {val_prec5:.3f} \n'
                     .format(val_prec1=val_prec1,
                             val_prec5=val_prec5))
        return

    train_data = get_dataset(name=args.dataset,
                             split='train',
                             transform=default_transform['train'],
                             download = args.download,
                             datasets_path=args.data_dir)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)

    logging.info('training regime: %s', regime)
    if args.model_type == 'int' or args.model_type == 'hybrid':
        for s in regime:
            if s['epoch'] == 0:
                ti_torch.GRAD_BITWIDTH = s['gb']
                break

    for epoch in range(args.start_epoch, args.epochs):
        if args.model_type == 'int' or args.model_type == 'hybrid':
            for s in regime:
                if s['epoch'] == epoch:
                    ti_torch.GRAD_BITWIDTH = s['gb']
                    logging.info('changing gradient bitwidth: %d', ti_torch.GRAD_BITWIDTH)
                    break
        # train
        train_loss, train_prec1, train_prec5= forward(
            train_loader, model, criterion, epoch, training=True, model_type=args.model_type,optimizer = optimizer,writer=writer_weight)

        val_loss, val_prec1, val_prec5= forward(
            test_loader, model, criterion, epoch, training=False, model_type=args.model_type)

        is_best = val_prec1 > best_prec1
        best_prec1 = max(val_prec1, best_prec1)
        logging.info("best_prec1: %f %s", best_prec1,save_path)
        save_checkpoint({
            'epoch': epoch,
            'model': args.model,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'regime': regime},
            is_best,
            save_path,
            'checkpoint.pth.tar',
            args.save_all)
        #record results
        logging.info(args.model_type+' '
                     'Epoch: {0} '
                     'Train Prec@1 {train_prec1:.3f} '
                     'Train Prec@5 {train_prec5:.3f} '
                     'Valid Prec@1 {val_prec1:.3f} '
                     'Valid Prec@5 {val_prec5:.3f} \n'
                     .format(epoch,
                             train_prec1=train_prec1, val_prec1=val_prec1,
                             train_prec5=train_prec5, val_prec5=val_prec5))
        results.add(epoch=epoch,
                    train_error1= 100 - train_prec1,
                    val_error1= 100 - val_prec1,
                    train_error5= 100 - train_prec5,
                    val_error5= 100 - val_prec5,
                   )

        results.plot(x='epoch', y=['train_error1', 'val_error1'],
                     legend=['train', 'val'],
                     title='Error@1', ylabel='error %')

        results.plot(x='epoch', y=['train_error5', 'val_error5'],
                     legend=['train', 'val'],
                     title='Error@5', ylabel='error %')

        results.save()

        if args.weight_hist:
            logging.info("writing weight histogram to %s", save_path)
            writer_weight.add_scalar('Loss/train', train_loss, epoch)
            writer_weight.add_scalar('Loss/test', val_loss, epoch)
            writer_weight.add_scalar('Accuracy/train', train_prec1, epoch)
            writer_weight.add_scalar('Accuracy/test', val_prec1, epoch)
            if args.model_type == 'int':
                for idx, l in enumerate(model.forward_layers):
                    if hasattr(l,'weight'):
                        weight = l.weight.float()*2**l.weight_exp.float()
                        writer_weight.add_histogram('Weight/'+l.__class__.__name__ +'_'+str(idx), weight, epoch)
                    # if hasattr(l,'bias'):
                        # bias = l.bias.float()*2**l.bias_exp.float()
                        # writer_weight.add_histogram('Bias/'+l.__class__.__name__ +'_'+str(idx), bias, epoch)
            elif args.model_type == 'float':
                for idx, l in enumerate(model.layers):
                    if hasattr(l,'weight'):
                        writer_weight.add_histogram('Weight/'+l.__class__.__name__ +'_'+str(idx), l.weight, epoch)
                    # if hasattr(l,'bias'):
                        # writer_weight.add_histogram('Bias/'+l.__class__.__name__ +'_'+str(idx), l.bias, epoch)
                for idx, l in enumerate(model.classifier):
                    if hasattr(l,'weight'):
                        writer_weight.add_histogram('Weight/'+l.__class__.__name__ +'_'+str(idx), l.weight, epoch)
                    # if hasattr(l,'bias'):
                        # writer_weight.add_histogram('Bias/'+l.__class__.__name__ +'_'+str(idx), l.bias, epoch)
            elif args.model_type == 'hybrid':
                for idx, l in enumerate(model.forward_layers):
                    if hasattr(l,'weight'):
                        weight = l.weight.float()*2**l.weight_exp.float()
                        writer_weight.add_histogram('Weight/'+l.__class__.__name__ +'_'+str(idx), weight, epoch)
                for idx, l in enumerate(model.fp32_layers):
                    if hasattr(l,'weight'):
                        writer_weight.add_histogram('Weight/'+l.__class__.__name__ +'_'+str(idx), l.weight, epoch)
 
def forward(data_loader, model, criterion, epoch, training, model_type, optimizer=None, writer=None):
    if training:
        model.train()
    else:
        model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()

    total_steps=len(data_loader)

    for i, (inputs, target) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs = inputs.to('cuda:0')
        target = target.to('cuda:0')

        # compute output
        output = model(inputs)
        if model_type == 'int':
            # omit the output exponent
            output, output_exp = output
            output = output.float()
            loss = criterion(output*(2**output_exp.float()), target)
        else:
            output_exp = 0
            loss = criterion(output, target)

        # measure accuracy and record loss
        losses.update(float(loss), inputs.size(0))
        prec1, prec5 = accuracy(output.detach(), target, topk=(1, 5))
        top1.update(float(prec1), inputs.size(0))
        top5.update(float(prec5), inputs.size(0))

        if training:
            if model_type == 'int':
                model.backward(target)

            elif model_type == 'hybrid':
                # float backward
                optimizer.update(epoch, epoch * len(data_loader) + i)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #int8 backward
                model.backward()
            else:
                optimizer.update(epoch, epoch * len(data_loader) + i)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.log_interval == 0 and training:
            logging.info('{model_type} [{0}][{1}/{2}] '
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                         'Data {data_time.val:.2f} '
                         'loss {loss.val:.3f} ({loss.avg:.3f}) '
                         'e {output_exp:d} '
                         '@1 {top1.val:.3f} ({top1.avg:.3f}) '
                         '@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                             epoch, i, len(data_loader),
                             model_type=model_type,
                             batch_time=batch_time,
                             data_time=data_time,
                             loss=losses,
                             output_exp=output_exp,
                             top1=top1, top5=top5))

            if args.grad_hist:
                if args.model_type == 'int':
                    for idx, l in enumerate(model.forward_layers):
                        if hasattr(l,'weight'):
                            grad = l.grad_int32acc
                            writer.add_histogram('Grad/'+l.__class__.__name__ +'_'+str(idx), grad, epoch*total_steps+i)

                elif args.model_type == 'float':
                    for idx, l in enumerate(model.layers):
                        if hasattr(l,'weight'):
                            writer.add_histogram('Grad/'+l.__class__.__name__ +'_'+str(idx), l.weight.grad, epoch*total_steps+i)
                    for idx, l in enumerate(model.classifier):
                        if hasattr(l,'weight'):
                            writer.add_histogram('Grad/'+l.__class__.__name__ +'_'+str(idx), l.weight.grad, epoch*total_steps+i)

    return losses.avg, top1.avg, top5.avg

if __name__ == '__main__':
    main()
