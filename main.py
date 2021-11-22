import os
import sys
import logging
import time
import torch
import torch.nn as nn
from log import setup_logging, save_checkpoint
from meters import AverageMeter, accuracy
import ti_torch
import shutil
from data import gen_loaders
from ti_vgg import TiVGG_cifar
from torch.utils.tensorboard import SummaryWriter
from options import args

def main():
    #set up log and training results save path
    if args.save == '':
        args.save = 'test' 
    save_path = os.path.join('./results/', args.save)

    if os.path.exists(save_path) and args.save =='test':
        print("remove previous test dir")
        shutil.rmtree(save_path)
    elif os.path.exists(save_path):
        print("saving path "+ save_path + " exist and not named test")
        sys.exit(0)
    else:
        os.makedirs(save_path)

    writer_path = os.path.join(save_path,'runs/vgg'+str(args.depth))
    logging.info("writing weight histogram to %s", writer_path)
    if os.path.exists(writer_path) and os.path.isdir(writer_path):
        shutil.rmtree(writer_path)
    writer_weight = SummaryWriter(writer_path)

    setup_logging(os.path.join(save_path, 'results.log'))

    logging.info("saving to %s", save_path)
    logging.info("run arguments: %s", args)

    logging.info("ACT rounding: %s", ti_torch.ACT_ROUND_METHOD.__name__)
    logging.info("ERROR rounding: %s", ti_torch.ERROR_ROUND_METHOD.__name__)
    logging.info("GRAD rounding: %s", ti_torch.GRAD_ROUND_METHOD.__name__)
    logging.info("Weight Init: %s", ti_torch.WEIGHT_INIT_METHOD.__name__)

    # random seed config
    if args.seed > 0:
        torch.manual_seed(args.seed)
        logging.info("random seed: %s", args.seed)
    else:
        logging.info("random seed: None")

    # Create Network
    logging.info('Create integer model')
    model = TiVGG_cifar(depth = args.depth)

    regime = model.regime
    logging.info('training regime: %s', regime)

    if args.init:
        ckpt = torch.load(args.init)
        logging.info("Init weights from: %s", args.init)
        model.load_state_dict(ckpt['state_dict'])

    best_prec1 = 0
    criterion = nn.CrossEntropyLoss()
    criterion.to('cuda')

    train_loader, test_loader = gen_loaders(args.data_dir, args.batch_size, args.workers)

    for epoch in range(0, args.epochs):
        for s in regime:
            if s['epoch'] == epoch:
                ti_torch.GRAD_BITWIDTH = s['gb']
                logging.info("Change Grad Bitwidth to : %s", ti_torch.GRAD_BITWIDTH)
                break

        train_loss, train_prec1, train_prec5= forward(
            train_loader, model, criterion, epoch, training=True)

        val_loss, val_prec1, val_prec5= forward(
            test_loader, model, criterion, epoch, training=False)

        writer_weight.add_scalar('Accuracy/train', train_prec1, epoch)
        writer_weight.add_scalar('Accuracy/test', val_prec1, epoch)
        writer_weight.add_scalar('Loss/train', train_loss, epoch)
        writer_weight.add_scalar('Loss/test', val_loss, epoch)

        for idx, l in enumerate(model.forward_layers):
            if isinstance(l,nn.Sequential):
                for bi, b in enumerate(l):
                    writer_weight.add_histogram('Weight/'+b.__class__.__name__+str(idx)+'_'+str(bi)+'_conv1', b.conv1.weight, epoch)
                    writer_weight.add_histogram('Weight/'+b.__class__.__name__+str(idx)+'_'+str(bi)+'_conv2', b.conv2.weight, epoch)
                    if hasattr(b,'conv3'):
                        writer_weight.add_histogram('Weight/'+b.__class__.__name__+str(idx)+'_'+str(bi)+'_conv3', b.conv3.weight, epoch)
                    if hasattr(b,'downsample'):
                        writer_weight.add_histogram('Weight/'+b.__class__.__name__+str(idx)+'_'+str(bi)+'_downsample', b.downsample.weight, epoch)
            elif hasattr(l,'weight'):
                writer_weight.add_histogram('Weight/'+l.__class__.__name__ +'_'+str(idx), l.weight, epoch)

        is_best = val_prec1 > best_prec1
        best_prec1 = max(val_prec1, best_prec1)

        writer_weight.add_scalar('Accuracy/best', best_prec1, epoch)

        logging.info("best_prec1: %f %s", best_prec1,save_path)
        save_checkpoint({'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'best_prec1': best_prec1},
                        is_best,
                        save_path,
                        'ckpt.pth',
                        args.save_all)
        #record results
        logging.info('Epoch: {0} '
                     'Train Prec@1 {train_prec1:.3f} '
                     'Train Loss {train_loss:.3f} '
                     'Valid Prec@1 {val_prec1:.3f} '
                     'Valid Loss {val_loss:.3f} \n'
                     .format(epoch,
                             train_prec1=train_prec1, val_prec1=val_prec1,
                             train_loss=train_loss, val_loss=val_loss))

def forward(data_loader, model, criterion, epoch, training):
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

    for i, data in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs, target = data
        inputs = inputs.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        if training:
            ti_torch.TRAINING = True
        else:
            ti_torch.TRAINING = False

        # compute output
        output, output_exp = model(inputs)
        ''' For display purpose only, not for actual training '''
        loss = criterion(output.float()*(2**output_exp.float()), target)

        # measure accuracy and record loss
        losses.update(float(loss), inputs.size(0))
        prec1, prec5 = accuracy(output.detach(), target, topk=(1, 5))
        top1.update(float(prec1), inputs.size(0))
        top5.update(float(prec5), inputs.size(0))

        if training:
            model.backward(target)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if training and i % args.log_interval == 0:
            result_eta_days = batch_time.avg*len(data_loader)*(args.epochs-(epoch+i*1.0/len(data_loader)))/86400
            eta_int_days = int(result_eta_days)
            eta_hours = 24*(result_eta_days-eta_int_days)
            logging.info('[{0}][{1:>3}/{2}] '
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                         'Data {data_time.val:.2f} '
                         'loss {loss.val:.3f} ({loss.avg:.3f}) '
                         '@1 {top1.val:.3f} ({top1.avg:.3f}) '
                         'ETA {eta_days:d} D {eta_hours:.1f} H'
                         .format(
                             epoch, i, len(data_loader),
                             batch_time=batch_time,
                             data_time=data_time,
                             loss=losses,
                             top1=top1,
                             eta_days=eta_int_days,
                             eta_hours=eta_hours))

    return losses.avg, top1.avg, top5.avg

if __name__ == '__main__':
    main()