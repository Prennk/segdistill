import argparse
import time
import datetime
import os
import shutil
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from utils.logger import setup_logger
from utils.score import SegmentationMetric

from dataset.cityscapes import CSTrainValSet
from dataset.ade20k import ADETrainSet, ADEDataValSet
from dataset.camvid import CamvidTrainSet, CamvidValSet
from dataset.voc import VOCDataTrainSet, VOCDataValSet
from dataset.coco_stuff_164k import CocoStuff164kTrainSet, CocoStuff164kValSet

from utils.flops import cal_multi_adds, cal_param_size
from models.model_zoo import get_segmentation_model
from losses import SegCrossEntropyLoss

def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training With Pytorch')
    # model and dataset
    parser.add_argument('--model', type=str, default='deeplabv3',
                        help='model name')  
    parser.add_argument('--backbone', type=str, default='resnet18',
                        help='backbone name')
    parser.add_argument('--dataset', type=str, default='citys',
                        help='dataset name')
    parser.add_argument('--data', type=str, default='./dataset/cityscapes/',  
                        help='dataset directory')
    parser.add_argument('--crop-size', type=int, default=[512, 1024], nargs='+',
                        help='crop image size: [height, width]')
    parser.add_argument('--workers', '-j', type=int, default=8,
                        metavar='N', help='dataloader threads')
    
    # training hyper params
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    parser.add_argument('--aux-weight', type=float, default=0.4,
                        help='auxiliary loss weight')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--max-iterations', type=int, default=40000, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.02, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M',
                        help='w-decay (default: 5e-4)')
    parser.add_argument('--ignore-label', type=int, default=-1, metavar='N',
                        help='input batch size for training (default: 8)')
    
    # cuda setting
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    # checkpoint and log
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--save-dir', default='~/.torch/models',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--save-epoch', type=int, default=10,
                        help='save model every checkpoint-epoch')
    parser.add_argument('--log-dir', default='../runs/logs/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--log-iter', type=int, default=10,
                        help='print log every log-iter')
    parser.add_argument('--save-per-iters', type=int, default=800,
                        help='per iters to save')
    parser.add_argument('--val-per-iters', type=int, default=800,
                        help='per iters to val')
    parser.add_argument('--pretrained-base', type=str, default='resnet18-5c106cde.pth',
                        help='pretrained backbone')
    parser.add_argument('--pretrained', type=str, default='None',
                        help='pretrained seg model')

                        
    # evaluation only
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='run validation every val-epoch')
    parser.add_argument('--skip-val', action='store_true', default=False,
                        help='skip validation during training')
                        
    args = parser.parse_args()

    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = False
        args.device = "cuda"
    else:
        args.device = "cpu"

    return args
    


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        if args.dataset == 'citys':
            train_dataset = CSTrainValSet(args.data, 
                                            list_path='./dataset/list/cityscapes/train.lst', 
                                            max_iters=args.max_iterations*args.batch_size, 
                                            crop_size=args.crop_size, scale=True, mirror=True)
            val_dataset = CSTrainValSet(args.data, 
                                        list_path='./dataset/list/cityscapes/val.lst', 
                                        crop_size=(1024, 2048), scale=False, mirror=False)
        elif args.dataset == 'voc':
            train_dataset = VOCDataTrainSet(args.data, './dataset/list/voc/train_aug.txt', max_iters=args.max_iterations*args.batch_size, 
                                          crop_size=args.crop_size, scale=True, mirror=True)
            val_dataset = VOCDataValSet(args.data, './dataset/list/voc/val.txt')
        elif args.dataset == 'camvid':
            train_dataset = CamvidTrainSet(args.data, './dataset/list/CamVid/camvid_train_list.txt', max_iters=args.max_iterations*args.batch_size,
                            ignore_label=args.ignore_label, crop_size=args.crop_size, scale=True, mirror=True)
            val_dataset = CamvidValSet(args.data, './dataset/list/CamVid/camvid_val_list.txt')
        elif args.dataset == 'ade20k':
            train_dataset = ADETrainSet(args.data, max_iters=args.max_iterations*args.batch_size, ignore_label=args.ignore_label,
                                        crop_size=args.crop_size, scale=True, mirror=True)
            val_dataset = ADEDataValSet(args.data)
        elif args.dataset == 'coco_stuff_164k':
            train_dataset = CocoStuff164kTrainSet(args.data, './dataset/list/coco_stuff_164k/coco_stuff_164k_train.txt', max_iters=args.max_iterations*args.batch_size, ignore_label=args.ignore_label,
                                        crop_size=args.crop_size, scale=True, mirror=True)
            val_dataset = CocoStuff164kValSet(args.data, './dataset/list/coco_stuff_164k/coco_stuff_164k_val.txt')
        else:
            raise ValueError('dataset unfind')

        # create network
        BatchNorm2d = nn.BatchNorm2d  # Tidak ada distribusi
        self.model = get_segmentation_model(model=args.model,
                                            backbone=args.backbone, 
                                            pretrained=args.pretrained, 
                                            pretrained_base=args.pretrained_base,
                                            aux=args.aux, 
                                            norm_layer=BatchNorm2d,
                                            num_class=train_dataset.num_class).to(self.device)

        # create logger
        self.logger = setup_logger('train', args.log_dir, 0)
        self.logger.info('Training with args: %s', args)

        self.model.eval()
        with torch.no_grad():
            self.logger.info('Params: %.2fM FLOPs: %.2fG'
                % (cal_param_size(self.model) / 1e6, cal_multi_adds(self.model, (1, 3, 1024, 2048))/1e9))
        

        args.batch_size = args.batch_size
        train_sampler = None
        train_batch_sampler = data.BatchSampler(data.RandomSampler(train_dataset), args.batch_size, drop_last=True)
        val_sampler = None
        val_batch_sampler = data.BatchSampler(data.SequentialSampler(val_dataset), 1, drop_last=True)

        self.train_loader = data.DataLoader(dataset=train_dataset, 
                                            batch_sampler=train_batch_sampler,
                                            num_workers=args.workers,
                                            pin_memory=True)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=args.workers,
                                          pin_memory=True)
        

        # resume checkpoint if needed
        if args.resume:
            if os.path.isfile(args.resume):
                name, ext = os.path.splitext(args.resume)
                assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
                print('Resuming training, loading {}...'.format(args.resume))
                self.model.load_state_dict(torch.load(args.resume, map_location=lambda storage, loc: storage))
            else:
                raise ValueError('Checkpoint file not found')

        # create optimizer
        if args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        else:
            raise ValueError('Optimizer not supported')

        # create loss function
        self.criterion = SegCrossEntropyLoss(ignore_index=args.ignore_label, aux_weight=args.aux_weight).to(self.device)
        
        # initialize training states
        self.best_miou = 0.0

    def adjust_lr(self, optimizer, iteration, args):
        lr = args.lr * (0.1 ** (iteration // (args.max_iterations // 3)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def train(self):
        args = self.args
        model = self.model
        criterion = self.criterion
        optimizer = self.optimizer
        train_loader = self.train_loader
        val_loader = self.val_loader
        logger = self.logger

        # Set the model to training mode
        model.train()

        start_time = time.time()
        for iteration, (images, labels) in enumerate(train_loader):
            if iteration >= args.max_iterations:
                break
            if (iteration+1) % args.val_per_iters == 0:
                self.validation(iteration)

            # Move data to device
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Adjust learning rate
            if (iteration+1) % args.log_iter == 0:
                self.adjust_lr(optimizer, iteration, args)

            if (iteration+1) % args.log_iter == 0:
                logger.info('Iteration [%d/%d], Loss: %.4f, Time: %.2f'
                             % (iteration + 1, args.max_iterations, loss.item(), time.time() - start_time))

            if (iteration+1) % args.save_per_iters == 0:
                self.save_checkpoint(iteration)

    def validation(self, iteration):
        model = self.model
        val_loader = self.val_loader
        logger = self.logger

        model.eval()
        metric = SegmentationMetric(num_classes=val_loader.dataset.num_class)
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                metric.update(labels.cpu().numpy(), preds.cpu().numpy())

        pixel_acc, mIoU = metric.get()
        logger.info('Validation Result - Iteration [%d], Pixel Accuracy: %.4f, mIoU: %.4f'
                     % (iteration + 1, pixel_acc, mIoU))
        
        # Save the best model
        if mIoU > self.best_miou:
            self.best_miou = mIoU
            self.save_checkpoint(iteration, is_best=True)

    def save_checkpoint(self, iteration, is_best=False):
        state = {
            'iteration': iteration,
            'state_dict': self.model.state_dict(),
            'best_miou': self.best_miou,
        }
        filename = os.path.join(self.args.save_dir, 'checkpoint.pth')
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(self.args.save_dir, 'model_best.pth'))
        self.logger.info('Checkpoint saved to %s', filename)

if __name__ == '__main__':
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()
