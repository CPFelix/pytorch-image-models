#!/usr/bin/env python3
"""PyTorch Inference Script

An example inference script that outputs top-k class ids for images in a folder into a csv.

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import os
import time
import argparse
import logging
import numpy as np
import torch

from timm.models import create_model, apply_test_time_pool
from timm.data import ImageDataset, create_loader, resolve_data_config
from timm.utils import AverageMeter, setup_default_logging
from shutil import copyfile
import torch.nn as nn
from timm.data import create_dataset

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('inference')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
# 增加数据集名称，以调用根据txt读取数据集
parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--output_dir', metavar='DIR', default='./',
                    help='path to output files')
parser.add_argument('--model', '-m', metavar='MODEL', default='dpn92',
                    help='model architecture (default: dpn92)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='Number classes in dataset')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--no-test-pool', dest='no_test_pool', action='store_true',
                    help='disable test time pool')
parser.add_argument('--topk', default=5, type=int,
                    metavar='N', help='Top-k to output to CSV')
# 指定GPU
parser.add_argument('--gpus', type=str, default=None, metavar='N',
                    help='gpu ids to use, e.g. "0,1"')

def main():
    setup_default_logging()
    args = parser.parse_args()
    # might as well try to do something useful...
    args.pretrained = args.pretrained or not args.checkpoint

    # create model
    model = create_model(
        args.model,
        num_classes=args.num_classes,
        in_chans=3,
        pretrained=args.pretrained,
        checkpoint_path=args.checkpoint)

    _logger.info('Model %s created, param count: %d' %
                 (args.model, sum([m.numel() for m in model.parameters()])))

    config = resolve_data_config(vars(args), model=model)
    model, test_time_pool = (model, False) if args.no_test_pool else apply_test_time_pool(model, config)


    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
    # 指定GPU训练
    elif args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        device_ids = list(int(id) for id in (args.gpus.split(",")))
        model = torch.nn.DataParallel(model, device_ids=list(range(len(device_ids)))).cuda()
    else:
        model = model.cuda()

    dataset_eval = create_dataset(args.dataset, root=args.data, is_training=False, batch_size=args.batch_size)
    loader = create_loader(
        dataset_eval,
        input_size=config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=True,
        interpolation=config['interpolation'],
        mean=config['mean'],
        std=config['std'],
        num_workers=args.workers,
        crop_pct=1.0 if test_time_pool else config['crop_pct'])
    classmap = loader.dataset.class_map()
    print(classmap)
    
    model.eval()

    k = min(args.topk, args.num_classes)
    batch_time = AverageMeter()
    end = time.time()
    topk_ids = []
    topk_probs = []
    labels_ids = []

    dict_result = {"total": {"total_nums": 0, "right_nums": 0, "accuracy": 0.0}}
    for index in classmap.values():
        dict_result[index] = {"total_nums": 0, "right_nums": 0, "accuracy": 0.0}

    with torch.no_grad():
        for batch_idx, (input, labels) in enumerate(loader):
            input = input.cuda()
            preds = model(input)
            # if (batch_idx == 1235):
            #     print(preds)
            #     print("stop")
            topk = preds.topk(k)[1]
            topk_ids.append(topk.cpu().numpy())
            labels_ids.extend(labels.cpu().numpy())
            # 获取softmax概率值
            softmax_op = nn.Softmax(dim=1)
            prob_tensor = softmax_op(preds)
            # print(prob_tensor)
            topk_prob = prob_tensor.topk(k)[0]
            topk_probs.append(topk_prob.cpu().detach().numpy())
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.log_freq == 0:
                _logger.info('Predict: [{0}/{1}] Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                    batch_idx, len(loader)-1, batch_time=batch_time))
            # 计算总体和各个类别准确率
            dict_result["total"]["total_nums"] += input.shape[0]
            top1 = preds.topk(1)[1].squeeze(1)
            flag = top1 == labels
            dict_result["total"]["right_nums"] += flag.sum().item()

            for i, pred in enumerate(top1):
                if (pred == torch.tensor(0)):
                    dict_result[0]["total_nums"] += 1
                    if (pred == labels[i]):
                        dict_result[0]["right_nums"] += 1
                if (pred == torch.tensor(1)):
                    dict_result[1]["total_nums"] += 1
                    if (pred == labels[i]):
                        dict_result[1]["right_nums"] += 1
                if (pred == torch.tensor(2)):
                    dict_result[2]["total_nums"] += 1
                    if (pred == labels[i]):
                        dict_result[2]["right_nums"] += 1
                if (pred == torch.tensor(3)):
                    dict_result[3]["total_nums"] += 1
                    if (pred == labels[i]):
                        dict_result[3]["right_nums"] += 1
                if (pred == torch.tensor(4)):
                    dict_result[4]["total_nums"] += 1
                    if (pred == labels [i]):
                        dict_result[4]["right_nums"] += 1


    for key,value in dict_result.items():
        dict_result[key]["accuracy"] = float(value["right_nums"]) / value["total_nums"]
        if (isinstance(key, int)):
            classname = list(classmap.keys())[list(classmap.values()).index(key)]
        else:
            classname = key
        print(classname + ':')
        print(value)


    topk_ids = np.concatenate(topk_ids, axis=0)
    top1_ids = topk_ids[:, 0]
    topk_probs = np.concatenate(topk_probs, axis=0)
    top1_probs = topk_probs[:, 0]

    with open(os.path.join(args.output_dir, './topk_ids.csv'), 'w') as out_file:
        filenames = loader.dataset.filenames()
        for filename, label, pred in zip(filenames, labels_ids, topk_ids):
            out_file.write('{0},{1}, {2}\n'.format(
                filename, label, ','.join([ str(k) for k in pred])))

        # 保存分类错误图片
        save_path = os.path.join(args.output_dir, "results")
        for filename, label, pred, prob in zip(filenames, labels_ids, top1_ids, top1_probs):
            # print(filename)
            part_filename = os.path.join(filename.split("/")[-2], filename.split("/")[-1])
            if (part_filename == "another/sigurate_jjc_R_3384.jpg"):
                print(filename + " " + str(label) + " " + str(pred) + " " + str(prob))
            if (label != pred):
                label_classname = list(classmap.keys())[list(classmap.values()).index(label)]
                pred_classname = list(classmap.keys())[list(classmap.values()).index(pred)]

                dir = label_classname + "V" + pred_classname
                img_path = os.path.join(save_path, dir)
                if not os.path.exists(img_path):
                    os.makedirs(img_path)
                source_file = os.path.join(args.data, "val", filename)
                # source_file = filename
                target_file = os.path.join(img_path, filename.split("/")[-1])
                copyfile(source_file, target_file)


if __name__ == '__main__':
    main()
