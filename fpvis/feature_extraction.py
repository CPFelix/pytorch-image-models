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
from timm.data.transforms_factory import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from shutil import copyfile
from PIL import Image
from misc_functions import get_example_params, save_class_activation_images
from torchvision import transforms
import torch.nn as nn
import math
from tqdm import tqdm

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('inference')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
parser.add_argument('--output_dir', metavar='DIR', default='./',
                    help='path to output files')
parser.add_argument('--model', '-m', metavar='MODEL', default='dpn92',
                    help='model architecture (default: dpn92)')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N',
                    help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
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

class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        netName = self.model.__class__.__name__
        if (netName == "AlexNet"):
            for module_pos, module in self.model.features._modules.items():
                x = module(x)  # Forward
                if int(module_pos) == self.target_layer:
                    x.register_hook(self.save_gradient)
                    conv_output = x  # Save the convolution output on that layer
        elif (netName == "ResNet"):
            for module_pos, module in self.model._modules.items():
                x = module(x)  # Forward
                if str(module_pos) == self.target_layer:
                    x.register_hook(self.save_gradient)
                    conv_output = x  # Save the convolution output on that layer
        elif (netName == "EfficientNet"):
            for module_pos, module in self.model._modules.items():
                x = module(x)  # Forward
                if str(module_pos) == self.target_layer:
                    x.register_hook(self.save_gradient)
                    conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        if (self.model.__class__.__name__ == "AlexNet"):
            x = self.model.classifier(x)
        return conv_output, x


class LayerCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Zero grads
        if (self.model.__class__.__name__ == "AlexNet"):
            # Target for backprop
            one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
            one_hot_output[0][target_class] = 1
            self.model.features.zero_grad()
            self.model.classifier.zero_grad()
            # Backward pass with specified target
            model_output.backward(gradient=one_hot_output, retain_graph=True)
            # Get hooked gradients
            guided_gradients = self.extractor.gradients.data.numpy()[0]
            # Get convolution outputs
            target = conv_output.data.numpy()[0]
            # Get weights from gradients
            weights = guided_gradients
            weights[weights < 0] = 0  # discard negative gradients
            # Element-wise multiply the weight with its conv output and then, sum
            cam = np.sum(weights * target, axis=0)
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
            cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
            cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                                                        input_image.shape[3]), Image.ANTIALIAS)) / 255
        elif (self.model.__class__.__name__ == "ResNet"):
            # Target for backprop
            one_hot_output = torch.cuda.FloatTensor(1, model_output.size()[-1]).zero_()
            one_hot_output[0][target_class] = 1
            self.model.zero_grad()
            # Backward pass with specified target
            model_output.backward(gradient=one_hot_output, retain_graph=True)
            # Get hooked gradients
            guided_gradients = self.extractor.gradients.data.cpu().numpy()[0]
            # Get convolution outputs
            target = conv_output.data.cpu().numpy()[0]
            # Get weights from gradients
            weights = guided_gradients
            weights[weights < 0] = 0  # discard negative gradients
            # Element-wise multiply the weight with its conv output and then, sum
            cam = np.sum(weights * target, axis=0)
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
            cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
            cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                                                        input_image.shape[3]), Image.ANTIALIAS)) / 255

        elif (self.model.__class__.__name__ == "EfficientNet"):
            # Target for backprop
            one_hot_output = torch.cuda.FloatTensor(1, model_output.size()[-1]).zero_()
            one_hot_output[0][target_class] = 1
            self.model.zero_grad()
            # Backward pass with specified target
            model_output.backward(gradient=one_hot_output, retain_graph=True)
            # Get hooked gradients
            guided_gradients = self.extractor.gradients.data.cpu().numpy()[0]
            # Get convolution outputs
            target = conv_output.data.cpu().numpy()[0]
            # Get weights from gradients
            weights = guided_gradients
            weights[weights < 0] = 0  # discard negative gradients
            # Element-wise multiply the weight with its conv output and then, sum
            cam = np.sum(weights * target, axis=0)
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
            cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
            cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                                                        input_image.shape[3]), Image.ANTIALIAS)) / 255

        return cam

def vgg_vis():
    # vgg example
    # Get params
    target_example = 1  # Snake
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) = \
        get_example_params(target_example)

    # pretrained_model = pretrained_model.cuda()
    # Layer cam
    layer_cam = LayerCam(pretrained_model, target_layer=12)
    # Generate cam mask
    cam = layer_cam.generate_cam(prep_img, target_class)
    # Save mask
    save_class_activation_images(original_image, cam, file_name_to_export)
    print('Layer cam completed')

def timm_vis():
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
    # print(config)
    model, test_time_pool = (model, False) if args.no_test_pool else apply_test_time_pool(model, config)

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
    # 指定GPU训练
    elif args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        model = model.cuda()
    else:
        model = model.cuda()

    model.eval()

    k = min(args.topk, args.num_classes)
    end = time.time()
    topk_ids = []
    topk_probs = []

    # DEBUG和RUN模式下取值不同
    # crop_pct = 1.0 if test_time_pool else config['crop_pct']
    # print(config['crop_pct'])
    # print("crop_pct: ", crop_pct)

    transform = create_transform(
        config['input_size'],
        is_training=False,
        use_prefetcher=True,
        no_aug=False,
        scale=None,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        color_jitter=0.4,
        auto_augment=None,
        interpolation='bicubic',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        crop_pct=0.95,
        tf_preprocessing=False,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_num_splits=0,
        separate=False,
    )

    # path = "/home/chenpengfei/Timm_remote/pytorch-image-models/output/train/20211228-202210-resnet50-128/results/sigurate-another"
    # path = "/home/chenpengfei/Timm_remote/pytorch-image-models/output/train/20211228-202210-resnet50-128/results/another-sigurate"
    path = "/home/chenpengfei/dataset/DSMhands7/val/fake_sigurate"
    dir = path.split("/")[-1]
    for imgname in tqdm(os.listdir(path)):
        imgPath = os.path.join(path, imgname)

        img = Image.open(imgPath).convert('RGB')
        input = transform(img)

        tensor = torch.zeros((1, *input.shape), dtype=torch.uint8)
        tensor[0] += torch.from_numpy(input)
        mean = torch.tensor([x * 255 for x in IMAGENET_DEFAULT_MEAN]).view(1, 3, 1, 1)
        std = torch.tensor([x * 255 for x in IMAGENET_DEFAULT_STD]).view(1, 3, 1, 1)
        input = tensor.float().sub_(mean).div_(std)
        # print(input)
        input = input.cuda()
        preds = model(input)
        # print(preds)
        topk = preds.topk(k)[1]
        topk_ids.append(topk.cpu().numpy())
        # {'another': 0, 'phone': 1, 'sigurate': 2}
        top1 = preds.topk(1)[1][0][0].cpu().numpy()
        # print("top1: ", top1)

        # 获取softmax概率值
        softmax_op = nn.Softmax(dim=1)
        prob_tensor = softmax_op(preds)
        # print(prob_tensor)
        topk_prob = prob_tensor.topk(k)[0]
        topk_probs.append(topk_prob.cpu().detach().numpy())
        top1_prob = prob_tensor.topk(1)[0][0][0].cpu().detach().numpy()
        # print("top1 prob: ", top1_prob)

        # target_layer = "layer4"
        target_layer = "conv_head"
        layer_cam = LayerCam(model, target_layer=target_layer)
        # Generate cam mask
        target_class = 1
        cam = layer_cam.generate_cam(input, target_class)

        # Save mask
        tfl_img = Image.open(imgPath).convert('RGB')
        img_size = [128, 128]
        crop_pct = 1.0
        scale_size = int(math.floor(img_size[0] / crop_pct))
        tfl = [
            # transforms.Resize(scale_size, Image.BILINEAR),
            # transforms.CenterCrop(img_size),
            transforms.Resize(img_size, Image.BILINEAR),
        ]
        model_name = "20211230-183650-mobilenetv2_100-128"
        for t in tfl:
            tfl_img = t(tfl_img)
        tfl_img_path = "./output/temp/" + model_name + "/" + dir + "/"
        if not os.path.exists(tfl_img_path):
            os.makedirs(tfl_img_path)
        partImgName = imgPath.split("/")[-1].split(".")[0]
        tfl_img_name = tfl_img_path + partImgName + "_tfl.jpg"
        tfl_img.save(tfl_img_name, quality=100)

        # 保存特征图
        fp_path = './output/fpvis/' + model_name + "/" + dir + "/"
        if not os.path.exists(fp_path):
            os.makedirs(fp_path)
        file_name_to_export = partImgName + "_" + str(args.model) + "_" + target_layer + "_" + str(top1) + "_" + str(top1_prob)
        save_class_activation_images(tfl_img, cam, fp_path, file_name_to_export)
        # print('Layer cam completed')

def main():
    timm_vis()
    # vgg_vis()


if __name__ == '__main__':
    main()
