import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer_v2, create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma
from functools import partial
import torch.nn as nn

from engine import train_one_epoch, evaluate
from losses import DistillationLoss
import utils
from gfnet import GFNet, GFNetPyramid

from data import data_helper
import os

from torch import optim


import warnings
warnings.filterwarnings("ignore", message="Argument interpolation should be")

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)

    parser.add_argument("--target", default=2, type=int, help="Target")
    parser.add_argument("--device", type=int, default=3, help="GPU num")

    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=50, type=int)

    parser.add_argument('--perturb_prob', default=0.5,  type=float)
    parser.add_argument('--mask_alpha', default=0.2,  type=float)

    parser.add_argument('--noise_mode', default=1,  type=int, help="0: close; 1: add noise")
    parser.add_argument('--uncertainty_model', default=2,  type=int, help="1:batch+mean 2:batch+element")
    parser.add_argument('--uncertainty_factor', default=1.0,  type=float)
    parser.add_argument('--mask_radio', default=0.5,  type=float)
    parser.add_argument('--gauss_or_uniform', default=0,  type=int, help="0: gaussian; 1: uniform; 2: random")
    parser.add_argument('--noise_layers', default=[0, 1, 2, 3],  nargs="+", type=int, help="where to use augmentation.")

    parser.add_argument('--set_training_mode', default=1,  type=int, help="0:eval 1:train")
    parser.add_argument('--freq_analyse', default=0, type=int, help="whether do frequency analyse")

    parser.add_argument('--data_root', default='/data/DataSets/', type=str, help='dataset path')
    parser.add_argument('--output_dir', default='/ALOFT_results/', help='path where to save, empty for no saving')

    # * Finetuning params
    parser.add_argument('--finetune', default='/pretrained_model/', help='finetune from checkpoint')

    # Model parameters
    parser.add_argument('--arch', default='gfnet-h-ti', type=str,
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--model-ema', default=False)
    parser.set_defaults(model_ema=False)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=1, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.set_defaults(repeated_aug=False)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--eval', default=0, type=int, help='Perform evaluation only')

    parser.add_argument('--data', default='PACS', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19', 'PACS', 'OfficeHome', 'VLCS', 'digits_dg'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--gray_flag", default=0, type=int, help="whether use random gray")
    parser.add_argument("--random_horiz_flip", default=0.5, type=float, help="Chance of random horizontal flip")
    parser.add_argument("--jitter", default=0.4, type=float, help="Color jitter amount")
    parser.add_argument("--tile_random_grayscale", default=0.1, type=float,
                        help="Chance of randomly greyscaling a tile")

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    # parser.add_argument('--eval', action='store_true', help='Perform evaluation only')

    parser.add_argument('--dist-eval', action='store_true', default=True, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


domain_map = {
    'PACS': ['photo', 'art_painting', 'cartoon', 'sketch'],
    'PACS_random_split': ['photo', 'art_painting', 'cartoon', 'sketch'],
    'OfficeHome': ['Art', 'Clipart', 'Product', 'RealWorld'],
    'VLCS': ["CALTECH", "LABELME", "PASCAL", "SUN"],
    'digits_dg': ['mnist', 'mnist_m', 'svhn', 'syn'],
}
classes_map = {
    'PACS': 7,
    'PACS_random_split': 7,
    'OfficeHome': 65,
    'VLCS': 5,
    'digits_dg': 32,
}
val_size_map = {
    'PACS': 0.1,
    'PACS_random_split': 0.1,
    'OfficeHome': 0.1,
    'VLCS': 0.3,
    'digits_dg': 0.2,
}


def get_domain(name):
    if name not in domain_map:
        raise ValueError('Name of dataset unknown %s' %name)
    return domain_map[name]


def main(args):
    utils.init_distributed_mode(args)

    domain = get_domain(args.data)
    args.target = domain.pop(args.target)
    args.source = domain
    print("Target domain: {}".format(args.target))
    args.data_root = os.path.join(args.data_root, "PACS") if "PACS" in args.data else os.path.join(args.data_root,
                                                                                                   args.data)
    args.n_classes = classes_map[args.data]
    args.n_domains = len(domain)
    args.val_size = val_size_map[args.data]

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    args.nb_classes = args.n_classes

    data_loader_train, data_loader_val = data_helper.get_train_dataloader(args, patches=False)
    data_loader_test = data_helper.get_val_dataloader(args, patches=False)

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print('standard mix up')
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    else:
        print('mix up is not used')

    print(f"Creating model: {args.arch}")

    if args.arch == 'gfnet-h-ti':
        model = GFNetPyramid(
            img_size=args.input_size, 
            patch_size=4,
            num_classes=args.n_classes,
            embed_dim=[64, 128, 256, 512], depth=[3, 3, 10, 3],
            mlp_ratio=[4, 4, 4, 4],
            norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.1,
            mask_radio=args.mask_radio,
            mask_alpha=args.mask_alpha,
            noise_mode=args.noise_mode,
            uncertainty_model=args.uncertainty_model, perturb_prob=args.perturb_prob,
            noise_layers=args.noise_layers, gauss_or_uniform=args.gauss_or_uniform,
        )
    elif args.arch == 'gfnet-h-s':
        model = GFNetPyramid(
            img_size=args.input_size, 
            patch_size=4,
            num_classes=args.n_classes,
            embed_dim=[96, 192, 384, 768], depth=[3, 3, 10, 3],
            mlp_ratio=[4, 4, 4, 4],
            norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.2,
            init_values=1e-5,
            mask_radio=args.mask_radio,
            mask_alpha=args.mask_alpha,
            noise_mode=args.noise_mode,
            uncertainty_model=args.uncertainty_model, perturb_prob=args.perturb_prob,
            noise_layers=args.noise_layers, gauss_or_uniform=args.gauss_or_uniform,
        )
    else:
        raise NotImplementedError


    if args.finetune:
        args.finetune += "/" + args.arch + ".pth"

        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]

        if args.arch in ['gfnet-ti', 'gfnet-xs', 'gfnet-s', 'gfnet-b']:
            num_patches = (args.input_size // 16) ** 2
        elif args.arch in ['gfnet-h-ti', 'gfnet-h-s', 'gfnet-h-b']:
            num_patches = (args.input_size // 4) ** 2
        else:
            raise NotImplementedError

        num_extra_tokens = 0
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)

        scale_up_ratio = new_size / orig_size
        # class_token and dist_token are kept unchanged
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        checkpoint_model['pos_embed'] = pos_tokens

        for name in checkpoint_model.keys():
            if 'complex_weight' in name:
                h, w, num_heads = checkpoint_model[name].shape[0:3] # h, w, c, 2
                origin_weight = checkpoint_model[name]
                upsample_h = h * new_size // orig_size
                upsample_w = upsample_h // 2 + 1
                origin_weight = origin_weight.reshape(1, h, w, num_heads * 2).permute(0, 3, 1, 2)
                new_weight = torch.nn.functional.interpolate(
                    origin_weight, size=(upsample_h, upsample_w), mode='bicubic', align_corners=True).permute(0, 2, 3, 1).reshape(upsample_h, upsample_w, num_heads, 2)
                checkpoint_model[name] = new_weight
        model.load_state_dict(checkpoint_model, strict=False)

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if args.opt == "adamw":
        linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
        args.lr = linear_scaled_lr
        optimizer = create_optimizer(args, model_without_ddp)
        lr_scheduler, _ = create_scheduler(args, optimizer)
    elif args.opt == "sgd":
        optimizer = create_optimizer_v2(model_without_ddp, opt='sgd', lr=args.lr, weight_decay=0.0005, momentum=0.9)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs * 0.8, gamma=0.1)

    loss_scaler = NativeScaler()
    criterion = LabelSmoothingCrossEntropy()

    if args.mixup > 0. or args.cutmix > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool='avg',
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'

    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )

    dir_name = args.opt + str(args.lr) + "E" + str(args.epochs)

    if args.arch == "gfnet-h-s":
        dir_name += "_gfnet-h-s"

    if args.noise_mode != 0:
        dir_name += "_noise" + "_M" + str(args.mask_radio) + "_p" + str(args.perturb_prob)
        if args.noise_mode == 1:
            dir_name += "_amp"
            if args.uncertainty_model != 0:
                if args.uncertainty_model == 1:
                    dir_name += "_batch_mean"
                else:
                    dir_name += "_batch_elem"
                if args.gauss_or_uniform == 1:
                    dir_name += "_unif"
                elif args.gauss_or_uniform == 2:
                    dir_name += "_randGauss"
                dir_name += "_f" + str(args.uncertainty_factor)

    if args.gray_flag == 0:
        dir_name += "_nogray"

    output_dir = os.path.join(args.output_dir, args.data, dir_name, args.target + str(args.seed))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir = Path(output_dir)

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            model_path = args.resume + "/" + args.target + str(args.seed) + "/checkpoint_last.pth"
            checkpoint = torch.load(model_path, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('lr scheduler will not be updated')
            # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])

    if args.eval:
        val_stats = evaluate(data_loader_val, model, device)['acc1']
        test_stats = evaluate(data_loader_test, model, device)['acc1']
        print(f"Accuracy of the network on the {len(data_loader_val.dataset)} val images: {val_stats:.2f}%")
        print(f"Accuracy of the network on the {len(data_loader_test.dataset)} test images: {test_stats:.2f}%")
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy_test = 0.0
    max_accuracy_val = 0.0
    max_val_test = 0.0
    max_val_epoch = 0
    max_test_epoch = 0

    if args.set_training_mode == 1:
        args.set_training_mode = True
    else:
        args.set_training_mode = False

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode=args.set_training_mode,  # keep in eval mode during finetuning
        )

        lr_scheduler.step(epoch)

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint_last.pth']
            for checkpoint_path in checkpoint_paths:
                if model_ema is not None:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'model_ema': get_state_dict(model_ema),
                    }, checkpoint_path)
                else:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                    }, checkpoint_path)
        
        if (epoch + 1) % 100 == 0:
            file_name = 'checkpoint_epoch%d.pth' % epoch
            checkpoint_path = output_dir / file_name
            if model_ema is not None:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'model_ema': get_state_dict(model_ema),
                }, checkpoint_path)
            else:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                }, checkpoint_path)

        val_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(data_loader_val.dataset)} val images: {val_stats['acc1']:.2f}%")
        test_stats = evaluate(data_loader_test, model, device)
        print(f"Accuracy of the network on the {len(data_loader_test.dataset)} test images: {test_stats['acc1']:.2f}%")

        max_accuracy_val = max(max_accuracy_val, val_stats["acc1"])
        print(f'Max accuracy val: {max_accuracy_val:.2f}%')
        print(f"Corresponding test accuracy: {test_stats['acc1']:.2f}%")
        max_accuracy_test = max(max_accuracy_test, test_stats["acc1"])
        if max_accuracy_val == test_stats["acc1"]:
            max_test_epoch = epoch
        # print(f'Max accuracy test: {max_accuracy_test:.2f}%')

        if max_accuracy_val == val_stats["acc1"]:
            max_val_test = test_stats['acc1']
            max_val_epoch = epoch

            checkpoint_path = output_dir / 'checkpoint_best.pth'
            if model_ema is not None:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'model_ema': get_state_dict(model_ema),
                }, checkpoint_path)
            else:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                }, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'val_{k}': v for k, v in val_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    log_stats = {**{f'Best val': max_accuracy_val},
                 **{f'Corresponding test': max_val_test},
                 **{f'At Epoch': max_val_epoch},
                 **{f'Best test': max_accuracy_test},
                 **{f'At Epoch': max_test_epoch},
                }
    with (output_dir / "log.txt").open("a") as f:
        f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('GFNet training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # torch.autograd.set_detect_anomaly(True)
    main(args)
