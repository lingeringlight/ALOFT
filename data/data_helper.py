from os.path import join

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data.JigsawLoader import *
from data.concat_dataset import ConcatDataset
from data.JigsawLoader import JigsawNewDataset, JigsawTestNewDataset

from .samplers import BatchSchedulerSampler
from datasets import build_transform


vlcs_datasets = ["CALTECH", "LABELME", "PASCAL", "SUN"]
pacs_datasets = ["art_painting", "cartoon", "photo", "sketch"]
officehome_datasets = ['Art', 'Clipart', 'Product', 'RealWorld']
available_datasets = officehome_datasets + pacs_datasets + vlcs_datasets


class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, limit):
        indices = torch.randperm(len(dataset))[:limit]
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def get_train_dataloader(args, patches):
    dataset_list = args.source
    assert isinstance(dataset_list, list)
    datasets = []
    val_datasets = []

    if args.dataloader_DG_GFNet == 0:
        img_transformer, tile_transformer = get_train_transformers(args)
        img_transformer_val = get_val_transformer(args)
    else:
        tile_transformer = None
        img_transformer = build_transform(is_train=True, args=args, infer_no_resize=False)
        img_transformer_val = build_transform(is_train=False, args=args, infer_no_resize=False)

    limit = None

    if "PACS" in args.data_root:
        dataset_path = join(args.data_root, "kfold")
    elif args.data == "miniDomainNet":
        dataset_path = "/data/DataSets/" + "DomainNet"
    else:
        dataset_path = args.data_root

    for i, dname in enumerate(dataset_list):
        if args.data == "PACS":
            name_train, name_val, labels_train, labels_val, domain_labels_train, domain_labels_val = \
                get_split_dataset_info_from_txt(txt_path=join(args.data_root, "pacs_label"), domain=dname,
                                            domain_label=i+1)
                # get_split_dataset_info_from_txt(txt_path=join(args.data_root, "splits"), domain=dname,
                #                             domain_label=i + 1)
        elif args.data == "miniDomainNet":
            name_train, name_val, labels_train, labels_val, domain_labels_train, domain_labels_val = \
                get_split_dataset_info_from_txt(txt_path=args.data_root, domain=dname, domain_label=i+1,
                                                val_percentage=args.val_size)
        else:
            name_train, name_val, labels_train, labels_val, domain_labels_train, domain_labels_val = \
                get_split_domain_info_from_dir(join(dataset_path, dname), dataset_name=args.data,
                                               val_percentage=args.val_size, domain_label=i+1)

        # if args.RandAug_flag == 1:
        #     train_dataset = JigsawDatasetRandAug(name_train, labels_train, domain_labels_train,
        #                                          dataset_path=dataset_path, patches=patches,
        #                                          img_transformer=img_transformer,
        #                                          bias_whole_image=args.bias_whole_image, args=args)
        # elif args.Fourier_flag == 1:
        #     train_dataset = JigsawDatasetFourier(name_train, labels_train, domain_labels_train,
        #                                          dataset_path=dataset_path, img_transformer=img_transformer, args=args,
        #                                          dataset_list=dataset_list)
        # elif args.tobias_flag == 1:
        #     train_dataset = JigsawDatasetTobias(name_train, labels_train, domain_labels_train,
        #                                          dataset_path=dataset_path, img_transformer=img_transformer, args=args,
        #                                          dataset_list=dataset_list)
        # else:
        train_dataset = JigsawNewDataset(name_train, labels_train, domain_labels_train,
                                             dataset_path=dataset_path, patches=patches,
                                             img_transformer=img_transformer, tile_transformer=tile_transformer,
                                             jig_classes=30)
        if limit:
            train_dataset = Subset(train_dataset, limit)
        datasets.append(train_dataset)
        if args.freq_analyse == 1:
            val_datasets.append(
                JigsawTestDatasetFreqAnalyse(name_val, labels_val, domain_labels_val, dataset_path=dataset_path,
                                     img_transformer=img_transformer_val, args=args, dataset_list=dataset_list))
        else:
            val_datasets.append(
                JigsawTestNewDataset(name_val, labels_val, domain_labels_val, dataset_path=dataset_path,
                                    img_transformer=img_transformer_val, patches=patches, jig_classes=30))
    dataset = ConcatDataset(datasets)
    val_dataset = ConcatDataset(val_datasets)

    if args.domain_sampler == 1:
        sampler = BatchSchedulerSampler(dataset, args.batch_size)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                             pin_memory=True, drop_last=True, sampler=sampler)
    else:
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                         pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                             pin_memory=True, drop_last=False)
    return loader, val_loader


def get_val_dataloader(args, patches=False, tSNE_flag=0):
    if "PACS" in args.data_root:
        dataset_path = join(args.data_root, "kfold")
    elif args.data == "miniDomainNet":
        dataset_path = "/data/DataSets/" + "DomainNet"
    else:
        dataset_path = args.data_root

    if args.data == "miniDomainNet":
        name_train, name_val, labels_train, labels_val, domain_label_train, domain_label_val = \
            get_split_dataset_info_from_txt(txt_path=args.data_root, domain=args.target, domain_label=0,
                                            val_percentage=args.val_size)
    else:
        name_train, name_val, labels_train, labels_val, domain_label_train, domain_label_val = get_split_domain_info_from_dir(
            join(dataset_path, args.target), dataset_name=args.data, val_percentage=args.val_size, domain_label=0)

    if tSNE_flag == 0:
        names = name_train + name_val
        labels = labels_train + labels_val
        domain_label = domain_label_train + domain_label_val
    else:
        names = name_val
        labels = labels_val
        domain_label = domain_label_val

    img_tr = get_val_transformer(args)
    dataset_list = args.source
    if args.freq_analyse == 1:
        val_dataset = JigsawTestDatasetFreqAnalyse(names, labels, domain_label, dataset_path=dataset_path,
                                           img_transformer=img_tr, args=args, dataset_list=dataset_list)
    else:
        val_dataset = JigsawTestNewDataset(names, labels, domain_label, dataset_path=dataset_path, patches=patches,
                                       img_transformer=img_tr, jig_classes=30)

    # if args.limit_target and len(val_dataset) > args.limit_target:
    #     val_dataset = Subset(val_dataset, args.limit_target)
    #     print("Using %d subset of val dataset" % args.limit_target)

    dataset = ConcatDataset([val_dataset])
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                         pin_memory=True, drop_last=False)
    return loader


def get_train_transformers(args):

    img_tr = [transforms.RandomResizedCrop((int(args.image_size), int(args.image_size)), (args.min_scale, args.max_scale))]
    if args.random_horiz_flip > 0.0:
        img_tr.append(transforms.RandomHorizontalFlip(args.random_horiz_flip))
    if args.jitter > 0.0:
        img_tr.append(transforms.ColorJitter(brightness=args.jitter, contrast=args.jitter, saturation=args.jitter, hue=min(0.5, args.jitter)))

    # this is special operation for JigenDG
    if args.gray_flag:
        img_tr.append(transforms.RandomGrayscale(args.tile_random_grayscale))

    img_tr.append(transforms.ToTensor())
    img_tr.append(transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    tile_tr = []
    if args.tile_random_grayscale:
        tile_tr.append(transforms.RandomGrayscale(args.tile_random_grayscale))
    tile_tr = tile_tr + [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    return transforms.Compose(img_tr), transforms.Compose(tile_tr)


def get_val_transformer(args):
    img_tr = [
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    return transforms.Compose(img_tr)


# def get_train_dataloader_RandAug(args, patches):
#     dataset_list = args.source
#     assert isinstance(dataset_list, list)
#     datasets = []
#     val_datasets = []
#     img_transformer, tile_transformer = get_train_transformers(args)
#     limit = args.limit_source
#     for dname in dataset_list:
#         name_train, labels_train = _dataset_info(join('/data/DataSets/PACS', 'pacs_label', '%s_train_kfold.txt' % dname))
#         name_val, labels_val = _dataset_info(join('/data/DataSets/PACS', 'pacs_label', '%s_crossval_kfold.txt' % dname))
#
#         train_dataset = JigsawDatasetRandAug(name_train, labels_train, patches=patches, img_transformer=img_transformer,
#                                      bias_whole_image=args.bias_whole_image, args=args)
#         if limit:
#             train_dataset = Subset(train_dataset, limit)
#         datasets.append(train_dataset)
#         val_datasets.append(
#             JigsawTestDatasetRandAug(name_val, labels_val, img_transformer=get_val_transformer(args),
#                               patches=patches, args=args))
#     dataset = ConcatDataset(datasets)
#     val_dataset = ConcatDataset(val_datasets)
#     loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
#     val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
#     return loader, val_loader


# def get_val_dataloader_RandAug(args, patches=False):
#     names, labels = _dataset_info(join('/data/DataSets/PACS', 'pacs_label', '%s_test_kfold.txt' % args.target))
#     img_tr = get_val_transformer(args)
#     val_dataset = JigsawTestDatasetRandAug(names, labels, patches=patches, img_transformer=img_tr, args=args)
#     if args.limit_target and len(val_dataset) > args.limit_target:
#         val_dataset = Subset(val_dataset, args.limit_target)
#         print("Using %d subset of val dataset" % args.limit_target)
#     dataset = ConcatDataset([val_dataset])
#     loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
#     return loader
