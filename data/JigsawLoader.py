import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from random import sample, random
import sys
import os


def get_random_subset(names, labels, percent):
    """

    :param names: list of names
    :param labels:  list of labels
    :param percent: 0 < float < 1
    :return:
    """
    samples = len(names)
    amount = int(samples * percent)
    random_index = sample(range(samples), amount)
    name_val = [names[k] for k in random_index]
    name_train = [v for k, v in enumerate(names) if k not in random_index]
    labels_val = [labels[k] for k in random_index]
    labels_train = [v for k, v in enumerate(labels) if k not in random_index]
    return name_train, name_val, labels_train, labels_val


def _dataset_info(txt_labels):
    # read from the official split txt
    file_names = []
    labels = []

    # with open(txt_labels, 'r') as f:
    #     images_list = f.readlines()
    #     for row in images_list:
    #         row = row.split(' ')
    #         file_names.append(row[0])
    #         labels.append(int(row[1]))

    for row in open(txt_labels, 'r'):
        row = row.split(' ')
        file_names.append(row[0])
        labels.append(int(row[1]))

    return file_names, labels


def find_classes(dir_name):
    if sys.version_info >= (3, 5):
        # Faster and available in Python 3.5 and above
        classes = [d.name for d in os.scandir(dir_name) if d.is_dir()]
    else:
        classes = [d for d in os.listdir(dir_name) if os.path.isdir(os.path.join(dir_name, d))]
    classes.sort()
    class_to_idx = {classes[i]: i+1 for i in range(len(classes))}
    return classes, class_to_idx


def get_split_domain_info_from_dir(domain_path, dataset_name=None, val_percentage=None, domain_label=None):
    # read from the directory
    domain_name = domain_path.split("/")[-1]
    if dataset_name == "VLCS":
        name_train, name_val, labels_train, labels_val = [], [], [], []
        classes, class_to_idx = find_classes(domain_path + "/full")
        # full为train
        for i, item in enumerate(classes):
            class_path = domain_path + "/" + "full" + "/" + item
            for root, _, fnames in sorted(os.walk(class_path)):
                for fname in sorted(fnames):
                    path = os.path.join(domain_name, "full", item, fname)
                    name_train.append(path)
                    labels_train.append(class_to_idx[item])
        # test为val
        for i, item in enumerate(classes):
            class_path = domain_path + "/" + "test" + "/" + item
            for root, _, fnames in sorted(os.walk(class_path)):
                for fname in sorted(fnames):
                    path = os.path.join(domain_name, "test", item, fname)
                    name_val.append(path)
                    labels_val.append(class_to_idx[item])
        domain_label_train = [domain_label for i in range(len(labels_train))]
        domain_label_val = [domain_label for i in range(len(labels_val))]
        return name_train, name_val, labels_train, labels_val, domain_label_train, domain_label_val

    elif dataset_name == "digits_dg":
        name_train, name_val, labels_train, labels_val = [], [], [], []
        classes, class_to_idx = find_classes(domain_path + "/train")
        # train
        for i, item in enumerate(classes):
            class_path = domain_path + "/" + "train" + "/" + item
            for root, _, fnames in sorted(os.walk(class_path)):
                for fname in sorted(fnames):
                    path = os.path.join(domain_name, "train", item, fname)
                    name_train.append(path)
                    labels_train.append(class_to_idx[item])
        # val
        for i, item in enumerate(classes):
            class_path = domain_path + "/" + "val" + "/" + item
            for root, _, fnames in sorted(os.walk(class_path)):
                for fname in sorted(fnames):
                    path = os.path.join(domain_name, "val", item, fname)
                    name_val.append(path)
                    labels_val.append(class_to_idx[item])

        # names = name_train + name_val
        # labels = labels_train + labels_val
        # name_train, name_val, labels_train, labels_val = get_random_subset(names, labels, val_percentage)

        domain_label_train = [domain_label for i in range(len(labels_train))]
        domain_label_val = [domain_label for i in range(len(labels_val))]
        return name_train, name_val, labels_train, labels_val, domain_label_train, domain_label_val

    elif dataset_name == "OfficeHome" or "PACS" in dataset_name:
        names, labels = [], []
        classes, class_to_idx = find_classes(domain_path)
        for i, item in enumerate(classes):
            class_path = domain_path + "/" + item
            for root, _, fnames in sorted(os.walk(class_path)):
                for fname in sorted(fnames):
                    path = os.path.join(domain_name, item, fname)
                    names.append(path)
                    labels.append(class_to_idx[item])
        name_train, name_val, labels_train, labels_val = get_random_subset(names, labels, val_percentage)
        domain_label_train = [domain_label for i in range(len(labels_train))]
        domain_label_val = [domain_label for i in range(len(labels_val))]
        return name_train, name_val, labels_train, labels_val, domain_label_train, domain_label_val

    else:
        raise ValueError("dataset is wrong.")


def get_split_dataset_info_from_txt(txt_path, domain, domain_label, val_percentage=None):
    if "PACS" in txt_path:
        train_name = "_train_kfold.txt"
        val_name = "_crossval_kfold.txt"

        train_txt = txt_path + "/" + domain + train_name
        val_txt = txt_path + "/" + domain + val_name

        train_names, train_labels = _dataset_info(train_txt)
        val_names, val_labels = _dataset_info(val_txt)
        train_domain_labels = [domain_label for i in range(len(train_labels))]
        val_domain_labels = [domain_label for i in range(len(val_labels))]
        return train_names, val_names, train_labels, val_labels, train_domain_labels, val_domain_labels

    elif "miniDomainNet" in txt_path:
        # begin at 0, need to add 1
        train_name = "_train.txt"
        val_name = "_test.txt"
        train_txt = txt_path + "/" + domain + train_name
        val_txt = txt_path + "/" + domain + val_name

        train_names, train_labels = _dataset_info(train_txt)
        val_names, val_labels = _dataset_info(val_txt)
        train_labels = [label + 1 for label in train_labels]
        val_labels = [label + 1 for label in val_labels]

        names = train_names + val_names
        labels = train_labels + val_labels
        train_names, val_names, train_labels, val_labels = get_random_subset(names, labels, val_percentage)

        train_domain_labels = [domain_label for i in range(len(train_labels))]
        val_domain_labels = [domain_label for i in range(len(val_labels))]
        return train_names, val_names, train_labels, val_labels, train_domain_labels, val_domain_labels
    else:
        raise NotImplementedError


def get_split_dataset_info(txt_list, val_percentage):
    names, labels = _dataset_info(txt_list)
    return get_random_subset(names, labels, val_percentage)


# 原始Jigsaw
class JigsawDataset(data.Dataset):
    def __init__(self, names, labels, jig_classes=100, img_transformer=None, tile_transformer=None, patches=True, bias_whole_image=None):
        self.data_path = ""
        self.names = names
        self.labels = labels

        self.N = len(self.names)
        self.permutations = self.__retrieve_permutations(jig_classes)
        self.grid_size = 3
        self.bias_whole_image = bias_whole_image
        if patches:
            self.patch_size = 64
        self._image_transformer = img_transformer
        self._augment_tile = tile_transformer
        if patches:
            self.returnFunc = lambda x: x
        else:
            def make_grid(x):
                return torchvision.utils.make_grid(x, self.grid_size, padding=0)
            self.returnFunc = make_grid

    def get_tile(self, img, n):
        w = float(img.size[0]) / self.grid_size
        y = int(n / self.grid_size)
        x = n % self.grid_size
        tile = img.crop([x * w, y * w, (x + 1) * w, (y + 1) * w])
        tile = self._augment_tile(tile)
        return tile

    def get_image(self, index):
        framename = self.data_path + '/' + self.names[index]
        img = Image.open(framename).convert('RGB')
        return self._image_transformer(img)

    def __getitem__(self, index):
        img = self.get_image(index)
        n_grids = self.grid_size ** 2
        tiles = [None] * n_grids
        for n in range(n_grids):
            tiles[n] = self.get_tile(img, n)

        order = np.random.randint(len(self.permutations) + 1)  # added 1 for class 0: unsorted
        if self.bias_whole_image:
            if self.bias_whole_image > random():
                order = 0
        if order == 0:
            data = tiles
        else:
            data = [tiles[self.permutations[order - 1][t]] for t in range(n_grids)]

        data = torch.stack(data, 0)
        return self.returnFunc(data), int(order), int(self.labels[index])

    def __len__(self):
        return len(self.names)

    def __retrieve_permutations(self, classes):
        all_perm = np.load('permutations_%d.npy' % (classes))
        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1

        return all_perm


class JigsawTestDataset(JigsawDataset):
    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)

    def __getitem__(self, index):
        framename = self.data_path + '/' + self.names[index]
        img = Image.open(framename).convert('RGB')
        return self._image_transformer(img), 0, int(self.labels[index])


class JigsawTestDatasetMultiple(JigsawDataset):
    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)
        self._image_transformer = transforms.Compose([
            transforms.Resize(255, Image.BILINEAR),
        ])
        self._image_transformer_full = transforms.Compose([
            transforms.Resize(225, Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self._augment_tile = transforms.Compose([
            transforms.Resize((75, 75), Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        framename = self.data_path + '/' + self.names[index]
        _img = Image.open(framename).convert('RGB')
        img = self._image_transformer(_img)

        w = float(img.size[0]) / self.grid_size
        n_grids = self.grid_size ** 2
        images = []
        jig_labels = []
        tiles = [None] * n_grids
        for n in range(n_grids):
            y = int(n / self.grid_size)
            x = n % self.grid_size
            tile = img.crop([x * w, y * w, (x + 1) * w, (y + 1) * w])
            tile = self._augment_tile(tile)
            tiles[n] = tile
        for order in range(0, len(self.permutations)+1, 3):
            if order==0:
                data = tiles
            else:
                data = [tiles[self.permutations[order-1][t]] for t in range(n_grids)]
            data = self.returnFunc(torch.stack(data, 0))
            images.append(data)
            jig_labels.append(order)
        images = torch.stack(images, 0)
        jig_labels = torch.LongTensor(jig_labels)
        return images, jig_labels, int(self.labels[index])


class JigsawNewDataset(data.Dataset):
    def __init__(self, names, labels, domain_labels, dataset_path, jig_classes=100, img_transformer=None,
                 tile_transformer=None, patches=True, bias_whole_image=None):
        self.data_path = dataset_path

        self.names = names
        self.labels = labels
        self.domain_labels = domain_labels

        self.N = len(self.names)
        # self.permutations = self.__retrieve_permutations(jig_classes)
        self.grid_size = 3
        self.bias_whole_image = bias_whole_image
        if patches:
            self.patch_size = 64
        self._image_transformer = img_transformer
        self._augment_tile = tile_transformer
        if patches:
            self.returnFunc = lambda x: x
        else:
            def make_grid(x):
                return torchvision.utils.make_grid(x, self.grid_size, padding=0)

            self.returnFunc = make_grid

    def get_tile(self, img, n):
        w = float(img.size[0]) / self.grid_size
        y = int(n / self.grid_size)
        x = n % self.grid_size
        tile = img.crop([x * w, y * w, (x + 1) * w, (y + 1) * w])
        tile = self._augment_tile(tile)
        return tile

    def get_image(self, index):
        framename = self.data_path + '/' + self.names[index]
        img = Image.open(framename).convert('RGB')
        return self._image_transformer(img)

    def __getitem__(self, index):
        framename = self.data_path + '/' + self.names[index]
        img = Image.open(framename).convert('RGB')
        # image, image_randaug, label, domain
        # return self._image_transformer(img), 0, int(self.labels[index] - 1), int(self.domain_labels[index] - 1)
        return self._image_transformer(img), int(self.labels[index] - 1)

    def __len__(self):
        return len(self.names)

    def __retrieve_permutations(self, classes):
        all_perm = np.load('permutations_%d.npy' % (classes))
        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1
        return all_perm


class JigsawTestNewDataset(JigsawNewDataset):
    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)

    def __getitem__(self, index):
        framename = self.data_path + '/' + self.names[index]
        img = Image.open(framename).convert('RGB')
        # return self._image_transformer(img), 0, int(self.labels[index] - 1), int(self.domain_labels[index] - 1)
        return self._image_transformer(img), int(self.labels[index] - 1)


# from .randaug import RandAugment
#
# class JigsawDatasetRandAug(data.Dataset):
#     def __init__(self, names, labels, domain_labels, patches=True, dataset_path=None, img_transformer=None,
#                  bias_whole_image=None, args=None):
#         self.data_path = dataset_path
#
#         self.names = names
#         self.labels = labels
#         self.domain_labels = domain_labels
#
#         self.N = len(self.names)
#         self.grid_size = 3
#         self.bias_whole_image = bias_whole_image
#         if patches:
#             self.patch_size = 64
#         # self._image_transformer = img_transformer
#         self._image_transformer = img_transformer
#         self._image_transformer_aug = RandAugment(args)
#         # self._image_transformer_val = img_transformer_val
#
#     # def get_image(self, index):
#     #     framename = self.data_path + '/' + self.names[index]
#     #     img = Image.open(framename).convert('RGB')
#     #     return self._image_transformer(img), self._image_transformer_aug(img)
#
#     def __getitem__(self, index):
#         framename = self.data_path + '/' + self.names[index]
#         img = Image.open(framename).convert('RGB')
#         img_randaug, _ = self._image_transformer_aug(img)
#         # img_randaug = self._image_transformer_val(img_randaug)
#         return self._image_transformer(img), img_randaug, int(self.labels[index] - 1), int(self.domain_labels[index]-1)
#
#     def __len__(self):
#         return len(self.names)
#
# class JigsawTestDatasetRandAug(JigsawDatasetRandAug):
#     def __init__(self, *args, **xargs):
#         super().__init__(*args, **xargs)
#
#     def __getitem__(self, index):
#         framename = self.data_path + '/' + self.names[index]
#         img = Image.open(framename).convert('RGB')
#         return self._image_transformer(img), 0, int(self.labels[index] - 1), int(self.domain_labels[index] - 1)
#
#
from .FourierTransform import FourierTransform

class JigsawDatasetFourier(data.Dataset):
    def __init__(self, names, labels, domain_labels, dataset_path=None, img_transformer=None, args=None,
                 dataset_list=None):
        self.data_path = dataset_path

        self.names = names
        self.labels = labels
        self.domain_labels = domain_labels

        self._image_transformer = img_transformer
        self._image_transformer_aug = FourierTransform(args, dataset_list=dataset_list, base_dir=dataset_path)
        self.Fourier_swap = args.Fourier_swap
        # self._image_transformer_val = img_transformer_val

    def __getitem__(self, index):
        framename = self.data_path + '/' + self.names[index]
        img = Image.open(framename).convert('RGB')
        class_label = int(self.labels[index] - 1)
        domain_label = int(self.domain_labels[index]-1)
        img_randaug, domain_s, lam = self._image_transformer_aug(img, domain_label)
        # img_randaug = self._image_transformer_val(img_randaug)
        if self.Fourier_swap == 1:
            domain_label = [domain_s, domain_label]
        else:
            domain_label = [domain_label, domain_label]
        return self._image_transformer(img), img_randaug, class_label, domain_label

    def __len__(self):
        return len(self.names)

class JigsawTestDatasetFourier(JigsawDatasetFourier):
    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)

    def __getitem__(self, index):
        framename = self.data_path + '/' + self.names[index]
        img = Image.open(framename).convert('RGB')
        return self._image_transformer(img), 0, int(self.labels[index] - 1), int(self.domain_labels[index] - 1)

class JigsawTestDatasetFreqAnalyse(JigsawDatasetFourier):
    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)

    def __getitem__(self, index):
        framename = self.data_path + '/' + self.names[index]
        img = Image.open(framename).convert('RGB')
        class_label = int(self.labels[index] - 1)
        domain_label = int(self.domain_labels[index]-1)
        img_aug, domain_s, lam = self._image_transformer_aug(img, domain_label)
        return img_aug, class_label
#
#
# from .Tobias import Tobias
#
# class JigsawDatasetTobias(data.Dataset):
#     def __init__(self, names, labels, domain_labels, dataset_path=None, img_transformer=None, args=None,
#                  dataset_list=None):
#         self.data_path = dataset_path
#
#         self.names = names
#         self.labels = labels
#         self.domain_labels = domain_labels
#
#         self._image_transformer = img_transformer
#         self._image_transformer_aug = Tobias(args, dataset_list=dataset_list, base_dir=dataset_path)
#
#     def __getitem__(self, index):
#         framename = self.data_path + '/' + self.names[index]
#         img = Image.open(framename).convert('RGB')
#         class_label = int(self.labels[index] - 1)
#         domain_label = int(self.domain_labels[index]-1)
#         img_randaug, domain_s = self._image_transformer_aug(img=img, img_name=self.names[index],
#                                                             domain_label=domain_label)
#         return self._image_transformer(img), img_randaug, class_label, domain_label
#
#     def __len__(self):
#         return len(self.names)
#
#
# class JigsawTestDatasetTobias(JigsawDatasetTobias):
#     def __init__(self, *args, **xargs):
#         super().__init__(*args, **xargs)
#
#     def __getitem__(self, index):
#         framename = self.data_path + '/' + self.names[index]
#         img = Image.open(framename).convert('RGB')
#         return self._image_transformer(img), 0, int(self.labels[index] - 1), int(self.domain_labels[index] - 1)
