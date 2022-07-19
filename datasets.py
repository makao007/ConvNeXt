# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import io
import csv
import zipfile
import os.path as osp
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform


def parse_text_file (text_file, folder='', skips=1, drop_last=True, image_index=0, label_index=1, split_tag=","):
    images = []
    labels = []
    if text_file.lower().endswith('.csv'):
        with open(text_file, mode='r', encoding="utf-8-sig") as r:
            reader = csv.reader(r, delimiter=split_tag)
            for index,row in enumerate(reader):
                if index < skips: continue
                images.append(osp.join(folder, row[image_index]))
                labels.append(int(row[label_index].strip()))
    else:
        with open(text_file, 'r', encoding='utf-8') as r:
            for index,i in enumerate(r):
                if (index >= skips) and split_tag in i and i.strip():
                    row = i.split(split_tag)
                    images.append(osp.join(folder, row[image_index]))
                    labels.append(int(row[label_index].strip()))
    if drop_last:
        images.pop()
        labels.pop()
    return images, labels

class ImageLabelDataset(Dataset):
    def __init__(self, folder:str, text_file:str, img_folder:str, transform=None, target_transform=None, **args):
        to_rgb      = args.get('to_rgb', True)
        skips       = int(args.get('skips',1))
        drop_last   = args.get('drop_last',True)
        image_index = int(args.get('image_index', 0))
        label_index = int(args.get('label_index', 1))
        split_tag   = args.get('split_tag', ',')
        text_file   = os.path.join(folder, text_file)
        
        self.files, self.labels = parse_text_file(text_file, img_folder, skips, drop_last, image_index, label_index, split_tag)
        self.folder = folder
        self.transform = transform
        self.target_transform = target_transform
        self.to_rgb = to_rgb
        self.lens   = len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        img = Image.open(os.path.join(self.folder, self.files[index]))
        if self.to_rgb:
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label

    def __len__(self):
        return self.lens


class ZipFileDataset(Dataset):
    def __init__(self, folder:str, transform=None, target_transform=None, **args):
        self.folder = folder
        to_rgb      = args.get('to_rgb', True)
        skips       = int(args.get('skips',1))
        drop_last   = args.get('drop_last',True)
        image_index = int(args.get('image_index', 0))
        label_index = int(args.get('label_index', 1))
        split_tag   = args.get('split_tag', '\t')
        text_file   = os.path.join(folder, args['text_file'])
        img_folder  = args.get('image_folder', '')

        self.zipfilepath = os.path.join(folder, args['zip_file'])
        self.files, self.labels = parse_text_file(text_file, img_folder, skips, drop_last, image_index, label_index, split_tag)
        self.transform = transform
        self.target_transform = target_transform
        self.to_rgb = to_rgb
        self.lens   = len(self.labels)
        self.zip_reader = None
        self.zip_namelist = []

    def __getitem__(self, index):
        img = self.get_zip_image (self.zipfilepath, self.files[index])
        label = self.labels[index]
        if self.to_rgb:
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label

    def __len__(self):
        return self.lens

    def get_zip_image (self, zipfilepath, name):
        if self.zip_reader is None:
            self.zip_reader = zipfile.ZipFile(zipfilepath, 'r')
            self.zip_namelist = self.zip_reader.namelist()
        if name in self.zip_namelist:
            data = self.zip_reader.read(name)
            try:
                img = Image.open(io.BytesIO(data))
            except Exception as e:
                print("Error, Read Image inside Zip File Fail: ", zipfilepath, name)
                print(e)
                random_img = np.random.rand(224, 224, 3) * 255
                img = Image.fromarray(np.uint8(random_img))
            return img

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        print("reading from datapath", args.data_path)
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    elif args.data_set == 'image_label':
        root = args.data_path if is_train else args.eval_data_path
        text = args.train_text if is_train else args.eval_text
        folder = args.train_file if is_train else args.eval_file
        dataset = ImageLabelDataset (root, text, folder, transform, ** vars(args))
        nb_classes = int(args.nb_classes)
    elif args.data_set == 'zip_label':
        root = args.data_path if is_train else args.eval_data_path
        dataset = ZipFileDataset (root, transform, **vars(args))
        nb_classes = int(args.nb_classes)
    else:
        raise NotImplementedError()
    print("Number of the class = %d" % nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if args.input_size >= 384:  
            t.append(
            transforms.Resize((args.input_size, args.input_size), 
                            interpolation=transforms.InterpolationMode.BICUBIC), 
        )
            print(f"Warping {args.input_size} size input images...")
        else:
            if args.crop_pct is None:
                args.crop_pct = 224 / 256
            size = int(args.input_size / args.crop_pct)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),  
            )
            t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
