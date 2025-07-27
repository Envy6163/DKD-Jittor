import os
import numpy as np
import jittor as jt
#from jittor.dataset.folder import ImageFolder
import jittor.transform as transform
#from simple_image_folder import SimpleImageFolder as ImageFolder

data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/imagenet')

from PIL import Image
from jittor.dataset.dataset import Dataset
from glob import glob

# 自建imagefolder
class ImageFolder(Dataset):
    def __init__(self, root, transform=None):
        super().__init__()
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}

        class_dirs = sorted(os.listdir(root))
        for idx, cls in enumerate(class_dirs):
            self.class_to_idx[cls] = idx
            cls_folder = os.path.join(root, cls)
            for img_path in glob(os.path.join(cls_folder, "*")):
                self.samples.append((img_path, idx))

        self.set_attrs(total_len=len(self.samples))

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class ImageNet(ImageFolder):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index


class ImageNetInstanceSample(ImageNet):
    """
    ImageNet dataset with optional sampling for contrastive learning.
    Returns (img, label, index, contrast_index) when is_sample=True.
    """
    def __init__(self, folder, transform=None, target_transform=None,
                 is_sample=False, k=4096):
        super().__init__(folder, transform=transform)
        self.k = k
        self.is_sample = is_sample

        if self.is_sample:
            print('preparing contrastive data...')
            num_classes = 1000
            num_samples = len(self.samples)
            label = np.zeros(num_samples, dtype=np.int32)
            for i in range(num_samples):
                _, target = self.samples[i]
                label[i] = target

            self.cls_positive = [[] for _ in range(num_classes)]
            for i in range(num_samples):
                self.cls_positive[label[i]].append(i)

            self.cls_negative = [[] for _ in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [np.asarray(p, dtype=np.int32) for p in self.cls_positive]
            self.cls_negative = [np.asarray(n, dtype=np.int32) for n in self.cls_negative]
            print('done.')

    def __getitem__(self, index):
        img, target, index = super().__getitem__(index)

        if self.is_sample:
            pos_idx = index
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=True)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx
        else:
            return img, target, index


def get_imagenet_train_transform(mean, std):
    normalize = transform.Normalize(mean=mean, std=std)
    train_transform = transform.Compose([
        transform.RandomResizedCrop(224),
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        normalize,
    ])
    return train_transform


def get_imagenet_test_transform(mean, std):
    normalize = transform.Normalize(mean=mean, std=std)
    test_transform = transform.Compose([
        transform.Resize(256),
        transform.CenterCrop(224),
        transform.ToTensor(),
        normalize,
    ])
    return test_transform


def get_imagenet_dataloaders(batch_size, val_batch_size, num_workers,
                             mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]):
    train_transform = get_imagenet_train_transform(mean, std)
    train_folder = os.path.join(data_folder, 'train')
    train_set = ImageNet(train_folder, transform=train_transform)
    num_data = len(train_set)
    train_loader = train_set.set_attrs(batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = get_imagenet_val_loader(val_batch_size, mean, std)
    return train_loader, test_loader, num_data


def get_imagenet_dataloaders_sample(batch_size, val_batch_size, num_workers, k=4096,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]):
    train_transform = get_imagenet_train_transform(mean, std)
    train_folder = os.path.join(data_folder, 'train')
    train_set = ImageNetInstanceSample(train_folder, transform=train_transform, is_sample=True, k=k)
    num_data = len(train_set)
    train_loader = train_set.set_attrs(batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = get_imagenet_val_loader(val_batch_size, mean, std)
    return train_loader, test_loader, num_data


def get_imagenet_val_loader(val_batch_size, mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]):
    test_transform = get_imagenet_test_transform(mean, std)
    test_folder = os.path.join(data_folder, 'val')
    test_set = ImageFolder(test_folder, transform=test_transform)
    test_loader = test_set.set_attrs(batch_size=val_batch_size, shuffle=False, num_workers=4)
    return test_loader
