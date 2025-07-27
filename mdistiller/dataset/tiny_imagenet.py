import os
import numpy as np
import jittor as jt
#from jittor.dataset.folder import ImageFolder
import jittor.transform as transform


data_folder = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../data/tiny-imagenet-200"
)

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

class ImageFolderInstance(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target, index


class ImageFolderInstanceSample(ImageFolderInstance):
    """: Folder datasets which returns (img, label, index, contrast_index):"""

    def __init__(self, folder, transform=None, target_transform=None,
                 is_sample=False, k=4096):
        super().__init__(folder, transform=transform)
        self.k = k
        self.is_sample = is_sample
        if self.is_sample:
            num_classes = 200
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

        print('dataset initialized!')

    def __getitem__(self, index):
        img, target, index = super().__getitem__(index)

        if self.is_sample:
            pos_idx = index
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=True)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx
        else:
            return img, target, index


def get_tinyimagenet_dataloader(batch_size, val_batch_size, num_workers):
    train_transform = transform.Compose([
        transform.RandomRotation(20),
        transform.RandomHorizontalFlip(prob=0.5),
        transform.ToTensor(),
        transform.Normalize([0.4802, 0.4481, 0.3975],
                            [0.2302, 0.2265, 0.2262]),
    ])
    test_transform = transform.Compose([
        transform.ToTensor(),
        transform.Normalize([0.4802, 0.4481, 0.3975],
                            [0.2302, 0.2265, 0.2262]),
    ])
    train_folder = os.path.join(data_folder, "train")
    test_folder = os.path.join(data_folder, "val")
    train_set = ImageFolderInstance(train_folder, transform=train_transform)
    test_set = ImageFolder(test_folder, transform=test_transform)

    num_data = len(train_set)
    train_loader = train_set.set_attrs(batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = test_set.set_attrs(batch_size=val_batch_size, shuffle=False, num_workers=1)
    return train_loader, test_loader, num_data


def get_tinyimagenet_dataloader_sample(batch_size, val_batch_size, num_workers, k):
    train_transform = transform.Compose([
        transform.RandomRotation(20),
        transform.RandomHorizontalFlip(prob=0.5),
        transform.ToTensor(),
        transform.Normalize([0.4802, 0.4481, 0.3975],
                            [0.2302, 0.2265, 0.2262]),
    ])
    test_transform = transform.Compose([
        transform.ToTensor(),
        transform.Normalize([0.4802, 0.4481, 0.3975],
                            [0.2302, 0.2265, 0.2262]),
    ])
    train_folder = os.path.join(data_folder, "train")
    test_folder = os.path.join(data_folder, "val")
    train_set = ImageFolderInstanceSample(train_folder, transform=train_transform, is_sample=True, k=k)
    test_set = ImageFolder(test_folder, transform=test_transform)

    num_data = len(train_set)
    train_loader = train_set.set_attrs(batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = test_set.set_attrs(batch_size=val_batch_size, shuffle=False, num_workers=1)
    return train_loader, test_loader, num_data
