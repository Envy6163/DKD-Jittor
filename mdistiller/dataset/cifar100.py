import os
import numpy as np
from PIL import Image
import jittor as jt
from jittor import transform
from jittor.dataset.cifar import CIFAR100
from jittor.dataset import Dataset

def get_data_folder(): # 获取/创建数据存储目录
    data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data")
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)
    return data_folder

class CIFAR100Instance(CIFAR100):
    """CIFAR100Instance Dataset using Jittor backend."""

    def __init__(self, **kwargs):
        super().__init__(download=True, **kwargs)
        self.labels = self.targets
        
    def __getitem__(self, index):
        img, target = self.data[index], int(self.labels[index])
        img = Image.fromarray(img)  # 保持与原始 torchvision 一致
        if self.transform:
            img = self.transform(img)
        return img, target, index  # 保持输出结构一致


# CIFAR-100 for CRD
class CIFAR100InstanceSample(CIFAR100):
    """
    CIFAR100Instance+Sample Dataset using Jittor backend.
    """
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=True,
        k=4096,
        mode="exact",
        is_sample=True,
        percent=1.0,
    ):
        super().__init__(root=root, train=train, transform=transform, download=download)
        self.k = k
        self.mode = mode
        self.is_sample = is_sample
        self.labels = self.targets
        
        num_classes = 100
        num_samples = len(self.data)
        label = self.labels

        self.cls_positive = [[] for _ in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for _ in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j != i:
                    self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(p) for p in self.cls_positive]
        self.cls_negative = [np.asarray(n) for n in self.cls_negative]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [
                np.random.permutation(self.cls_negative[i])[:n]
                for i in range(num_classes)
            ]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.labels[index])
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)

        if not self.is_sample:
            return img, target, index

        # 启用采样逻辑
        if self.mode == "exact":
            pos_idx = index
        elif self.mode == "relax":
            pos_idx = np.random.choice(self.cls_positive[target], 1)[0]
        else:
            raise NotImplementedError(self.mode)

        replace = self.k > len(self.cls_negative[target])
        neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
        sample_idx = np.hstack(([pos_idx], neg_idx))
        return img, target, index, sample_idx
        
# # 自定义pad
from PIL import ImageOps
class MyPad:
    def __init__(self, padding, fill=0):
        self.padding = padding
        self.fill = fill

    def __call__(self, img):
        # 确保输入是 PIL 图像
        if not isinstance(img, Image.Image):
            raise TypeError("Expected PIL.Image, got {}".format(type(img)))
        return ImageOps.expand(img, border=self.padding, fill=self.fill)
        
def get_cifar100_train_transform():
    train_transform = transform.Compose([
        MyPad(padding=4), # 增加pading后训练output_1
        transform.RandomCrop(32), # jittor不支持pad，自定义实现
        
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.ImageNormalize(mean=[0.5071, 0.4867, 0.4408],
                            std=[0.2675, 0.2565, 0.2761]),
    ])
    return train_transform

def get_cifar100_test_transform():
    test_transform = transform.Compose([
        transform.ToTensor(),
        transform.ImageNormalize(mean=[0.5071, 0.4867, 0.4408],
                            std=[0.2675, 0.2565, 0.2761]),
    ])
    return test_transform

def get_cifar100_dataloaders(batch_size, val_batch_size, num_workers):
    data_folder = get_data_folder()
    train_transform = get_cifar100_train_transform()
    test_transform = get_cifar100_test_transform()

    train_set = CIFAR100Instance(train=True, transform=train_transform, root=data_folder)
    test_set = CIFAR100(train=False, transform=test_transform, root=data_folder, download=True)

    num_data = len(train_set)

    train_loader = train_set.set_attrs(batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = test_set.set_attrs(batch_size=val_batch_size, shuffle=False, num_workers=1)

    return train_loader, test_loader, num_data

# CIFAR-100 for CRD
def get_cifar100_dataloaders_sample(batch_size, val_batch_size, num_workers, k, mode="exact"):
    data_folder = get_data_folder()
    train_transform = get_cifar100_train_transform()
    test_transform = get_cifar100_test_transform()

    train_set = CIFAR100InstanceSample(
        root=data_folder,
        train=True,
        transform=train_transform,
        k=k,
        mode=mode,
        is_sample=True,
        percent=1.0,
    )

    test_set = CIFAR100(train=False, transform=test_transform, root=data_folder, download=True)

    num_data = len(train_set)

    train_loader = train_set.set_attrs(batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = test_set.set_attrs(batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, num_data

