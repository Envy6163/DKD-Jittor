import os
from PIL import Image
import jittor as jt
from jittor.dataset.dataset import Dataset
from glob import glob

class SimpleImageFolder(Dataset):
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
