import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): Path to 'tiny-imagenet-200' directory
            split (str): 'train' or 'val'
            transform (callable): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.data = []  # List of (img_path, label_idx)
        self.class_to_idx = {}  # e.g., {'n01443537': 0, ...}
        self.idx_to_class = []  # Inverse mapping

        self._prepare_dataset()

    def _prepare_dataset(self):
        wnids_path = os.path.join(self.root_dir, 'wnids.txt')
        with open(wnids_path, 'r') as f:
            wnids = [line.strip() for line in f.readlines()]
        self.class_to_idx = {wnid: i for i, wnid in enumerate(wnids)}
        self.idx_to_class = wnids

        if self.split == 'train':
            train_dir = os.path.join(self.root_dir, 'train')
            for wnid in wnids:
                class_dir = os.path.join(train_dir, wnid, 'images')
                for fname in os.listdir(class_dir):
                    if fname.endswith('.JPEG'):
                        img_path = os.path.join(class_dir, fname)
                        self.data.append((img_path, self.class_to_idx[wnid]))

        elif self.split == 'val':
            val_dir = os.path.join(self.root_dir, 'val')
            ann_path = os.path.join(val_dir, 'val_annotations.txt')
            ann_dict = {}
            with open(ann_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split('\t')
                    filename, wnid = parts[0], parts[1]
                    ann_dict[filename] = self.class_to_idx[wnid]
            images_dir = os.path.join(val_dir, 'images')
            for fname in os.listdir(images_dir):
                if fname.endswith('.JPEG') and fname in ann_dict:
                    img_path = os.path.join(images_dir, fname)
                    label = ann_dict[fname]
                    self.data.append((img_path, label))
        else:
            raise ValueError("split must be 'train' or 'val'")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
