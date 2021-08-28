import glob
import os

from core import field
from skimage.io import imread
from torch.utils.data import Dataset


class PreCachedXview2Building(Dataset):
    def __init__(self,
                 image_dir,
                 targets_dir,
                 transforms=None
                 ):
        self.target_fps = sorted([fp for fp in glob.glob(os.path.join(targets_dir, '*.png')) if 'pre' in fp])

        self.image_fps = [os.path.join(image_dir, os.path.basename(fp.replace('_target.png', '.png'))) for fp in
                          self.target_fps]

        self.transforms = transforms

    def __getitem__(self, idx):
        img = imread(self.image_fps[idx])
        mask = imread(self.target_fps[idx])

        if self.transforms:
            blob = self.transforms(**dict(image=img, mask=mask))
            img = blob['image']
            mask = blob['mask']

        y = dict()
        y[field.MASK1] = mask
        y['image_filename'] = os.path.basename(self.image_fps[idx])
        return img, y

    def __len__(self):
        return len(self.image_fps)
