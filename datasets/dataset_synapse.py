import os
import random
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import cv2
from os.path import splitext, isfile, join
from os import listdir
from PIL import Image, ImageCms
from pathlib import Path

def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)
    

# def random_rot_flip(image, label):
#     k = np.random.randint(0, 4)
#     image = np.rot90(image, k)
#     label = np.rot90(label, k)
#     axis = np.random.randint(0, 2)
#     image = np.flip(image, axis=axis).copy()
#     label = np.flip(label, axis=axis).copy()
#     return image, label


# def random_rotate(image, label):
#     angle = np.random.randint(-20, 20)
#     image = ndimage.rotate(image, angle, order=0, reshape=False)
#     label = ndimage.rotate(label, angle, order=0, reshape=False)
#     return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, name = sample['image'], sample['image_name']
        image = np.array(image)
        image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)   # 224,224,3
        # image = np.repeat(image[:,:,np.newaxis],3,axis=2)
        image = np.transpose(image, (2, 0, 1))                                       # 3,224,224
        image = torch.from_numpy(image.astype(np.float32))      # torch.Size([1, 224, 224, 3])->(224,224,3)
        sample = {'image': image, 'image_name': name}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.data_dir = Path(base_dir)
        self.ids = [splitext(file)[0] for file in listdir(base_dir) if isfile(join(base_dir, file)) and not file.startswith('.')]

    def __len__(self):
        return len(self.ids)


    def __getitem__(self, idx):

        if self.split == "label_data":
            name = self.ids[idx]
            img_file = list(self.data_dir.glob(name + '.*'))
            assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
            image = load_image(img_file[0])    # （PIL：mode"RGB" size(300,170)）

        sample = {'image': image, 'image_name': name}
        if self.transform:
            sample = self.transform(sample)
        return sample




