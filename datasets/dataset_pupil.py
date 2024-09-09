import os
import random
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import cv2
from pathlib import Path
from os.path import splitext, isfile, join
from os import listdir
from PIL import Image, ImageCms

def random_rot_flip(image, label):

    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)

    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()


    return image, label


def random_rotate(image, label):

    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)

   
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label= sample['image'], sample['label']
        if random.random() > 0.5:
            image, label= random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y, _ = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = cv2.resize(image, dsize=(self.output_size[1], self.output_size[0]), interpolation=cv2.INTER_CUBIC)
            label = cv2.resize(label, dsize=(self.output_size[1], self.output_size[0]), interpolation=cv2.INTER_NEAREST)

        image = np.transpose(image, (2, 0, 1))
        # label = np.transpose(label, (2, 0, 1))
        image = torch.from_numpy(image.astype(np.float32))      # torch.Size([1, 224, 224, 3])->(224,224,3)
        label = torch.from_numpy(label.astype(np.float32))                   # torch.Size([224, 224])

        sample = {'image': image, 'label': label.long()}
        return sample

class RandomGenerator_test(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        x, y, _ = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = cv2.resize(image, dsize=(self.output_size[1], self.output_size[0]), interpolation=cv2.INTER_CUBIC)
            label = cv2.resize(label, dsize=(self.output_size[1], self.output_size[0]), interpolation=cv2.INTER_NEAREST)
        image = np.transpose(image, (2, 0, 1))
        # label = np.transpose(label, (2, 0, 1))
        image = torch.from_numpy(image.astype(np.float32))      # torch.Size([1, 224, 224, 3])->(224,224,3)
        label = torch.from_numpy(label.astype(np.float32))                   # torch.Size([224, 224])
        sample = {'image': image, 'label': label.long()}
        return sample

class Synapse_dataset2(Dataset):
    def __init__(self, base_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.ids = [splitext(file)[0] for file in listdir(base_dir) if isfile(join(base_dir, file)) and not file.startswith('.')]
        self.data_dir = Path(base_dir)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        slice_name = self.ids[idx].strip('\n')
        train_label = str(self.data_dir).replace('image' , 'iris')
        # train_fill = str(self.data_dir).replace('image' , 'fill')
        image_path = os.path.join(self.data_dir, slice_name + '.jpg')
        label_path = os.path.join(train_label, slice_name + '.png')
        # fill_path = os.path.join(train_fill, slice_name + '.png')
        
        image = Image.open(image_path)
        image2 = np.array(image)

        label = np.array(Image.open(label_path).convert('L'))
        # fill = np.array(Image.open(fill_path).convert('L'))
        label = label / 255

        sample = {'image': image2 , 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = slice_name
        return sample
