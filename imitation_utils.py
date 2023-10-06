"""
Handles dataset organization for dataloader.
Additionally used as a utility to check model accuracy.
"""

import csv
import os
import torch

from PIL import Image, ImageOps

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

class CarlaDataset(Dataset):
    """
    Classification data from CarlaDataset.
    Represented as tuples of 3 x 150 x 200 images and their vectors of data/labels
    """
    def __init__(self, dataset_path):

        self.csv_tuples = []
        self.dataset_path = dataset_path
        self.transform = transforms.Compose([
            transforms.Resize(96),
            # transforms.CenterCrop(224), 
            transforms.ToTensor()])

        # Extract data from csv.
        labels_path = os.path.join(dataset_path, "labels.csv")
        with open(labels_path, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            for row in reader:
                self.csv_tuples.append((row[0], row[1], row[2], row[3], row[4])) 
                # 0 is rgb fname, 1 is sem name, 2 is data/labels vector

        # Cut out the csv headers from extracted data.
        self.csv_tuples = self.csv_tuples[1:]


    def __len__(self):
        """
        Your code here
        returns length of dataset.
        """
        return len(self.csv_tuples)

    @staticmethod
    def rgb2gray(rgb, norm=True):
        # rgb image -> gray [0, 1]
        np_rgb = np.array(rgb)
        gray = np.dot(np_rgb[..., :], [0.299, 0.587, 0.114])
        if norm:
            # normalize
            gray = gray / 128. - 1.
        return gray

    def __getitem__(self, idx):
        """
        Your code here
        return a tuple: img, label
        """

        # All pairs of image and label are added to csv_tuples string list.
        # Grab the data vector from 2nd index, get only labels and change to floats
        data = [float(self.csv_tuples[idx][2]),float(self.csv_tuples[idx][3]), float(self.csv_tuples[idx][4])]

        # border = (0, 150, 0, 0) # cut 0 from left, 30 from top, right, bottom
        image_path = os.path.join(self.dataset_path, self.csv_tuples[idx][0])
        # Rgb image as a tensor
        gray_image = Image.open(image_path).convert('L')
        # gray_image = self.rgb2gray(rgb_image)
        # # # rgb_image = ImageOps.crop(rgb_image, border)
        # # rgb_tensor = self.transform(gray_image)
        # gray_tensor = torch.from_numpy(gray_image).double()
        gray_tensor = self.transform(gray_image)

        # Sem image as an input tensor
        sem_image = Image.open(os.path.join(self.dataset_path, self.csv_tuples[idx][1]))
        # sem_image = ImageOps.crop(sem_image, border)
        sem_tensor = self.transform(sem_image)

        # return rgb_tensor, sem_tensor, torch.tensor(data)
        return gray_tensor, sem_tensor, torch.tensor(data), image_path


def load_data(dataset_path, num_workers=0, batch_size=32):
    """
    Driver function to create dataset and return constructed dataloader.
    """
    dataset = CarlaDataset(dataset_path)
    # return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False, drop_last=False)


def accuracy(yhat, y): # yhat is not labels it is prediction
    """
    Returns accuracy between true labels and predictions.
    """
    def rmse(y_tensor, yhat_tensor):
        return torch.sqrt(torch.mean(torch.pow((y_tensor - yhat_tensor), 2)))

    steer_rmse = throttle_rmse = brake_rmse = 0
    for j in range(len(y)):
        steer_rmse += rmse(y[j, 0], yhat[j, 0])
        throttle_rmse += rmse(y[j, 1], yhat[j, 1])
        brake_rmse += rmse(y[j, 2], yhat[j, 2])

    steer_rmse /= len(y)
    throttle_rmse /= len(y)
    brake_rmse /= len(y)

    # accuracy_steer = round(1 - steer_rmse.item(), 3)
    # accuracy_throttle = round(1 - throttle_rmse.item(), 3)
    # accuracy_brake = round(1 - brake_rmse.item(), 3)
    accuracy_steer = round(steer_rmse.item(), 3)
    accuracy_throttle = round(throttle_rmse.item(), 3)
    accuracy_brake = round(brake_rmse.item(), 3)
    accuracy_avg = round((accuracy_steer + accuracy_throttle + accuracy_brake) / 3, 3)
    return accuracy_steer, accuracy_throttle, accuracy_brake, accuracy_avg
