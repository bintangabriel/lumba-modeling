import os
import zipfile
from PIL import Image
import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from io import BytesIO

class ReadDataset(Dataset):
    def __init__(self, zip_file, image_transform=None, mask_transform=None):
      self.zip_file = zip_file
      self.image_transform = image_transform
      self.mask_transform = mask_transform
      self.images = []
      self.masks = []

      # Filter to include only images with corresponding masks
      with zipfile.ZipFile(zip_file, 'r') as z:
        for img_filename in z.namelist():
          if img_filename.endswith(('.jpg', '.jpeg', '.png')) and 'images/' in img_filename:
            mask_filename = img_filename.replace('images/', 'masks/').replace('.jpg', '.png').replace('.jpeg', '.png')
            if mask_filename in z.namelist():
              self.images.append(img_filename)
              self.masks.append(mask_filename)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
      with zipfile.ZipFile(self.zip_file, 'r') as z:
        img_filename = self.images[index]
        mask_filename = self.masks[index]
        
        img_data = z.read(img_filename)
        mask_data = z.read(mask_filename)
        
        image = Image.open(BytesIO(img_data)).convert("RGB")
        mask = Image.open(BytesIO(mask_data)).convert("L")
        
        if self.image_transform:
          image = self.image_transform(image)
        if self.mask_transform:
          mask = self.mask_transform(mask)
        
        mask = torch.where(mask > 0, 1, 0).float()
        
        return image, mask
