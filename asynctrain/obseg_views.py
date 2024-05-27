import pandas as pd
from torchvision import transforms
from PIL import Image
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
from datetime import datetime
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchmetrics as TM
import random
from ml_model.models.unet import UNet
import os
import zipfile
from io import BytesIO


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

DEVICE = torch.device("cuda")
iou = TM.classification.BinaryJaccardIndex().to(DEVICE)
pixel_metric = TM.classification.BinaryAccuracy().to(DEVICE)
precision = TM.classification.BinaryPrecision().to(DEVICE)
recall = TM.classification.BinaryRecall().to(DEVICE)
dice_metric = TM.classification.BinaryF1Score().to(DEVICE)

class ReadDataset():
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
        
      print(f"Loaded {len(self.images)} images and {len(self.masks)} masks.")
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


async def asyncobjectsegmentationtrain(dataset, training_record, model_metadata):
  
  # Get model that will be used
  model_name = model_metadata['model_name']
  model = ''

  # Todo: Use weights
  if model_name == 'unet':
    unet = UNet(in_channels=3, out_channels=1).to(DEVICE)
    model = unet
    # model = load_model_weights(model=model, weights_path='')
  elif model_name == 'deeplab':
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    model.classifier[4] = nn.Conv2d(256, 3, kernel_size=1)
    # model = load_model_weights(model=model, weights_path='')
  elif model_name == 'fcn':
    model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)
    model.classifier[4] = nn.Conv2d(512, 1, kernel_size=1)  # Change the final layer to output 1 channel
    # model = load_model_weights(model=model, weights_path='')

  # Prepare the dataset
  tf_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0], std=[1.0])
  ])

  tf_val_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0], std=[1.0])
  ])

  td = ReadDataset(zip_file=dataset, image_transform=tf_train, mask_transform=tf_train)
  total_size = len(td)
  train_size = int(0.8 * total_size)
  val_size = total_size - train_size

  train_dataset, val_dataset = random_split(td, [train_size, val_size])

  train_dataset = ReadDataset(zip_file=dataset, image_transform=tf_train, mask_transform=tf_train)
  val_dataset = ReadDataset(zip_file=dataset, image_transform=tf_val_test, mask_transform=tf_val_test)

  train_loader = DataLoader(
    train_dataset,
    batch_size=20,
    num_workers=2,
    pin_memory=True,
    shuffle=True,
  )

  val_loader = DataLoader(
    val_dataset,
    batch_size=20,
    num_workers=2,
    pin_memory=True,
    shuffle=False,
  )
  
  num_pos, num_neg = count_positive_negative_samples(train_loader)

  set_seed(42)

  pos_weight = num_neg / num_pos
  pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32)

  bce_weighted = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor).to(DEVICE)

  loss_fn = bce_weighted
  learning_rate = 1e-4
  optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

  train_model(
     model=model,
     train_loader=train_loader,
     val_loader=val_loader,
     optimizer=optimizer,
     loss_fn=loss_fn,
     epochs=20,
     device=DEVICE
  )


def set_seed(seed=42):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

def load_model_weights(model, weights_path):
  model.load_state_dict(torch.load(weights_path))
  return model

def count_positive_negative_samples(data_loader):
  num_pos = 0
  num_neg = 0

  for _, masks in data_loader:
      num_pos += (masks == 1).sum().item()
      num_neg += (masks == 0).sum().item()

  return num_pos, num_neg

def validate_model(model, val_loader, DEVICE, loss_fn):
  model.eval()
  val_loss = 0
  val_iou = 0
  val_accuracy = 0
  val_precision = 0
  val_recall = 0
  val_dice = 0
  total_batches = len(val_loader)
  
  with torch.no_grad():
    for data, targets in val_loader:
      data, targets = data.to(DEVICE), targets.to(DEVICE)

      # need if-else condition
      if (isinstance(model, UNet)):
        predictions = model(data) #unet only
      predictions = model(data)['out'] 
      loss = loss_fn(predictions, targets)
      val_loss += loss.item()

      batch_iou = iou(predictions, targets)
      batch_accuracy = pixel_metric(predictions, targets)
      batch_precision = precision(predictions, targets)
      batch_recall = recall(predictions, targets)
      batch_dice = dice_metric(predictions, targets)
      val_iou += batch_iou.item()
      val_accuracy += batch_accuracy.item()
      val_precision += batch_precision.item()
      val_recall += batch_recall.item()
      val_dice += batch_dice.item()

  val_loss /= total_batches
  val_accuracy /= total_batches
  val_iou /= total_batches
  val_precision /= total_batches
  val_recall /= total_batches
  val_dice /= total_batches
  
  return val_loss, val_accuracy, val_iou, val_precision, val_recall, val_dice

def train_model(model, train_loader, val_loader, optimizer, loss_fn, epochs, device):
  best_val_iou = 0.0  # Initialize the best validation IoU score
  scaler = torch.cuda.amp.GradScaler()
  start_time = datetime.now()
  print(f'Starting training at {start_time}')

  for epoch in range(epochs):
      epoch_iou = 0
      epoch_accuracy = 0
      epoch_precision = 0
      epoch_recall = 0
      epoch_dice = 0

      loop = tqdm(train_loader)
      total_batches = len(loop)
      model.train()

      for batch_idx, (data, targets) in enumerate(loop):
          data, targets = data.to(device), targets.to(device)
          with torch.cuda.amp.autocast():

              # need if-else condition
            if (isinstance(model, UNet)):
              predictions = model(data) # unet only
            predictions = model(data)['out'] 

            loss = loss_fn(predictions, targets)

          optimizer.zero_grad()
          scaler.scale(loss).backward()
          scaler.step(optimizer)
          scaler.update()

          # Update metrics
          epoch_iou += iou(predictions, targets).item()
          epoch_accuracy += pixel_metric(predictions, targets).item()
          epoch_precision += precision(predictions, targets).item()
          epoch_recall += recall(predictions, targets).item()
          epoch_dice += dice_metric(predictions, targets).item()
          
      # Average metrics
      epoch_iou /= total_batches
      epoch_accuracy /= total_batches
      epoch_precision /= total_batches
      epoch_recall /= total_batches
      epoch_dice /= total_batches
      

      # Validate on the validation set
      val_loss, val_accuracy, val_iou, val_precision, val_recall, val_dice = validate_model(model, val_loader, device, loss_fn)
      

      print(f"Epoch: {epoch+1}, Train IoU: {epoch_iou:.4f}, Train Accuracy: {epoch_accuracy:.4f}, Train Precision: {epoch_precision:.4f}, Train Recall: {epoch_recall:.4f}, Train Dice: {epoch_dice:.4f}")
      print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation IoU: {val_iou:.4f}, Validation Precision: {val_precision:.4f}, Validation Recall: {val_recall:.4f}, Validation Dice: {val_dice:.4f}")

  torch.save(model.state_dict(), f'model/deeplabv3_model.pth')
  
  end_time = datetime.now()
  print(f'Ending training at {end_time}')
  time_taken = end_time - start_time
  print(f'Time taken: {time_taken}')
