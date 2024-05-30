import pandas as pd
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
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
from modeling import settings
import requests

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
          self.images.append(img_filename)
        elif img_filename.endswith(('.jpg', '.jpeg', '.png')) and 'mask/' in img_filename:
          self.masks.append(img_filename)
      
    print(f"Loaded {len(self.images)} images and {len(self.masks)} masks.")
    # print(f'images: {self.images}')
  def __len__(self):
    return len(self.images)
  
  def __getitem__(self, index):
    with zipfile.ZipFile(self.zip_file, 'r') as z:
      img_filename = self.images[index]
      mask_filename = self.masks[index]
      print(f"Index: {index} - Image filename: {img_filename}")
      print(f"Index: {index} - Mask filename: {mask_filename}")
      
      img_data = z.read(img_filename)
      mask_data = z.read(mask_filename)
      
      
      try:
        image = Image.open(BytesIO(img_data)).convert("RGB")
      except UnidentifiedImageError as e:
        print(f"Error loading image {img_filename}: {e}")
        return None
      
      try:
        mask = Image.open(BytesIO(mask_data)).convert("L")
      except UnidentifiedImageError as e:
        print(f"Error loading mask {mask_filename}: {e}")
        return None
      
      if self.image_transform:
        image = self.image_transform(image)
      if self.mask_transform:
        mask = self.mask_transform(mask)
      
      mask = torch.where(mask > 0, 1, 0).float()
      
      return image, mask


async def asyncobjectsegmentationtrain(dataset, model_metadata):
  
  # Get model that will be used
  model_type = model_metadata['model_type']
  epoch = model_metadata['epoch']
  learning_rate = model_metadata['learning_rate']
  print(f'model name: {model_type}')
  model = None

  # TODO: Use weights
  if model_type.startswith('unet'):
    unet = UNet(in_channels=3, out_channels=1).to(DEVICE)
    model = unet   
    base_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_file = os.path.join(base_directory, 'ml_model', 'models', 'weights', 'unet_best_weights.pth')

    print(f'path to unet weight: {weights_file}')
    model = load_model_weights(model=model, weights_path=weights_file)
  elif model_type.startswith('deeplab'):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
    model = model.to(DEVICE)
    # model = load_model_weights(model=model, weights_path='')
  elif model_type.startswith('fcn'):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)
    model.classifier[4] = nn.Conv2d(512, 1, kernel_size=1)  # Change the final layer to output 1 channel
    model = model.to(DEVICE)
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
  print(f'total td: {total_size}')
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

  if model_type.endswith('0'):
    bce_weighted = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor).to(DEVICE)
  else:
    bce_weighted = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor).to(DEVICE)

  loss_fn = bce_weighted
  learning_rate = float(learning_rate)
  optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

  train_model(
     model=model,
     train_loader=train_loader,
     val_loader=val_loader,
     optimizer=optimizer,
     loss_fn=loss_fn,
     epochs=int(epoch),
     device=DEVICE,
     metadata=model_metadata
  )
  url = f'http://{settings.BACKEND_SERVICE_INTERNAL_IP}:{settings.BACKEND_SERVICE_RUNNING_PORT}/modeling/save/'
  requests.post(url, data=model_metadata)


def set_seed(seed=42):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

def load_model_weights(model, weights_path):
  state_dict = torch.load(weights_path)
  model_state_dict = model.state_dict()
  filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}

  # Update the model’s state dictionary
  model_state_dict.update(filtered_state_dict)
  model.load_state_dict(model_state_dict)
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
      else: 
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

def train_model(model, train_loader, val_loader, optimizer, loss_fn, epochs, device, metadata):

  model_name = metadata['model_name']
  username = metadata['username']
  workspace = metadata['workspace']
  filename = metadata['filename']
  id = metadata['id']

  best_val_iou = 0.0  # Initialize the best validation IoU score
  scaler = torch.cuda.amp.GradScaler()
  start_time = datetime.now()
  url = f'http://{settings.BACKEND_SERVICE_INTERNAL_IP}:{settings.BACKEND_SERVICE_RUNNING_PORT}/modeling/updaterecord/'
  json = {'id': id, 'status':'in progress'}
  record = requests.post(url, json=json)
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
            else:
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

  base_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  weights_file = os.path.join(base_directory, 'ml_model', 'models', 'weights', f'{model_name}_{username}_{workspace}.pth')
  torch.save(model.state_dict(), weights_file)
  
  end_time = datetime.now()
  print(f'Ending training at {end_time}')
  time_taken = end_time - start_time
  print(f'Time taken: {time_taken}')

  url = f'http://{settings.BACKEND_SERVICE_INTERNAL_IP}:{settings.BACKEND_SERVICE_RUNNING_PORT}/modeling/updaterecord/'
  json = {'id': id, 'status':'completed'}
  record = requests.post(url, json=json)