from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import torch
import h5py

import pandas as pd
from torchvision import transforms
from PIL import Image
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from torchvision.transforms.functional import to_pil_image

from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from dataset import ReadDataset
from ml_model.models import UNet

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

DEVICE = torch.device("cuda")

async def asyncobjectsegmentationtrain(dataset, training_record, model_metadata):
  # TODO: Fill with Object Segmentation code train code
  # dataset will be represent user's dataset (still on zip)

  model = UNet(in_channels=3, out_channels=1).to(DEVICE)

  loss_fn = nn.BCEWithLogitsLoss() # cross entropy
  learning_rate = 1e-4
  optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9) # sgd
  scaler = torch.cuda.amp.GradScaler()

  for epoch in range(20):
    epoch_dice_score = 0
    epoch_accuracy = 0
    epoch_iou = 0
    loop = tqdm(train_loader)
    total_batches = len(loop)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update metrics
        batch_dice_score = calculate_dice_score(predictions.detach(), targets)
        batch_accuracy = calculate_accuracy(predictions.detach(), targets)
        batch_iou = calculate_mean_iou(predictions.detach(), targets)
        epoch_dice_score += batch_dice_score.item()
        epoch_accuracy += batch_accuracy.item()
        epoch_iou += batch_iou.item()

    # Average metrics
    epoch_dice_score /= total_batches
    epoch_accuracy /= total_batches
    epoch_iou /= total_batches
    
    # Early stopping and checkpointing
    val_loss = epoch_dice_score  # Assuming we consider dice score as validation loss
    stop_training, best_score, no_improve_count = early_stopping(val_loss, best_score, no_improve_count)
    if stop_training:
        print(f"Stopping early at epoch {epoch+1}")
        break
    
    print(f"Epoch: {epoch+1}, Dice score: {epoch_dice_score:.4f}, Accuracy: {epoch_accuracy:.4f}, Mean IoU: {epoch_iou:.4f}")

# Save final model
  save_model_to_h5(model.state_dict(), 'final_model.h5')

def save_model_to_h5(model, filename):
    with h5py.File(filename, 'w') as f:
        for name, layer in model.named_parameters():
            f.create_dataset(name, data=layer.data.cpu().numpy())

# Define directories -> dataset user
img_dir = "data/train_val/train_val/JPEGImages"
mask_dir = "data/train_val/train_val/SegmentationClass"

# Initialize the dataset list
dataset = []

# Define supported file extensions
supported_image_extensions = (".jpg", ".jpeg")
supported_mask_extensions = (".png",)

# Helper function to get base filename without extension
def get_base_name(filename):
    return os.path.splitext(filename)[0]

# Iterate over image files and match with corresponding mask files
for img_filename in os.listdir(img_dir):
    if img_filename.endswith(supported_image_extensions):
        base_name = get_base_name(img_filename)
        img_path = os.path.join(img_dir, img_filename)
        mask_filename = f"{base_name}.png"  # Assuming mask filename is base_name.png
        mask_path = os.path.join(mask_dir, mask_filename)

        if os.path.exists(mask_path):  # Check if corresponding mask exists
            dataset.append({
                "image_path": img_path,
                "mask_path": mask_path
            })

# Convert to DataFrame
df = pd.DataFrame(dataset)

# Save to CSV
df.to_csv("pascal_dataset.csv", index=False)

class ReadDataset(Dataset):
    def __init__(self, dataframe, transform=None): 
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        img_path = self.dataframe.iloc[index]['image_path']
        mask_path = self.dataframe.iloc[index]['mask_path']

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        return self.transform(image), self.transform(mask)


df = pd.read_csv('pascal_dataset.csv')

tf_func = transforms.Compose([
    transforms.Resize((224, 224)),
#     transform.Normalize(mean=[0.0,0.0,0.0], std=[1.0,1.0,1.0])
    transforms.ToTensor()
])

td = ReadDataset(dataframe=df, transform=tf_func)
train_loader = DataLoader(
    td,
    batch_size=1,
    num_workers=0,
    pin_memory=True,
    shuffle=True,
)

# initialize early stopping parameters
best_score = None
no_improve_count = 0
patience = 5
min_delta = 0.001

# early stopping epoch
def early_stopping(current_loss, best_score, no_improve_count):
    stop_training = False
    if best_score is None:
        best_score = current_loss
    elif current_loss < best_score - min_delta:
        best_score = current_loss
        no_improve_count = 0
    else:
        no_improve_count += 1
        if no_improve_count >= patience:
            stop_training = True
    return stop_training, best_score, no_improve_count

# evaluation metrics
def calculate_dice_score(preds, y):
  smooth = 1e-8  # To avoid division by zero
  intersection = (preds * y).sum()
  union = (preds + y).sum() + smooth
  dice_score = (2 * intersection) / union
  return dice_score

def calculate_accuracy(preds, y):
  thresholded_preds = (preds > 0.5).float()  # Apply threshold
  correct_predictions = (thresholded_preds == y).sum()
  total_pixels = torch.numel(preds)
  accuracy = correct_predictions / total_pixels
  return accuracy

def calculate_mean_iou(preds, y, smooth=1e-8):
    # Convert probabilities to binary predictions
    preds = torch.sigmoid(preds)  # Assuming the output of the model is logits
    preds = (preds > 0.5).float()

    # Calculate intersection and union
    intersection = (preds * y).sum()
    total = (preds + y).sum()
    union = total - intersection

    # Compute IoU and handle division by zero
    iou = (intersection + smooth) / (union + smooth)
    return iou

def plot_images(data_loader, model, device):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        for images, masks in data_loader:  # Assuming the loader provides a batch of images and their corresponding masks
            images, masks = images.to(device), masks.to(device)
            preds = model(images)  # Get predictions from the model
            preds = torch.sigmoid(preds)  # Apply sigmoid to get probabilities
            preds = (preds > 0.5).float()  # Threshold the probabilities to get binary predictions

            images = images.cpu()
            masks = masks.cpu()
            preds = preds.cpu()

            figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
            for i in range(images.shape[0]):  # Display the first image in the batch
                ax[0].imshow(TF.to_pil_image(images[i]), cmap='gray')
                ax[0].set_title('Original Image')
                ax[0].axis('off')

                ax[1].imshow(TF.to_pil_image(masks[i]), cmap='gray')
                ax[1].set_title('Ground Truth Mask')
                ax[1].axis('off')

                ax[2].imshow(TF.to_pil_image(preds[i]), cmap='gray')
                ax[2].set_title('Predicted Mask')
                ax[2].axis('off')

                break  # Only display the first image and mask in the batch for clarity
            plt.show()