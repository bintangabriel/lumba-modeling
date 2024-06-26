from torchvision import transforms
from PIL import Image
import torch
import numpy as np
from ml_model.models.unet import UNet
import os
from io import BytesIO
from asynctrain.obseg_views import load_model_weights
import torch.nn as nn
from django.http import JsonResponse
import base64

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inference(model, image, device=DEVICE):
    # Load and preprocess the image
    image = Image.open(BytesIO(image)).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0], std=[1.0])
    ])
    
    image = preprocess(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        # Perform inference
        if isinstance(model, UNet):
            output = model(image)  # For UNet
        else:
            output = model(image)['out']  # For other models like Deeplab, FCN
        
        # Apply sigmoid activation and threshold to get binary mask
        output = torch.sigmoid(output)
        binary_mask = (output > 0.5).float()
    
    # Remove batch dimension and convert to PIL Image
    binary_mask = binary_mask.squeeze().cpu().numpy()
    binary_mask = (binary_mask * 255).astype(np.uint8)
    binary_mask = Image.fromarray(binary_mask)

    return binary_mask


def object_segmentation_inference(req):

  model_metadata = req.POST.dict()
  image = req.FILES.get('file')

  model_name = model_metadata['model_name']
  username = model_metadata['username']
  workspace = model_metadata['workspace']
  model_type = model_metadata['model_type']

  if model_type.startswith('50_deeplab'):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
    model = model.to(DEVICE)
  elif model_type.startswith('101_deeplab'):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
    model = model.to(DEVICE)
  elif model_type.startswith('50_fcn'):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)
    model.classifier[4] = nn.Conv2d(512, 1, kernel_size=1)  # Change the final layer to output 1 channel
    model = model.to(DEVICE)
  elif model_type.startswith('101_fcn'):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet101', pretrained=True)
    model.classifier[4] = nn.Conv2d(512, 1, kernel_size=1)  # Change the final layer to output 1 channel
    model = model.to(DEVICE)
    # model = load_model_weights(model=model, weights_path='')

  base_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  weights_file = os.path.join(base_directory, 'ml_model', 'models', 'weights', f'{model_name}_{username}_{workspace}.pth')

  model = load_model_weights(model=model, weights_path=weights_file)

  # Perform inference
  binary_mask = inference(model, image.read(), DEVICE)

  buffered = BytesIO()
  binary_mask.save(buffered, format="PNG")
  img_str = base64.b64encode(buffered.getvalue()).decode()

  # Return the base64 image string in the response
  return JsonResponse({"image": img_str})

  # Save or display the resulting mask