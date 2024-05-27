import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class FCNBackbone(nn.Module):
    def __init__(self, backbone_model, n_class):
        super(FCNBackbone, self).__init__()
        # Load the pre-trained backbone
        self.backbone = timm.create_model(backbone_model, pretrained=True, features_only=True)
        # Define the number of output classes
        self.n_class = n_class
        # Define the upsampling layers
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        # Define batch normalization layers
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(32)
        # Define 1x1 convolution to match dimensions
        self.conv1x1_x4 = nn.Conv2d(512, 512, kernel_size=1)
        self.conv1x1_x3 = nn.Conv2d(512, 256, kernel_size=1)
        # Define the classifier layer
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        # Extract features using the backbone
        features = self.backbone(x)
        
        # Retrieve features from the backbone
        x5 = features[-1]  
        x4 = features[-2]  
        x3 = features[-3]  
        
        score = self.deconv1(x5)  
        score = self.bn1(F.interpolate(score, size=x4.shape[2:], mode='bilinear', align_corners=True) + self.conv1x1_x4(x4))  # Element-wise add, resize score to match x4
        
        score = self.deconv2(score)            
        score = self.bn2(F.interpolate(score, size=x3.shape[2:], mode='bilinear', align_corners=True) + self.conv1x1_x3(x3))  # Element-wise add, resize score to match x3

        
        score = self.bn3(self.deconv3(score)) 
        score = self.bn4(self.deconv4(score))  
        score = self.bn5(self.deconv5(score))  
        score = self.classifier(score)        

        return score  # size=(N, n_class, x.H, x.W)