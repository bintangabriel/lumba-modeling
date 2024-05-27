import torch
import torch.nn as nn

class DeepLabV3(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super(CustomDeepLabV3, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=pretrained)
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        
    def forward(self, x):
        return self.model(x)

# Example usage
if __name__ == "__main__":
    model = DeepLabV3(num_classes=1, pretrained=True)
    print(model)