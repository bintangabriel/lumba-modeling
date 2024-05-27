import torch
import torch.nn as nn

class FCN(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super(CustomDeepLabV3, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)
        self.model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
        
    def forward(self, x):
        return self.model(x)

# Example usage
if __name__ == "__main__":
    model = FCN(num_classes=1, pretrained=True)
    print(model)