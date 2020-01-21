import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as vmodels
from base import BaseModel
import copy

class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class Resnet_fc(nn.Module):
    def __init__(self, nb_classes=0, toFreeze=False):
        super(Resnet_fc, self).__init__()

        base_model = vmodels.resnet50(pretrained=True)
        base_model_copy = copy.deepcopy(base_model)
        self.feature_extractor = nn.Sequential(*list(base_model_copy.children())[:-2])

        if toFreeze:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        else:
            for param in self.feature_extractor.parameters():
                param.requires_grad = True

        self.gap = nn.AvgPool2d(7, 1)
        self.linear = nn.Linear(2048, nb_classes)

    def forward(self, inputs):
        x = self.feature_extractor(inputs)
        x = self.gap(x).squeeze(-1).squeeze(-1)
        x = self.linear(x)

        return x
