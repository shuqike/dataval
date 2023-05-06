import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.classifier import Casifier
from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoModelForImageClassification, ResNetForImageClassification, ConvNextFeatureExtractor, ConvNextForImageClassification


class ReallySimple(torch.nn.Module):
    def __init__(self, input_dim=32) -> None:
        super(ReallySimple, self).__init__()
        self.layer = torch.nn.Linear(input_dim, 10)

    def forward(self, x):
        x = self.layer(x)
        x = F.softmax(x, dim=1)
        return x


class LeNetMNIST(torch.nn.Module):
    def __init__(self):
        super(LeNetMNIST, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, 5, 1)
        self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
        self.fc1 = torch.nn.Linear(4 * 4 * 50, 500)
        self.fc2 = torch.nn.Linear(500, 10)
        self.ceriation = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.max_pool2d(x, 2, 2)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.max_pool2d(x, 2, 2)
        x = torch.nn.functional.relu(x)
        x = x.view(-1, 4 * 4 * 50)
        x = self.fc1(x)
        x = self.fc2(x)
        x = torch.nn.functional.softmax(x, dim=1)
        return x


class LeNet10(torch.nn.Module):
    def __init__(self):
        super(LeNet10, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.softmax(self.fc3(x),dim=1)
        return x


class LeNet100(torch.nn.Module):
    def __init__(self):
        super(LeNet100, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 100)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.softmax(self.fc3(x),dim=1)
        return x


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class R9Cifar10(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4), 
                                        nn.Flatten(), 
                                        nn.Linear(512, num_classes))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


class MobileNet(Casifier):
    def _get_model(self):
        self._model = AutoModelForImageClassification.from_pretrained('google/mobilenet_v1_0.75_192')

    def _get_processor(self):
        self._processor = AutoImageProcessor.from_pretrained('google/mobilenet_v1_0.75_192')


class ResNet18(Casifier):
    def _get_model(self):
        self._model = ResNetForImageClassification.from_pretrained('microsoft/resnet-18')

    def _get_processor(self):
        self._processor = AutoFeatureExtractor.from_pretrained('microsoft/resnet-18')


class ResNet50(Casifier):
    def _get_model(self):
        self._model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

    def _get_processor(self):
        self._processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")


class ConvNeXTTiny(Casifier):
    def _get_model(self):
        self._model = ConvNextForImageClassification.from_pretrained("facebook/convnext-tiny-224")

    def _get_processor(self):
        self._processor = ConvNextFeatureExtractor.from_pretrained("facebook/convnext-tiny-224")
