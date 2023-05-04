import torch
from src.models.classifier import Casifier
from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoModelForImageClassification, ResNetForImageClassification, ConvNextFeatureExtractor, ConvNextForImageClassification


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
