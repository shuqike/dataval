import typing
import torch
import torchvision


class Vestimator(torch.nn.Module):
    def __init__(self, input_dim, layer_num, hidden_num, output_dim):
        super().__init__()

        layers = [torch.nn.Linear(input_dim, hidden_num), torch.nn.ReLU(inplace=True)]
        for _ in range(layer_num - 2):
            layers.append(torch.nn.Linear(hidden_num, hidden_num))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Linear(hidden_num, output_dim))
        layers.append(torch.nn.ReLU(inplace=True))

        self.net_ = torch.nn.Sequential(*layers)
        self.combine_layer = torch.nn.Sequential(
            torch.nn.Linear(output_dim*2, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x, y, y_hat_diff):
        """Feed forward.
        :param x: train_features
        :param y: train_labels
        :param y_hat_diff: l1 difference between predicion and grountruth
        """
        x = torch.flatten(x, start_dim=2)
        x = torch.cat([x, y], dim=2)
        x = self.net_(x.float())
        x = torch.cat([x, y_hat_diff], dim=2)
        output = self.combine_layer(x)
        return output


class Vestimator_1(Vestimator):
    def __init__(self, input_dim, layer_num, hidden_num, output_dim):
        super().__init__(input_dim, layer_num, hidden_num, output_dim)
        self.combine_layer = torch.nn.Sequential(
            torch.nn.Linear(output_dim*3, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x, y, y_hat_diff, weak_y_hat_diff):
        """Feed forward.
        :param x: train_features
        :param y: train_labels
        :param y_hat_diff: l1 difference between predicion and grountruth
        :param y_hat_diff: l1 difference between weak predicion and grountruth
        """
        x = torch.flatten(x, start_dim=2)
        x = torch.cat([x, y], dim=1)
        x = self.net_(x)
        x = torch.cat([x, y_hat_diff, weak_y_hat_diff], dim=1)
        output = self.combine_layer(x)
        return output


class BasicBlock(torch.nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1,
        downsample: torch.nn.Module = None
    ) -> None:
        super(BasicBlock, self).__init__()
        # Multiplicative factor for the subsequent conv2d layer's output channels.
        # It is 1 for ResNet18 and ResNet34.
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = torch.nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1,
            bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(
            out_channels, 
            out_channels*self.expansion, 
            kernel_size=3, 
            padding=1,
            bias=False
        )
        self.bn2 = torch.nn.BatchNorm2d(out_channels*self.expansion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return  out


class ResVestimatorMNIST(torch.nn.Module):
    def __init__(self):
        super(ResVestimatorMNIST, self).__init__()
        self.expansion = 1
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=1), 
            torch.nn.MaxPool2d(kernel_size=2, stride=2), 
            self._make_layer(BasicBlock, 20, 20, 1, 2), 
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=990, out_features=10), 
            torch.nn.ReLU(inplace=True)
        )
        self.combine_layer = torch.nn.Sequential(
            torch.nn.Linear(20, 1),
            torch.nn.Sigmoid()
        )

    def _make_layer(
        self, 
        block: typing.Type[BasicBlock],
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: int = 1
    ) -> torch.nn.Sequential:
        downsample = None
        if stride != 1:
            """
            This should pass from `layer2` to `layer4` or 
            when building ResNets50 and above. Section 3.3 of the paper
            Deep Residual Learning for Image Recognition
            (https://arxiv.org/pdf/1512.03385v1.pdf).
            """
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels, 
                    out_channels*self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False 
                ),
                torch.nn.BatchNorm2d(out_channels * self.expansion),
            )
        layers = []
        layers.append(
            block(
                in_channels, out_channels, stride, self.expansion, downsample
            )
        )
        in_channels = out_channels * self.expansion
        for i in range(1, blocks):
            layers.append(block(
                in_channels,
                out_channels,
                expansion=self.expansion
            ))
        return torch.nn.Sequential(*layers)

    def forward(self, x, y, y_hat_diff):
        x = self.conv(x.float())
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = torch.cat([x, y], dim=1)
        x = self.fc(x.float())
        x = torch.cat([x, y_hat_diff], dim=1)
        output = self.combine_layer(x)
        return output

class R18VestimatorMNIST(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torchvision.models.resnet18(num_classes=10)
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=20, out_features=10), 
            torch.nn.ReLU(inplace=True)
        )
        self.combine_layer = torch.nn.Sequential(
            torch.nn.Linear(20, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x, y, y_hat_diff):
        x = self.model(x.float())
        x = torch.cat([x, y], dim=1)
        x = self.fc(x.float())
        x = torch.cat([x, y_hat_diff], dim=1)
        x = self.combine_layer(x)
        return x
