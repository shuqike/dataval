import torch


class Vestimator_1(torch.nn.Module):
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
        x = torch.cat([x, y], dim=1)
        x = self.net_(x)
        x = torch.cat([x, y_hat_diff], dim=1)
        output = self.combine_layer(x)
        return output
