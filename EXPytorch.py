import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(1, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 1, bias=False)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def true_fun(X):
    noise = np.random.rand(X.shape[0]) * 0.4 - 0.2
    return np.cos(1.5 * np.pi * X) + X + noise


def plot_results(model):
    x = np.linspace(0, 5, 100)
    input_x = torch.from_numpy(x).float().unsqueeze(1)  # Creates a Tensor from a numpy.ndarray
    plt.plot(x, true_fun(x), label="Truth")
    plt.plot(x, model(input_x).detach().numpy(), label="Prediction")
    plt.legend(loc='lower right', fontsize=15)
    plt.xlim((0, 5))
    plt.ylim((-1, 5))
    plt.grid()
    plt.show()


def main():
    data_x = np.random.rand(10000) * 5
    model = Model()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    plot_results(model)

    for step in range(10000):
        batch_x = np.random.choice(data_x, 32)
        batch_x_tensor = torch.from_numpy(batch_x).float().unsqueeze(1)
        pred = model(batch_x_tensor)

        batch_y = true_fun(batch_x)
        truth = torch.from_numpy(batch_y).float().unsqueeze(1)
        loss = F.mse_loss(pred, truth)

        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

    plot_results(model)


if __name__ == '__main__':
    main()
