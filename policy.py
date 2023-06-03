import torch.nn as nn


class ActorNet(nn.Module):
    def __init__(self):
        super(ActorNet, self).__init__()
        # feature extractor aka base encoder network
        self.f = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2),  # 32x29x29
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),  # 32x27x27
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),  # 32x25x25
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),  # 32x23x23
            nn.Flatten(),  # 16928

        )

        # projection head (for the contrastive loss)
        self.g = nn.Sequential(
            nn.ReLU(),
            nn.Linear(16928, 50),
        )

        # The downstream task (the actual actor)
        self.d = nn.Sequential(
            nn.ReLU(),
            nn.Linear(16928, 1024),
            nn.ReLU(),
            nn.Linear(16928, 17),
        )

    def forward(self, x, contrastive: bool):
        h = self.f(x)
        if contrastive:
            return self.g(h)
        else:
            return self.d(h)
