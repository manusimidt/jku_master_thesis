import torch.nn as nn


# TODO network architecutre is way to big (seep page 20 of the paper)
# conv1: filter: 8x8 stride:4 32x14x14
# conv2: filter: 4x4 stride:2 64x6x6
# conv3: filter: 3x3 stride:1 64x4x4
# fc: 1024x256

class ActorNet(nn.Module):
    def __init__(self):
        super(ActorNet, self).__init__()
        # feature extractor aka base encoder network
        self.f = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4),  # 32x14x14
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),  # 64x6x6
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),  # 64x4x4
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
        )

        # projection head (for the contrastive loss)
        self.g = nn.Sequential(
            nn.Linear(256, 50),
        )

        # The downstream task (the actual actor)
        self.d = nn.Sequential(
            nn.Linear(256, 2),
        )

    def disable_embedding_weights(self):
        for p in self.f.parameters():
            p.requires_grad = False
        for p in self.g.parameters():
            p.requires_grad = False

    def forward(self, x, contrastive: bool):
        h = self.f(x)
        if contrastive:
            return self.g(h)
        else:
            return self.d(h)
