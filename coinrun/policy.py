import torch.nn as nn


class CoinRunActor(nn.Module):
    def __init__(self):
        super(CoinRunActor, self).__init__()
        # feature extractor aka base encoder network
        self.f = nn.Sequential(
            # in 3 x 64 x 64
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2),  # 32x31x31
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),  # 64x15x15
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=3),  # 64x5x5
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(64 * 5 * 5, 128),
        )

        # projection head (for the contrastive loss)
        self.g = nn.Sequential(
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
        )

        # The downstream task (the actual actor)
        self.d = nn.Sequential(
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 15),
        )

    def disable_embedding_weights(self):
        for p in self.f.parameters():
            p.requires_grad = False
        for p in self.g.parameters():
            p.requires_grad = False

    def forward(self, x, contrastive: bool = False, full_network: bool = True):
        h = self.f(x)
        if contrastive:
            return self.g(h)
        else:
            if full_network:
                return self.d(h)
            else:
                # Stop gradient backpropagation from downstream task layer into embedding
                return self.d(h)
                # return self.d(h.detach())


class CoinRunCritic(nn.Module):
    def __init__(self):
        super(CoinRunCritic, self).__init__()
        # feature extractor aka base encoder network
        self.f = nn.Sequential(
            # in 3 x 64 x 64
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2),  # 32x30x30
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2),  # 64x13x13
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2),  # 64x5x5
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(64 * 5 * 5, 512),
            nn.ReLU(),
            #nn.LayerNorm(512),
            # nn.Dropout(p=.4),
        )
        self.d = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        h = self.f(x)
        return self.d(h)
