import torch.nn as nn


class ActorNet(nn.Module):
    def __init__(self):
        super(ActorNet, self).__init__()
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
            nn.Linear(64 * 5 * 5, 256),
            nn.ReLU(),
            # nn.Dropout(p=.4),
        )

        # projection head (for the contrastive loss)
        self.g = nn.Sequential(
            nn.Linear(256, 64),
        )

        # The downstream task (the actual actor)
        self.d = nn.Sequential(
            nn.Linear(256, 17),  # Crafter has 17 actions
            # nn.Dropout(p=.4)
        )

    def disable_embedding_weights(self):
        for p in self.f.parameters():
            p.requires_grad = False
        for p in self.g.parameters():
            p.requires_grad = False

    def forward(self, x, contrastive: bool, full_network: bool = False):
        h = self.f(x)
        if contrastive:
            return self.g(h)
        else:
            if full_network:
                return self.d(h)
            else:
                # Stop gradient backpropagation from downstream task layer into embedding
                return self.d(h.detach())
