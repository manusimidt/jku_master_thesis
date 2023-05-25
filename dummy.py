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
            nn.ReLU(),
            nn.Flatten(),  # 16928

        )

        # projection head (for the contrastive loss)
        self.g = nn.Sequential(
            nn.Linear(16928, 50),
            nn.Linear(50, 1024)
        )

        # The downstream task (the actual actor)
        self.d = nn.Sequential(
            nn.Linear(16928, 1024),
            nn.ReLU(),
            nn.Linear(16928, 17),
        )


"""
First train f(.)  and g(.) using contrastive learning and the PSM

After this train f(.) and d(.) using standard PPO 

- iterate over these two? 
"""
