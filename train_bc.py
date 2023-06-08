import numpy as np

from common import set_seed
from env import VanillaEnv, TRAIN_CONFIGURATIONS, BCDataset, generate_bc_dataset
from policy import ActorNet
from rl.common.buffer2 import Transition, Episode
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List
from rl.ppo.ppo import PPO


@torch.enable_grad()
def train(net: ActorNet, dataLoader: DataLoader, optim_actor, loss_actor=nn.CrossEntropyLoss()):
    device = next(net.parameters()).device
    net.train()
    actor_errors = []

    for batch in dataLoader:
        # X is the observation
        # y contains the choosen action and the return estimate from the critic
        X, y = batch[0].to(device), batch[1].to(device)

        pred_action_logits = net.forward(X, contrastive=False, full_network=True)

        actor_error = loss_actor(pred_action_logits, y.to(torch.int64))

        optim_actor.zero_grad()
        actor_error.backward()
        optim_actor.step()

        actor_errors.append(actor_error.item())

    return actor_errors


@torch.no_grad()
def evaluate(net: ActorNet, dataLoader: DataLoader, loss_actor):
    device = next(net.parameters()).device
    net.eval()
    actor_errors = []

    for batch in dataLoader:
        X, y = batch[0].to(device), batch[1].to(device)
        pred_action_logits = net.forward(X, contrastive=False)
        actor_errors.append(loss_actor(pred_action_logits, y.to(torch.int64)).item())

    return actor_errors


if __name__ == '__main__':
    seed = 31
    set_seed(seed, env=None)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training on ", device)

    lr = 0.001
    batch_size = 256
    batch_count = 100
    n_epochs = 20
    balanced = True
    environment = 'vanilla_env'
    base_model = ''

    model = ActorNet().to(device)
    if len(base_model) > 0:
        print("Loading model ", base_model)
        ckp = torch.load(base_model, map_location=device)
        conf = ckp['info']['conf']

        model.load_state_dict(ckp['state_dict'])
        model.disable_embedding_weights()
        optim_actor = optim.Adam(model.parameters(), lr=lr)

    else:
        print("Training BC model without any contrastive learning")
        optim_actor = optim.Adam(model.parameters(), lr=lr)
        base_model = './ckpts/pure_bc_without_pse-narrow_grid.pth'
        conf = TRAIN_CONFIGURATIONS['narrow_grid']

    env = VanillaEnv(configurations=list(conf))

    print(f"Generating expert episodes...")
    train_loader, test_loader = generate_bc_dataset([env], batch_size, batch_count, balanced=balanced)

    loss_actor = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        actor_errors = train(model, train_loader, optim_actor, loss_actor)
        val_actor_errors = evaluate(model, test_loader, loss_actor)

        print(
            f"""Epoch {epoch} Train errors: Actor {np.mean(actor_errors):.8f}, Val errors: Actor {np.mean(val_actor_errors):.8f} """)

    state = {
        'state_dict': model.state_dict(),
        'optimizer': optim_actor.state_dict(),
        'info': {'conf': conf}
    }
    torch.save(state, base_model.replace('.pth', '-bc.pth'))
