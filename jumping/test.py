import torch
import numpy as np
from common.training_helpers import cosine_similarity, contrastive_loss_repository
from common.psm import psm_f_fast, psm_f_fast_repo

# The embedding is two-dimensional, n_states x contrastive_loss_head size
np.random.seed(1)
e1 = (np.random.randint(low=0, high=255, size=(56, 64)) / 255).astype(np.float64)
e2 = (np.random.randint(low=0, high=255, size=(56, 64)) / 255).astype(np.float64)
e1[23:52, 45:50] = 0.01

# Dimension is (n_states,)
a1 = np.random.randint(0, 8, size=(56,))
a2 = np.random.randint(0, 8, size=(56,))
a2[10:20] = a1[30:40]  # just make the action sequences partly similar

temp = 0.1
gamma = 0.8

e1_tensor, e2_tensor = torch.from_numpy(e1), torch.from_numpy(e2)
a1_tensor, a2_tensor = torch.from_numpy(a1), torch.from_numpy(a2)

psm_matrix = psm_f_fast_repo(a1_tensor, a2_tensor, gamma)
sim_matrix = cosine_similarity(e1_tensor, e2_tensor)
loss = contrastive_loss_repository(sim_matrix, psm_matrix, temp)

# print("PSM MATRIX: ")
# print(psm_matrix.numpy())
print("SIM MATRIX: ")
print(sim_matrix)
print(e1)
print("LOSS: ")
print(loss.numpy())

print(cosine_similarity(torch.from_numpy(np.array([[.234, .345, .234], [.562, 2.5436, 7.2335]])),
                        torch.from_numpy(np.array([[.954, .156, .111], [.456, 1.245, 4.456]]))))
