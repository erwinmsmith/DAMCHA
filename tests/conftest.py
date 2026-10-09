import torch

# Tiny regression tensors run faster and deterministically without thread oversubscription.
torch.set_num_threads(1)
