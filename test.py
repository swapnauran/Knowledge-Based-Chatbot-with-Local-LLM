import torch
print(torch.__version__)  # Should show 2.2.0+cu118
print(torch.cuda.is_available())  # Should be True