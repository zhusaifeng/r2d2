import torch
print(torch.__version__)
print(torch.cuda.is_available())
import torch
pt = torch.load('models/r2d2_WASF_N16.pt')
print(pt)
