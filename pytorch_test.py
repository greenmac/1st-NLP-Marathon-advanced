import torch
import torchvision

print("判斷是否可用GPU運算：", torch.cuda.is_available())
print('='*50)

a = torch.tensor([[1, 2], [3, 4], [5, 6]])
print(a)
