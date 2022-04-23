import torch
x = torch.FloatTensor([[1., 2.]])
w1 = torch.FloatTensor([[2.], [1.]])
w2 = torch.FloatTensor([3.])
w1.requires_grad = True
w2.requires_grad = True

d = torch.matmul(x, w1)
d[:] = 1   # 稍微调换一下位置, 就没有问题了
f = torch.matmul(d, w2)
f.backward()
print(w2.grad)