import torch

if __name__=="__main__":
    a=torch.tensor([1,2,3,4,5,6])
    b = a[1:a.size()[0] - 1]
    b = b[torch.randperm(b.size(0))]
    a[1:a.size()[0] - 1] = b[:]
    print(a)