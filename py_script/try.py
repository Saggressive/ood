import torch
if __name__=="__main__":
    a=torch.tensor(1)
    print(torch.typename(a))
    a.float()
    print(torch.typename(a))
