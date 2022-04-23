import torch
if __name__=="__main__":
    import torch

    a = torch.tensor(1.0, requires_grad=True)
    a_ = a.clone().detach()
    a_.requires_grad=True
    y = a ** 2
    # p=2*y
    # p.backward()
    # print(y.grad)
    b=a_ * 3
    z = a ** 2 + b
    y.backward()
    print(a.grad)  # 2
    z.backward()
    print(a_.grad)
    print(a.grad)
    print(b.grad)
    '''
    输出：
    tensor(2.) 
    None
    tensor(7.) # 2*2+3=7
    '''


