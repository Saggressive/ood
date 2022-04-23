import torch
def soft_logits(input : torch.Tensor ,target : torch.Tensor ,mode:str="average",tmp: float = 1):
    input=torch.div(input,tmp)
    denominator = torch.log(torch.sum(torch.exp(input), dim=1))
    log_pro = -input + denominator.view(-1, 1)
    soft_scores = torch.sum(torch.mul(target,log_pro),dim=1)
    if mode=="average":
        return torch.mean(soft_scores)
    elif mode=="sum":
        return torch.sum(soft_scores)
    elif mode=="no reduction":
        return soft_scores
    else:
        raise ValueError("loss mode error")
if __name__=="__main__":
    a=torch.tensor([[0.01,0.98,0.01]])
    b=torch.tensor([[1.0,0,0]])
    # a=torch.div(a,10)
    loss=torch.nn.CrossEntropyLoss()
    print(soft_logits(a,b))
    print(loss(a,b))
