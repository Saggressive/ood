import numpy as np
from sklearn.metrics import f1_score,classification_report
import torch
from torch.nn import CrossEntropyLoss
if __name__=="__main__":
    a=torch.tensor([1,2])
    b=torch.tensor([[0.1,0.9],[0.9,0.1]])
    loss=CrossEntropyLoss()
    print(loss(b,a))