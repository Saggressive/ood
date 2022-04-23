import numpy as np
from sklearn.metrics import f1_score,classification_report
import torch
from torch.nn import CrossEntropyLoss
if __name__=="__main__":
    loss=CrossEntropyLoss(reduction='none')
    y_label=torch.tensor([1,0])
    # y_binary=torch.tensor(y_label,dtype=torch.bool)
    y_pred=torch.tensor([[0.05,0.9,0.05],[0.1,0.8,0.1]])
    y_pred_b = torch.tensor([[0.05, 0.9], [0.1, 0.8]])
    # l=loss(y_pred,y_label)
    # print(l)
    # print(y_pred[(1-y_label).bool()])
    print(torch.cat([y_pred,y_pred_b],dim=1))