import numpy as np
from sklearn.metrics import f1_score,classification_report
import torch
from torch.nn import CrossEntropyLoss
if __name__=="__main__":
    a=torch.tensor([0,1])
    print(a.argmax(dim=0))