import numpy as np
from sklearn.metrics import f1_score,classification_report
import torch
from torch.nn import CrossEntropyLoss
if __name__=="__main__":
    a=torch.tensor([1,1])
    print(torch.div(torch.sum(a),2))