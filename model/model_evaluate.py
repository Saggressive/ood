import torch
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report
import numpy
import json
from tqdm import tqdm
import json
import numpy as np
def evaluate_base(predict: numpy.ndarray,labels: numpy.ndarray) -> tuple:

    acc=accuracy_score(labels,predict)
    # precision=precision_score(labels,predict,average="macro",)
    # recall=recall_score(labels,predict,average="macro")
    f1=f1_score(labels,predict,average="macro")
    # print(classification_report(labels,predict))
    return acc,f1
    # return acc
def evaluate_acc(predict: numpy.ndarray,labels: numpy.ndarray) -> tuple:

    acc=accuracy_score(labels,predict)
    # precision=precision_score(labels,predict,average="macro",)
    # recall=recall_score(labels,predict,average="macro")
    # f1=f1_score(labels,predict,average="macro")
    # return acc,precision,recall,f1
    return acc

def evaluate_myacc(predict: numpy.ndarray,labels: numpy.ndarray,sum_number: list):
    true_number=0
    i=0
    print(predict)
    print(labels)
    for num in sum_number:
        flag=True
        for index in range(num):
            if(predict[i]<labels[i]):
                flag=False
            i+=1
        if flag:
            true_number+=1
    return true_number*1.0/len(sum_number)

def get_best_acc(config,model,val_loader) :

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    all_predict, all_labels, all_scores = torch.tensor([]), torch.tensor([]), torch.tensor([])
    bar= tqdm(range(len(val_loader)))
    for i, batch in enumerate(val_loader):
        with torch.no_grad():
            output = model(input_ids=batch["input_ids"].to(device),
                           attention_mask=batch["attention_mask"].to(device),
                           labels=batch["labels"].to(device))
            logits = output.logits.view(-1, config["num_labels"])
            predict_scores = logits[:,1]
            all_scores = torch.cat([all_scores, predict_scores.view(-1).cpu()])
            labels = batch["labels"].cpu().detach()
            all_labels = torch.cat([all_labels, torch.argmax(labels,dim=1)])
        bar.update()
    roberta_scores = all_scores.numpy()
    op0 = np.linspace(0, 1, num=30, endpoint=True, retstep=False, dtype=None)
    op1 = np.linspace(0, 1, num=30, endpoint=True, retstep=False, dtype=None)
    with open(config["t5_score_path"], "r") as f:
        t5_scores = json.load(f)
        t5_scores = np.array(t5_scores)
    best_acc, best_op = -1, -1
    for j in op0:
        for i in op1:
            scores = j * t5_scores + i * roberta_scores
            max_index_group = []
            for index, score in enumerate(scores):
                if index % config["beam_num"] == 0:
                    max_index = index
                    max_num = scores[max_index]
                else:
                    if score > max_num:
                        max_index = index
                        max_num = scores[max_index]
                if (index + 1) % config["beam_num"]== 0:
                    max_index_group.append(max_index)
            assert len(max_index_group) == 1034
            max_labels_group = [all_labels[i] for i in max_index_group]
            acc = sum(max_labels_group) / len(max_labels_group)
            if acc > best_acc:
                best_acc = acc
                best_groups=max_index_group
    return best_acc,best_groups