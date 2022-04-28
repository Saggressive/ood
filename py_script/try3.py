import torch
import os
import sys
os.chdir("../")
sys.path.append("./")
from processdata.pre_process import tokenizer_process,process,get_label_dict
from processdata.subwords_enhance import get_four_fold,get_two
from model.ood_model import BertForSequenceClassification
from model.datatset import synthesis_data,DataCollator
from datasets import load_dataset
from transformers import BertTokenizer,DataCollatorWithPadding
from transformers import AdamW,get_cosine_with_hard_restarts_schedule_with_warmup,get_linear_schedule_with_warmup
from torch.utils.data import DataLoader,WeightedRandomSampler
from tqdm import tqdm
from model.model_evaluate import evaluate_base,get_best_acc
import shutil
import json
from torch.nn.functional import softmax

import warnings
warnings.filterwarnings("ignore")
def test(config,model,val_loader,device,labels_dict):
    model.eval()
    model.to(device)
    # epoch_val_loss = 0
    print("test")
    all_predict, all_labels, all_scores ,all_binary_predict= torch.tensor([]), torch.tensor([]), torch.tensor([]),torch.tensor([])
    all_binary_labels = torch.tensor([])
    for i, batch in enumerate(val_loader):
        with torch.no_grad():
            output = model(input_ids=batch["input_ids"].to(device),
                           attention_mask=batch["attention_mask"].to(device),
                           labels=batch["labels"].to(device), binary_labels=batch["binary_labels"].to(device),
                           alpha=config["alpha"], beta=config["beta"],
                           batch_size=config["batch_size"],ood_label=len(labels_dict.keys()),tmp=config["val_tmp"])
            logits = output.logits.view(-1, config["num_labels"])
            logits = torch.div(logits, config["val_tmp"])
            logits = softmax(logits, dim=1)
            binary_logits = output.binary_logits.view(-1, 2)
            binary_predict_labels = torch.argmax(binary_logits, dim=1).cpu()
            # epoch_val_loss += (output.loss.detach().item()) * len(batch["labels"])
            max_values_indices = torch.max(logits, dim=1)
            predict_labels = max_values_indices[1].cpu().detach()
            predict_scores = max_values_indices[0].cpu().detach()
            labels = batch["labels"].cpu()
            binary_labels = batch["binary_labels"].cpu()
            append_label = []
            for label_index, label in enumerate(labels):
                label_binary = binary_labels[label_index]
                if label_binary == torch.tensor(1):
                    assert label == len(labels_dict.keys())
                    append_label.append(torch.tensor(len(labels_dict.keys())))
                    # print(label)
                else:
                    append_label.append(label)
            append_label = torch.tensor(append_label)
            all_labels = torch.cat([all_labels, append_label])
            all_predict = torch.cat([all_predict, predict_labels])
            all_binary_predict = torch.cat([all_binary_predict, binary_predict_labels])
            all_scores = torch.cat([all_scores, predict_scores])
            all_binary_labels = torch.cat([all_binary_labels, binary_labels])

    all_predict, all_labels = all_predict.numpy(), all_labels.numpy()

    # for index, item in enumerate(all_binary_predict.clone()):
    #     if item == torch.tensor(1):
    #         all_predict[index] = len(labels_dict.keys())
    acc, f1_list = evaluate_base(all_predict, all_labels,mode='None')
    f1=sum(f1_list)/len(f1_list)
    know_f1=sum(f1_list[0:-1])/len(f1_list[0:-1])
    unknow_f1=f1_list[-1]
    ood_labels_binary = (all_labels == len(labels_dict.keys()))
    ood_labels = all_labels[ood_labels_binary]
    ood_predict = all_predict[ood_labels_binary]
    ood_acc, ood_f1 = evaluate_base(ood_predict, ood_labels,mode="weighted")

    id_labels = all_labels[~ood_labels_binary]
    id_predict = all_predict[~ood_labels_binary]
    id_acc, id_f1 = evaluate_base(id_predict, id_labels)
    all_binary_labels, all_binary_predict = all_binary_labels.numpy(), all_binary_predict.numpy()
    binary_acc, binary_f1 = evaluate_base(all_binary_predict, all_binary_labels)


    # return acc,f1,epoch_val_loss
    print("test")
    print(f"{acc},{f1}")
    print(f"ood:{ood_acc},{unknow_f1}")
    print(f"id:{id_acc},{know_f1}")
    print(f"binary:{binary_acc},{binary_f1}")
    return acc, f1, ood_acc, unknow_f1