import torch
import os
import sys
os.chdir("../")
sys.path.append("./")
from processdata.pre_process import tokenizer_process,process,get_label_dict
from model.ood_model import BertForSequenceClassification
from datasets import load_dataset
from transformers import BertTokenizer,DataCollatorWithPadding
from transformers import AdamW,get_cosine_with_hard_restarts_schedule_with_warmup
from torch.utils.data import DataLoader,WeightedRandomSampler
from tqdm import tqdm
from model.model_evaluate import evaluate_base,get_best_acc
import shutil
import json
from torch.nn.functional import softmax
import tensorboard_logger as tb_logger
import numpy as np
import warnings
warnings.filterwarnings("ignore")
def train(config:dict=None) ->None:
    logger = tb_logger.Logger(logdir=config["tb_folder"], flush_secs=2)
    labels_dict=get_label_dict(config["data_path"],config["know_rate"])
    tokenizer = BertTokenizer.from_pretrained(config["pretrained_path"])
    model = BertForSequenceClassification.from_pretrained(config["pretrained_path"], num_labels=len(labels_dict.keys()))

    datacollator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="max_length",
        max_length=config["token_length"],
        pad_to_multiple_of=8,
        return_tensors='pt'
    )

    config["num_labels"]=len(labels_dict.keys())
    use_soft=config["use_soft"]
    dataset = load_dataset(config["train_script_path"],use_soft=use_soft,labels_dict=labels_dict,cache_dir="./cache")
    train_dataset,val_dataset,test_dataset=dataset["train"],dataset["validation"],dataset["test"]
    if config["use_neg"]:
        neg_dataset=load_dataset(config["neg_script_path"],use_soft=use_soft,labels_dict=labels_dict,cache_dir="./cache")
        neg_dataset=neg_dataset["train"]
        neg_dataset=neg_dataset.map(
            lambda batch: tokenizer_process(batch, tokenizer, config["token_length"]),
            batched=True,
            num_proc=1,
            load_from_cache_file=False  # not data_training_args.overwrite_cache,
        )
        neg_dataset=neg_dataset.remove_columns(['text','label'])
        neg_loader = DataLoader(neg_dataset, batch_size=config["batch_size"], collate_fn=datacollator, shuffle=True)
    train_dataset = train_dataset.map(
        lambda batch: tokenizer_process(batch, tokenizer, config["token_length"]),
        batched=True,
        num_proc=1,
        load_from_cache_file=False  # not data_training_args.overwrite_cache,
    )

    # if config["use_balance"]==False:#如果是使用soft-softmax还要做修改
    #     target = []
    #     for i in train_dataset:
    #         target.append(i["labels"])
    #     target = torch.tensor(target)
    #     class_sample_count = torch.tensor(
    #         [(target == t).sum() for t in torch.unique(target, sorted=True)])
    #     weight = 1. / class_sample_count.float()
    #     samples_weight = torch.tensor([weight[t["labels"]] for t in train_dataset])
    #     sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

    val_dataset=val_dataset.map(
        lambda batch:tokenizer_process(batch,tokenizer,config["token_length"]),
        batched=True,
        num_proc=1,
        load_from_cache_file=False
    )
    test_dataset=test_dataset.map(
        lambda batch: tokenizer_process(batch, tokenizer, config["token_length"]),
        batched=True,
        num_proc=1,
        load_from_cache_file=False
    )
    train_dataset,val_dataset,test_dataset=train_dataset.remove_columns(['text','label']),\
                                           val_dataset.remove_columns(['text','label']),test_dataset.remove_columns(['text','label'])

    # if config["use_balance"]:
    #     train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], collate_fn=datacollator,
    #                               sampler=sampler)
    # else:
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], collate_fn=datacollator,shuffle=True)
    val_loader=DataLoader(val_dataset,batch_size=config["batch_size"],shuffle=False,collate_fn=datacollator)
    test_loader=DataLoader(test_dataset,batch_size=config["batch_size"],shuffle=False,collate_fn=datacollator)
    params_classifer, params_roberta = [], []
    for i in model.named_parameters():
        if "classifier" in i[0]:
            params_classifer.append(i[1])
        else:
            params_roberta.append(i[1])
    optimizer=AdamW([{'params':params_roberta},{'params': params_classifer, 'lr': config["linear_lr"],"weight_decay":config["linear_decay"]}],
                    lr=config["lr"], weight_decay=config["weight_decay"])
    num_training_steps = config["epoch"] * len(train_loader)
    lr_steps=int(num_training_steps/config["accumulation_steps"])
    lr_scheduler =get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
                    num_warmup_steps=lr_steps*0.01,num_training_steps=lr_steps,num_cycles=3)

    progress_bar = tqdm(range(num_training_steps))
    device="cuda" if torch.cuda.is_available() else "cpu"
    best_val_f1,best_val_acc=-1,-1
    best_test_f1,best_test_acc=-1,-1
    print(device)
    model.to(device)

    if os.path.exists(config["save_dir"]) == False:
        os.makedirs(config["save_dir"])

    labels_dict_path = config["save_dir"] + os.sep + 'dict.txt'
    with open(labels_dict_path, "w") as f:
        json.dump(labels_dict, f, indent=4)

    for index in range(config["epoch"]):
        model.train()
        epoch_train_loss=0
        if config["use_neg"]==True:
            for i, (pos_batch,neg_batch) in enumerate(zip(train_loader,neg_loader)):
                batch={}
                batch["input_ids"]=torch.cat([pos_batch["input_ids"],neg_batch["input_ids"]],dim=0)
                batch["attention_mask"]=torch.cat([pos_batch["attention_mask"],neg_batch["attention_mask"]],dim=0)
                batch["labels"]=torch.cat([pos_batch["labels"],neg_batch["labels"]],dim=0)
                output = model(input_ids=batch["input_ids"].to(device),
                               attention_mask=batch["attention_mask"].to(device),
                               labels=batch["labels"].to(device), mode="train", use_soft=use_soft,tmp=config["tmp"])
                ce_loss = output.loss

                with torch.no_grad():
                    epoch_train_loss += ce_loss.cpu().item() * len(batch["labels"])

                ce_loss = ce_loss / config["accumulation_steps"]
                ce_loss.backward()
                # accelerator.backward(loss)
                if ((i + 1) % config["accumulation_steps"]) == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()
                progress_bar.update(1)
        else:
            for i, batch in enumerate(train_loader):
                output = model(input_ids=batch["input_ids"].to(device),
                               attention_mask=batch["attention_mask"].to(device),
                               labels=batch["labels"].to(device), mode="train", use_soft=True,tmp=config["tmp"])
                ce_loss = output.loss

                with torch.no_grad():
                    epoch_train_loss += ce_loss.cpu().item() * len(batch["labels"])

                ce_loss = ce_loss / config["accumulation_steps"]
                ce_loss.backward()
                # accelerator.backward(loss)
                if ((i + 1) % config["accumulation_steps"]) == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()
                progress_bar.update(1)


        val_acc,val_f1,op=eval(config, model, val_loader, device,labels_dict)
        mean_train_loss=epoch_train_loss/len(train_dataset)
        logger.log_value("train loss",mean_train_loss,index)
        # logger.log_value("val loss",mean_val_loss,index)
        logger.log_value("f1",val_f1,index)
        logger.log_value("acc",val_acc,index)
        if index==0 and os.path.exists(config["val_result_path"]):
            os.remove(config["val_result_path"])
        if index==0 and os.path.exists(config["test_result_path"]):
            os.remove(config["test_result_path"])

        with open(config["val_result_path"], "a") as f:
            f.write(f"epoch:{index},train loss:{mean_train_loss},acc:{val_acc},f1-macro:{val_f1}\n")

        if val_f1>best_val_f1:
            best_val_f1=val_f1
            best_val_acc=val_acc
            test_acc,test_f1,op=test(config,model,test_loader,device,labels_dict , op)

            logger.log_value("test acc",test_acc,index)
            logger.log_value("test f1",test_f1,index)
            with open(config["test_result_path"], "a") as f:
                f.write(
                    f"acc:{test_acc},f1-macro:{test_f1}\n")
                f.write(f"op{op}\n")
            if test_f1>best_test_f1 and config["is_save"]:
                tokenizer.save_pretrained(config["save_dir"])
                model.save_pretrained(config["save_dir"])
            if test_f1>best_test_f1:
                best_test_f1 = test_f1
                best_test_acc = test_acc
    with open(config["config_save_path"], "w") as f:
        json.dump(config, f, indent=4)


def eval(config,model,val_loader,device,labels_dict):
    model.eval()

    # epoch_val_loss = 0
    all_predict, all_labels, all_scores = torch.tensor([]), torch.tensor([]), torch.tensor([])
    for i, batch in enumerate(val_loader):
        if i==1:
            break
        with torch.no_grad():
            output = model(input_ids=batch["input_ids"].to(device),attention_mask=batch["attention_mask"].to(device),
                           labels=batch["labels"].to(device),mode="val",tmp=config["val_tmp"])
            logits = output.logits.view(-1, config["num_labels"])
            logits=torch.div(logits,config["val_tmp"])
            logits=softmax(logits,dim=1)
            # epoch_val_loss += (output.loss.detach().item()) * len(batch["labels"])
            max_values_indices = torch.max(logits, dim=1)
            predict_labels = max_values_indices[1].cpu().detach()
            predict_scores = max_values_indices[0].cpu().detach()
            labels = batch["labels"].cpu()
            append_label=[]
            if config["use_soft"]:
                for label_index,label in enumerate(labels):
                    label_binary = (label == torch.tensor(1 / len(labels_dict.keys())))
                    if torch.sum(label_binary).item() == len(labels_dict.keys()):
                        append_label.append(torch.tensor(len(labels_dict.keys())))
                        # print(label)
                    else:
                        append_label.append(torch.argmax(label))
                append_label=torch.tensor(append_label)
                all_labels = torch.cat([all_labels, append_label])
            else:
                all_labels = torch.cat([all_labels, labels])
            all_predict = torch.cat([all_predict, predict_labels])
            all_scores = torch.cat([all_scores, predict_scores])

    all_predict, all_labels = all_predict.numpy(), all_labels.numpy()
    thresolds = np.linspace(0, 1, 1000, endpoint=False)
    best_f1,best_op,best_acc=-1,-1,-1
    best_ood_acc,best_ood_f1=-1,-1
    for i,value in enumerate(thresolds):
        all_predict_clone=all_predict.copy()
        for index, item in enumerate(all_scores):
            if item <value:
                all_predict_clone[index]=len(labels_dict.keys())
        acc,  f1 = evaluate_base(all_predict_clone, all_labels)
        ood_labels_binary=(all_labels==len(labels_dict.keys()))
        ood_labels=all_labels[ood_labels_binary]
        ood_predict=all_predict_clone[ood_labels_binary]
        ood_acc,ood_f1=evaluate_base(ood_predict,ood_labels)
        if f1>best_f1:
            best_f1=f1
            best_acc=acc
            best_op=value
            best_ood_acc=ood_acc
            best_ood_f1=ood_f1
    # all_predict_clone=all_predict.copy()
    # for index, item in enumerate(all_scores):
    #     if item <0.06:
    #         all_predict_clone[index]=len(labels_dict.keys())
    #     # ood_labels_binary=(all_labels==len(labels_dict.keys()))
    #     # ood_labels=all_labels[ood_labels_binary]
    #     # ood_predict=all_predict[ood_labels_binary]
    # acc,  f1 = evaluate_base(all_predict_clone, all_labels)
    # if f1>best_f1:
    #     best_f1=f1
    #     best_acc=acc

    # return acc,f1,epoch_val_loss
    print(f"\nacc:{best_acc},f1:{best_f1},threshold:{best_op}")
    print(f"\nood acc:{best_ood_acc}")
    return best_acc,best_f1,best_op

def test(config,model,val_loader,device,labels_dict , op):
    model.eval()

    # epoch_val_loss = 0
    all_predict, all_labels, all_scores = torch.tensor([]), torch.tensor([]), torch.tensor([])

    for i, batch in enumerate(val_loader):
        with torch.no_grad():
            output = model(input_ids=batch["input_ids"].to(device),attention_mask=batch["attention_mask"].to(device),
                           labels=batch["labels"].to(device),mode="val",tmp=config["val_tmp"])
            logits = output.logits.view(-1, config["num_labels"])
            logits = torch.div(logits, config["val_tmp"])
            logits = softmax(logits, dim=1)
            # epoch_val_loss += (output.loss.detach().item()) * len(batch["labels"])
            max_values_indices = torch.max(logits, dim=1)
            predict_labels = max_values_indices[1].cpu().detach()
            predict_scores = max_values_indices[0].cpu().detach()
            labels = batch["labels"].cpu()
            append_label = []
            if config["use_soft"]:
                for label_index, label in enumerate(labels):
                    label_binary = (label == torch.tensor(1 / len(labels_dict.keys())))
                    if torch.sum(label_binary).item() == len(labels_dict.keys()):
                        append_label.append(torch.tensor(len(labels_dict.keys())))
                        # print(label)
                    else:
                        append_label.append(torch.argmax(label))
                append_label = torch.tensor(append_label)
                all_labels = torch.cat([all_labels, append_label])
            else:
                all_labels = torch.cat([all_labels, labels])
            all_predict = torch.cat([all_predict, predict_labels])
            all_scores = torch.cat([all_scores, predict_scores])

    all_predict, all_labels = all_predict.numpy(), all_labels.numpy()

    for index, item in enumerate(all_scores):
        if item <op:
            all_predict[index]=len(labels_dict.keys())
        acc,  f1 = evaluate_base(all_predict, all_labels)

    # return acc,f1,epoch_val_loss
    print(f"\n{acc},{f1}")
    return acc,f1,op

def main_test(config):
    # labels_dict = get_label_dict(config["data_path"], config["know_rate"])
    labels_dict_path = config["save_dir"] + os.sep + 'dict.txt'
    with open(labels_dict_path,"r") as f:
        labels_dict=json.load(f)
    print(labels_dict)
    tokenizer = BertTokenizer.from_pretrained(config["save_dir"])
    model = BertForSequenceClassification.from_pretrained(config["save_dir"], num_labels=len(labels_dict.keys()))
    config["num_labels"] = len(labels_dict.keys())
    dataset = load_dataset(config["train_script_path"], labels_dict=labels_dict,use_soft=True)
    train_dataset, val_dataset, test_dataset = dataset["train"], dataset["validation"], dataset["test"]

    test_dataset = test_dataset.map(
        lambda batch: tokenizer_process(batch, tokenizer, config["token_length"]),
        batched=True,
        num_proc=1,
        load_from_cache_file=False
    )
    test_dataset = test_dataset.remove_columns(['text', 'label'])
    datacollator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="max_length",
        max_length=config["token_length"],
        pad_to_multiple_of=8,
        return_tensors='pt'
    )


    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=datacollator)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model.to(device)

    test_acc, test_f1, op = eval(config, model, test_loader, device, labels_dict)
    # test(config, model, test_loader, device, labels_dict, 0.984)

if __name__=="__main__":
    #只需要修改save_dir 和 tb_folder
    save_dir="./best_model_idea_no_neg"
    config={
        "pretrained_path":"./bert-base-uncased",
        "data_path":"./oos",
        "know_rate":0.25,
        "train_script_path":"./dataset/oos_data",
        "neg_script_path":"./dataset/neg_data",
        "epoch":100,
        "accumulation_steps":1,
        "batch_size":256,
        "num_labels":2,
        "save_dir":save_dir,
        "is_save":True,
        "config_save_path":save_dir+os.sep+"ood_config.txt",
        "val_result_path":save_dir+os.sep+"result_val.txt",
        "test_result_path": save_dir + os.sep + "result_test.txt",
        "lr":1e-5,
        "weight_decay":1e-4,
        "linear_lr":1e-4,
        "linear_decay":1e-5,
        "use_balance":False,
        "tb_folder":"./tb_folder_idea_no_neg",
        "tmp":1,
        "val_tmp":0.5,
        "use_soft":True,
        "use_neg":True,
        "token_length":50,
    }
    # train(config)
    # eval(config)
    main_test(config)