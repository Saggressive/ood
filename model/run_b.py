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
import tensorboard_logger as tb_logger
import numpy as np
import warnings
warnings.filterwarnings("ignore")
def train(config:dict=None) ->None:
    logger = tb_logger.Logger(logdir=config["tb_folder"], flush_secs=2)
    labels_dict=get_label_dict(config["data_path"],config["know_rate"])
    print(labels_dict)
    tokenizer = BertTokenizer.from_pretrained(config["pretrained_path"])
    model = BertForSequenceClassification.from_pretrained(config["pretrained_path"], num_labels=len(labels_dict.keys())+1)
    model.config.output_hidden_states = True
    datacollator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="max_length",
        max_length=config["token_length"],
        pad_to_multiple_of=8,
        return_tensors='pt'
    )

    config["num_labels"]=len(labels_dict.keys())+1
    dataset = load_dataset(config["train_script_path"],labels_dict=labels_dict,cache_dir=config["cache_dir"])
    train_dataset,val_dataset,test_dataset=dataset["train"],dataset["validation"],dataset["test"]
    if config["use_neg"]:
        neg_dataset=load_dataset(config["neg_script_path"],labels_dict=labels_dict,cache_dir=config["cache_dir"])
        neg_train_dataset,neg_val_dataset=neg_dataset["train"],neg_dataset["validation"]
        neg_train_dataset=neg_train_dataset.map(
            lambda batch: tokenizer_process(batch, tokenizer, config["token_length"]),
            batched=True,
            num_proc=1,
            load_from_cache_file=False  # not data_training_args.overwrite_cache,
        )
        neg_train_dataset=neg_train_dataset.remove_columns(['text','label','binary_label'])
        neg_train_loader = DataLoader(neg_train_dataset, batch_size=config["neg_multiple"]*config["batch_size"]//2, collate_fn=datacollator, shuffle=True)

        neg_val_dataset = neg_val_dataset.map(
            lambda batch: tokenizer_process(batch, tokenizer, config["token_length"]),
            batched=True,
            num_proc=1,
            load_from_cache_file=False  # not data_training_args.overwrite_cache,
        )
        neg_val_dataset = neg_val_dataset.remove_columns(['text', 'label', 'binary_label'])
        neg_val_loader = DataLoader(neg_val_dataset, batch_size=config["neg_multiple"]*config["batch_size"]//2, collate_fn=datacollator,
                                      shuffle=True)
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
    train_dataset,val_dataset,test_dataset=train_dataset.remove_columns(['text','label','binary_label']),\
        val_dataset.remove_columns(['text','label','binary_label']),test_dataset.remove_columns(['text','label','binary_label'])

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

    # lr_scheduler =get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
    #                 num_warmup_steps=lr_steps*0.01,num_training_steps=lr_steps,num_cycles=3)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,num_warmup_steps=lr_steps*0.01,
                                                   num_training_steps=lr_steps)
    # progress_bar = tqdm(range(num_training_steps))
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
    eval(config, model, val_loader, neg_val_loader, device, labels_dict)

    for index in range(config["epoch"]):
        model.train()
        epoch_train_loss=0
        dataset = synthesis_data("oos/train.tsv", labels_dict,config["neg_multiple"]*(len(train_dataset)+config["batch_size"]))
        collator = DataCollator(tokenizer, labels_dict, config)
        syn_dataloader = DataLoader(dataset, batch_size=config["batch_size"]*config["neg_multiple"], shuffle=True,
                                    collate_fn=collator, drop_last=True)

        if config["use_neg"]==True:
            # assert len(train_loader) <= len(neg_train_loader)
            # assert len(train_loader) <= len(syn_dataloader)
            for i, (pos_batch,neg_batch,syn_batch) in enumerate(zip(train_loader,neg_train_loader,syn_dataloader)):
                batch={}
                batch["input_ids"]=torch.cat([pos_batch["input_ids"],neg_batch["input_ids"]],dim=0)
                batch["attention_mask"]=torch.cat([pos_batch["attention_mask"],neg_batch["attention_mask"]],dim=0)
                batch["labels"]=torch.cat([pos_batch["labels"],neg_batch["labels"]],dim=0)
                batch["binary_labels"]=torch.cat([pos_batch["binary_labels"],neg_batch["binary_labels"]],dim=0)
                #子词级别
                synthesis_batch = get_two(pos_batch, config,labels_dict)
                batch["input_ids"]=torch.cat([batch["input_ids"],synthesis_batch["input_ids"]],dim=0)
                batch["attention_mask"]=torch.cat([batch["attention_mask"],synthesis_batch["attention_mask"]],dim=0)
                batch["labels"]=torch.cat([batch["labels"],synthesis_batch["labels"]],dim=0)
                batch["binary_labels"]=torch.cat([batch["binary_labels"],synthesis_batch["binary_labels"]],dim=0)
                #单词级别和句子级别
                batch["input_ids"] = torch.cat([batch["input_ids"], syn_batch["input_ids"]], dim=0)
                batch["attention_mask"] = torch.cat([batch["attention_mask"], syn_batch["attention_mask"]], dim=0)
                batch["labels"] = torch.cat([batch["labels"], syn_batch["labels"]], dim=0)
                batch["binary_labels"] = torch.cat([batch["binary_labels"], syn_batch["binary_labels"]], dim=0)

                output = model(input_ids=batch["input_ids"].to(device),
                               attention_mask=batch["attention_mask"].to(device),
                               labels=batch["labels"].to(device), binary_labels=batch["binary_labels"].to(device),
                               alpha=config["alpha"],beta=config["beta"],mode="train",
                               batch_size=config["batch_size"],ood_label=len(labels_dict.keys()),tmp=config["tmp"])
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
                # progress_bar.update(1)
        else:
            for i, batch in enumerate(train_loader):
                output = model(input_ids=batch["input_ids"].to(device),
                               attention_mask=batch["attention_mask"].to(device),
                               labels=batch["labels"].to(device),binary_labels=batch["binary_labels"].to(device),
                               alpha=config["alpha"],beta=config["beta"],mode="train",
                               batch_size=config["batch_size"],ood_label=len(labels_dict.keys()),tmp=config["tmp"])
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
                # progress_bar.update(1)


        val_acc,val_f1,val_oos_acc,val_oos_f1=eval(config, model, val_loader,neg_val_loader, device,labels_dict)
        mean_train_loss=epoch_train_loss/len(train_dataset)
        logger.log_value("train loss",mean_train_loss,index)
        # logger.log_value("val loss",mean_val_loss,index)
        logger.log_value("f1",val_f1,index)
        logger.log_value("acc",val_acc,index)
        logger.log_value("oos f1", val_oos_f1, index)
        logger.log_value("oos acc", val_oos_acc, index)
        if index==0 and os.path.exists(config["val_result_path"]):
            os.remove(config["val_result_path"])
        if index==0 and os.path.exists(config["test_result_path"]):
            os.remove(config["test_result_path"])

        with open(config["val_result_path"], "a") as f:
            f.write(f"epoch:{index},train loss:{mean_train_loss},acc:{val_acc},f1-macro:{val_f1}\n")

        if val_f1>best_val_f1:
            best_val_f1=val_f1
            best_val_acc=val_acc
        test_acc,test_f1,test_oos_acc,test_oos_f1=test(config,model,test_loader,device,labels_dict)

        logger.log_value("test acc",test_acc,index)
        logger.log_value("test f1",test_f1,index)
        logger.log_value("test oos acc", test_oos_acc, index)
        logger.log_value("test oos f1", test_oos_f1, index)
        with open(config["test_result_path"], "a") as f:
            f.write(
                f"acc:{test_acc},f1-macro:{test_f1}\n")
        if test_f1>best_test_f1 and config["is_save"]:
            tokenizer.save_pretrained(config["save_dir"])
            # model.save_pretrained(config["save_dir"])
            torch.save(model, config["model_save_path"])
        if test_f1>best_test_f1:
            best_test_f1 = test_f1
            best_test_acc = test_acc
    with open(config["config_save_path"], "w") as f:
        json.dump(config, f, indent=4)


# def test(config,model,val_loader,device,labels_dict):
#     model.eval()
#     model.to(device)
#     # epoch_val_loss = 0
#     print("test")
#     all_predict, all_labels, all_scores ,all_binary_predict= torch.tensor([]), torch.tensor([]), torch.tensor([]),torch.tensor([])
#     all_binary_labels = torch.tensor([])
#     for i, batch in enumerate(val_loader):
#         with torch.no_grad():
#             output = model(input_ids=batch["input_ids"].to(device),
#                            attention_mask=batch["attention_mask"].to(device),
#                            labels=batch["labels"].to(device), binary_labels=batch["binary_labels"].to(device),
#                            alpha=config["alpha"], beta=config["beta"],
#                            batch_size=config["batch_size"],ood_label=len(labels_dict.keys()),tmp=config["val_tmp"])
#             logits = output.logits.view(-1, config["num_labels"])
#             logits = torch.div(logits, config["val_tmp"])
#             logits = softmax(logits, dim=1)
#             binary_logits = output.binary_logits.view(-1, 2)
#             binary_predict_labels = torch.argmax(binary_logits, dim=1).cpu()
#             # epoch_val_loss += (output.loss.detach().item()) * len(batch["labels"])
#             max_values_indices = torch.max(logits, dim=1)
#             predict_labels = max_values_indices[1].cpu().detach()
#             predict_scores = max_values_indices[0].cpu().detach()
#             labels = batch["labels"].cpu()
#             binary_labels = batch["binary_labels"].cpu()
#             append_label = []
#             for label_index, label in enumerate(labels):
#                 label_binary = binary_labels[label_index]
#                 if label_binary == torch.tensor(1):
#                     assert label == len(labels_dict.keys())
#                     append_label.append(torch.tensor(len(labels_dict.keys())))
#                     # print(label)
#                 else:
#                     append_label.append(label)
#             append_label = torch.tensor(append_label)
#             all_labels = torch.cat([all_labels, append_label])
#             all_predict = torch.cat([all_predict, predict_labels])
#             all_binary_predict = torch.cat([all_binary_predict, binary_predict_labels])
#             all_scores = torch.cat([all_scores, predict_scores])
#             all_binary_labels = torch.cat([all_binary_labels, binary_labels])
#
#     all_predict, all_labels = all_predict.numpy(), all_labels.numpy()
#     best_f1, best_op, best_acc = -1, -1, -1
#     best_ood_acc, best_ood_f1 = -1, -1
#
#     # for index, item in enumerate(all_binary_predict.clone()):
#     #     if item == torch.tensor(1):
#     #         all_predict[index] = len(labels_dict.keys())
#     acc, f1 = evaluate_base(all_predict, all_labels)
#     ood_labels_binary = (all_labels == len(labels_dict.keys()))
#     ood_labels = all_labels[ood_labels_binary]
#     ood_predict = all_predict[ood_labels_binary]
#     ood_acc, ood_f1 = evaluate_base(ood_predict, ood_labels,mode="weighted")
#
#     id_labels = all_labels[~ood_labels_binary]
#     id_predict = all_predict[~ood_labels_binary]
#     id_acc, id_f1 = evaluate_base(id_predict, id_labels)
#     all_binary_labels, all_binary_predict = all_binary_labels.numpy(), all_binary_predict.numpy()
#     binary_acc, binary_f1 = evaluate_base(all_binary_predict, all_binary_labels)
#     if f1 > best_f1:
#         best_f1 = f1
#         best_acc = acc
#         best_ood_acc = ood_acc
#         best_ood_f1 = ood_f1
#
#     # return acc,f1,epoch_val_loss
#     print("test")
#     print(f"{best_acc},{best_f1}")
#     print(f"ood:{best_ood_acc},{best_ood_f1}")
#     print(f"id:{id_acc},{id_f1}")
#     print(f"binary:{binary_acc},{binary_f1}")
#     return best_acc, best_f1, best_ood_acc, best_ood_f1

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
    acc, f1_list = evaluate_base(all_predict, all_labels,mode=None)
    f1=np.mean(f1_list)
    know_f1=np.mean(f1_list[0:-1])
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
def eval(config,model,pos_val_loader,neg_val_loader,device,labels_dict):
    model.eval()
    model.to(device)
    print("eval")
    # epoch_val_loss = 0
    all_predict, all_labels, all_scores ,all_binary_predict= torch.tensor([]), torch.tensor([]), torch.tensor([]),torch.tensor([])
    all_binary_labels=torch.tensor([])
    for i, (pos_batch,neg_batch) in enumerate(zip(pos_val_loader,neg_val_loader)):
        batch={}
        batch["input_ids"] = torch.cat([pos_batch["input_ids"], neg_batch["input_ids"]], dim=0)
        batch["attention_mask"] = torch.cat([pos_batch["attention_mask"], neg_batch["attention_mask"]], dim=0)
        batch["labels"] = torch.cat([pos_batch["labels"], neg_batch["labels"]], dim=0)
        batch["binary_labels"]=torch.cat([pos_batch["binary_labels"], neg_batch["binary_labels"]], dim=0)
        with torch.no_grad():
            output = model(input_ids=batch["input_ids"].to(device),
                           attention_mask=batch["attention_mask"].to(device),
                           labels=batch["labels"].to(device), binary_labels=batch["binary_labels"].to(device),
                           alpha=config["alpha"], beta=config["beta"],
                           batch_size=config["batch_size"],ood_label=len(labels_dict.keys()),tmp=config["val_tmp"])
            logits = output.logits.view(-1, config["num_labels"])
            logits=torch.div(logits,config["val_tmp"])
            logits=softmax(logits,dim=1)
            binary_logits=output.binary_logits.view(-1,2)
            binary_predict_labels=torch.argmax(binary_logits,dim=1).cpu()
            # epoch_val_loss += (output.loss.detach().item()) * len(batch["labels"])
            max_values_indices = torch.max(logits, dim=1)
            predict_labels = max_values_indices[1].cpu().detach()
            predict_scores = max_values_indices[0].cpu().detach()
            labels = batch["labels"].cpu()
            binary_labels=batch["binary_labels"].cpu()
            append_label=[]
            for label_index, label in enumerate(labels):
                label_binary = binary_labels[label_index]
                if label_binary == torch.tensor(1):
                    assert label==len(labels_dict.keys())
                    append_label.append(torch.tensor(len(labels_dict.keys())))
                    # print(label)
                else:
                    append_label.append(label)
            append_label = torch.tensor(append_label)
            all_labels = torch.cat([all_labels, append_label])
            all_predict = torch.cat([all_predict, predict_labels])
            all_binary_predict=torch.cat([all_binary_predict,binary_predict_labels])
            all_scores = torch.cat([all_scores, predict_scores])
            all_binary_labels=torch.cat([all_binary_labels,binary_labels])

    all_predict, all_labels = all_predict.numpy(), all_labels.numpy()

    best_f1,best_op,best_acc=-1,-1,-1
    best_ood_acc,best_ood_f1=-1,-1

    # for index, item in enumerate(all_binary_predict.clone()):
    #     if item==torch.tensor(1):
    #         all_predict[index]=len(labels_dict.keys())
    acc,  f1 = evaluate_base(all_predict, all_labels)
    ood_labels_binary=(all_labels==len(labels_dict.keys()))
    ood_labels=all_labels[ood_labels_binary]
    ood_predict=all_predict[ood_labels_binary]
    ood_acc,ood_f1=evaluate_base(ood_predict,ood_labels,mode="weighted")

    id_labels = all_labels[~ood_labels_binary]
    id_predict = all_predict[~ood_labels_binary]
    id_acc, id_f1 = evaluate_base(id_predict, id_labels)
    all_binary_labels, all_binary_predict = all_binary_labels.numpy(), all_binary_predict.numpy()
    binary_acc,binary_f1=evaluate_base(all_binary_predict,all_binary_labels)
    if f1>best_f1:
        best_f1=f1
        best_acc=acc
        best_ood_acc=ood_acc
        best_ood_f1=ood_f1

    # return acc,f1,epoch_val_loss
    print("eval")
    print(f"{best_acc},{best_f1}")
    print(f"ood:{best_ood_acc},{best_ood_f1}")
    print(f"id:{id_acc},{id_f1}")
    print(f"binary:{binary_acc},{binary_f1}")
    return best_acc,best_f1,best_ood_acc,best_ood_f1

def main_test(config):
    # labels_dict = get_label_dict(config["data_path"], config["know_rate"])
    labels_dict_path = config["save_dir"] + os.sep + 'dict.txt'
    with open(labels_dict_path,"r") as f:
        labels_dict=json.load(f)
    print(labels_dict)
    tokenizer = BertTokenizer.from_pretrained(config["pretrained_path"])
    model = torch.load(config["model_save_path"])
    config["num_labels"] = len(labels_dict.keys())+1
    dataset = load_dataset(config["train_script_path"], labels_dict=labels_dict,cache_dir=config["cache_dir"])
    train_dataset, val_dataset, test_dataset = dataset["train"], dataset["validation"], dataset["test"]
    device="cuda" if torch.cuda.is_available()else "cpu"
    datacollator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="max_length",
        max_length=config["token_length"],
        pad_to_multiple_of=8,
        return_tensors='pt'
    )
    test_dataset = test_dataset.map(
        lambda batch: tokenizer_process(batch, tokenizer, config["token_length"]),
        batched=True,
        num_proc=1,
        load_from_cache_file=False
    )
    test_dataset = test_dataset.remove_columns(['text', 'label', 'binary_label'])

    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=datacollator)
    # neg_dataset = load_dataset(config["neg_script_path"], labels_dict=labels_dict, cache_dir="./cache")
    # neg_train_dataset, neg_val_dataset = neg_dataset["train"], neg_dataset["validation"]
    # neg_val_dataset = neg_val_dataset.map(
    #     lambda batch: tokenizer_process(batch, tokenizer, config["token_length"]),
    #     batched=True,
    #     num_proc=1,
    #     load_from_cache_file=False  # not data_training_args.overwrite_cache,
    # )
    # neg_val_dataset = neg_val_dataset.remove_columns(['text', 'label', 'binary_label'])
    # neg_val_loader = DataLoader(neg_val_dataset, batch_size=config["batch_size"], collate_fn=datacollator,
    #                             shuffle=True)
    # eval(config, model, test_loader, neg_val_loader,device , labels_dict)
    test(config, model, test_loader, device, labels_dict)

if __name__=="__main__":
    #只需要修改save_dir 和 tb_folder
    save_dir="./best_model_idea_binary"
    config={
        "pretrained_path":"./bert-base-uncased",
        "data_path":"./oos",
        "train_script_path":"./dataset/oos_data",
        "neg_script_path":"./dataset/neg_data",
        "epoch":400,
        "accumulation_steps":1,
        "num_labels":2,
        "save_dir":save_dir,
        "model_save_path":save_dir+os.sep+"best.pt",
        "is_save":True,
        "config_save_path":save_dir+os.sep+"ood_config.txt",
        "val_result_path":save_dir+os.sep+"result_val.txt",
        "test_result_path": save_dir + os.sep + "result_test.txt",
        "lr":1e-5,
        "weight_decay":1e-5,
        "linear_lr":1e-4,
        "linear_decay":1e-4,
        "use_balance":False,
        "tmp":0.1,
        "val_tmp":1,
        "use_neg":True,
        "token_length":64,
        "alpha":1.0,
        "beta":0.0,
        "neg_multiple":6,
        "cache_dir":"cache_v1",
        "tb_folder": "./tb_folder_idea_binary",
        "know_rate": 0.25,
        "batch_size": 96,
    }
    # train(config)
    # eval(config)
    main_test(config)