from torch.utils.data import DataLoader,Dataset
import csv
import numpy as np
from processdata.pre_process import get_label_dict
from transformers import BertTokenizer
import torch
import random
class synthesis_data(Dataset):
    def __init__(self,train_data_path,labels_dict):
        super(synthesis_data).__init__()
        self.lines=[]
        self.labels_dict=labels_dict
        with open(train_data_path, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            for idx, line in enumerate(reader):
                if line[1] in self.labels_dict.keys():
                    self.lines.append(line)

    def __len__(self):
        return len(self.lines)*8

    def __getitem__(self, item):
        while(1):
            cdt = np.random.choice(len(self.lines), 2, replace=False)
            if self.lines[cdt[0]][1]==self.lines[cdt[1]][1]:
                continue
            else:
                choice = np.random.random()
                if choice>1/2:
                    text0_list=self.lines[cdt[0]][0].split(" ")
                    text1_list=self.lines[cdt[1]][0].split(" ")
                    min_len=min(len(text0_list),len(text1_list))

                    random_array = np.random.randint(0, 2, min_len)
                    syn_list=[]
                    for i in range(min_len):
                        if random_array[i]==0:
                            syn_list.append(text0_list[i])
                        else:
                            syn_list.append(text1_list[i])
                    return " ".join(syn_list)
                else:
                    text0_list = self.lines[cdt[0]][0].split(" ")
                    text1_list = self.lines[cdt[1]][0].split(" ")
                    text0_len,text1_len=len(text0_list),len(text1_list)
                    syn_list = []
                    cdt = [0,1]
                    random.shuffle(cdt)
                    text0_split=text0_list[0+cdt[0]*int(text0_len/2):int(text0_len/2)+cdt[0]*int(text0_len/2)]
                    text1_split=text1_list[0+cdt[1]*int(text1_len/2):int(text1_len/2)+cdt[1]*int(text1_len/2)]
                    syn_list.extend(text0_split[0:int(len(text0_split)/2)])
                    syn_list.extend(text1_split[0:int(len(text1_split)/2)])
                    syn_list.extend(text0_split[int(len(text0_split)/2):])
                    syn_list.extend(text1_split[int(len(text1_split)/2):])
                    return " ".join(syn_list)
class DataCollator():
    def __init__(self,tokenizer,labels_dict,config):
        self.tokenizer=tokenizer
        self.labels_dict=labels_dict
        self.config=config

    def __call__(self,item):
        model_inputs: dict = self.tokenizer(
            item,
            max_length=self.config["token_length"],
            padding=False,
            truncation=True,
            return_overflowing_tokens=False,
        )
        model_inputs["labels"] = torch.tensor([len(self.labels_dict)]*self.config["batch_size"])
        model_inputs["binary_labels"] = torch.tensor([1]*self.config["batch_size"])
        batch = self.tokenizer.pad(
            model_inputs,
            padding="max_length",
            max_length=self.config["token_length"],
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch

if __name__=="__main__":
    config={}
    config["token_length"]=64
    config["batch_size"]=32
    tokenizer = BertTokenizer.from_pretrained("../bert-base-uncased")
    labels_dict = get_label_dict("../oos", 0.25)
    dataset=synthesis_data("../oos/train.tsv",labels_dict)
    collator=DataCollator(tokenizer,labels_dict,config)
    dataloader=DataLoader(dataset,batch_size=32,shuffle=True,collate_fn=collator,drop_last=True)
    for index,i in enumerate(dataloader):
        print(tokenizer.batch_decode(i["input_ids"],skip_special_tokens=True))


