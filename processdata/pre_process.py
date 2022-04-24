# import sys
# import os
# os.chdir("../")
# sys.path.insert(0, './')
import numpy
import re
import os
import csv

def normalize(query: str) -> str:#sql处理
    def comma_fix(s):
        # Remove spaces in front of commas
        return s.replace(" , ", ", ")

    def white_space_fix(s):
        # Remove double and triple spaces
        return " ".join(s.split())

    def lower(s):
        # Convert everything except text between (single or double) quotation marks to lower case
        return re.sub(r"\b(?<!['\"])(\w+)(?!['\"])\b", lambda match: match.group(1).lower(), s)

    return comma_fix(white_space_fix(lower(query)))


def get_label_dict(data_path:str , know_label_rate : float) ->dict:
    labels_dict={}
    train_path=data_path + os.sep + "train.tsv"
    with open(train_path, "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter="\t", quotechar=None)
        lines = []
        for line in reader:
            lines.append(line)
    labels=[i[1].strip() for i in lines]
    labels=list(set(labels))
    know_label_num=len(labels)*know_label_rate
    know_label=labels[0:int(know_label_num)]

    for index,i in enumerate(know_label):
        labels_dict[i]=index
    return labels_dict


def process(example,labels_dict ,mode):
    text = example["text"]
    know_label=list(labels_dict.keys())

    if mode=="train":
        if example["label"] in know_label:
            text_label = example["label"]
        else:
            return
    elif mode=="val":
        if example["label"] in know_label:
            text_label = example["label"]
        else:
            text_label = "oos"
    elif mode=="test":
        if example["label"] in know_label:
            text_label = example["label"]
        else:
            text_label = "oos"
    else:
        raise ValueError("mode error")

    labels_dict["oos"]=len(know_label)
    return text,labels_dict[text_label]

def tokenizer_process(batch,tokenizer,max_source_length):#调用时要使用匿名函数
    inputs=batch["text"]
    model_inputs: dict = tokenizer(
        inputs,
        max_length=max_source_length,
        padding=False,
        truncation=True,
        return_overflowing_tokens=False,
    )
    model_inputs["labels"] = numpy.array((batch["label"]))
    model_inputs["binary_labels"]=numpy.array(batch["binary_label"])
    # model_inputs["labels"] = 0
    return model_inputs

