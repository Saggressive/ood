from datasets import load_dataset
from processdata.pre_process import get_label_dict,tokenizer_process
from functools import partial
import csv
def process(example, labels_dict, mode):
    text = example["text"]
    know_label = list(labels_dict.keys())

    if mode == "train":
        if example["text_label"] in know_label:
            text_label = example["text_label"]
            return {"input_str": text, "label": labels_dict[text_label]}
        else:
            return {"input_str":None, "label":None}
    elif mode == "val":
        if example["text_label"] in know_label:
            text_label = example["text_label"]
            return {"input_str": text, "label": labels_dict[text_label]}
        else:
            # text_label = "oos"
            return {"input_str": text, "label": len(know_label)}
    elif mode == "test":
        if example["text_label"] in know_label:
            text_label = example["text_label"]
            return {"input_str": text, "label": labels_dict[text_label]}
        else:
            # text_label = "oos"
            return {"input_str": text, "label": len(know_label)}
    else:
        raise ValueError("mode error")
if __name__=="__main__":
    labels_dict = get_label_dict("../oos", 0.25)
    print(labels_dict)
    dataset = load_dataset("../dataset/oos_data", labels_dict=labels_dict, cache_dir="../cache")
    train_dataset, val_dataset, test_dataset = dataset["train"], dataset["validation"], dataset["test"]
    train_dataset = train_dataset["train"].map(
        lambda batch: tokenizer_process(batch, tokenizer, 512),
        batched=True,
        num_proc=1,
        load_from_cache_file=False  # not data_training_args.overwrite_cache,
    )
    for i in train_dataset:
        print(i)

