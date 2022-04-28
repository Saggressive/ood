import torch
import numpy as np
import random
def get_four_fold(pos_batch,config):
    synthesis_batch = {}
    synthesis_input_ids_list, synthesis_mask_list, synthesis_labels_list, synthesis_binary_list = [], [], [], []
    pos_batch_size = len(pos_batch["input_ids"])
    seq_len = len(pos_batch["input_ids"][0])
    while len(synthesis_input_ids_list) < int(config["neg_multiple"]) * len(pos_batch["input_ids"]):
        cdt = np.random.choice(config["batch_size"], 4, replace=False)
        if np.sum((cdt>=pos_batch_size).astype(int))>0:
            continue
        labels_cdt = [pos_batch["labels"][i] for i in cdt]
        labels_cdt = np.array(labels_cdt)
        max_number = np.max(np.bincount(labels_cdt))
        if max_number > int(cdt.shape[0]/2):
            continue
        random_array = torch.tensor(np.random.randint(0, 4, seq_len))
        cdt_input0, cdt_input1 = pos_batch["input_ids"][cdt[0]], pos_batch["input_ids"][cdt[1]]
        cdt_input2, cdt_input3 = pos_batch["input_ids"][cdt[2]], pos_batch["input_ids"][cdt[3]]
        cdt_random_arry0, cdt_random_arry1, cdt_random_arry2, cdt_random_arry3 = \
            (random_array == 0).int(), (random_array == 1).int(), (random_array == 2).int(), (random_array == 3).int()
        synthesis_input = cdt_random_arry0 * cdt_input0 + cdt_random_arry1 * cdt_input1 + cdt_random_arry2 * cdt_input2 + cdt_random_arry3 * cdt_input3
        synthesis_input = synthesis_input[(synthesis_input) != 0]
        synthesis_input = synthesis_input[(synthesis_input) != 102]
        no_padding_len = synthesis_input.size()[0]
        synthesis_input = torch.cat([synthesis_input, torch.tensor([0] * (seq_len - no_padding_len))])
        synthesis_input[no_padding_len - 1] = torch.tensor(102)
        synthesis_input_ids_list.append(synthesis_input)
        synthesis_mask_list.append((~(synthesis_input == 0)).int())
        synthesis_labels_list.append(torch.tensor(config["num_labels"]))
        synthesis_binary_list.append(torch.tensor(1))
    synthesis_input_ids = torch.stack(synthesis_input_ids_list, dim=0)
    synthesis_mask = torch.stack(synthesis_mask_list, dim=0)
    synthesis_labels = torch.tensor(synthesis_labels_list)
    synthesis_binary = torch.tensor(synthesis_binary_list)
    synthesis_batch["input_ids"], synthesis_batch["attention_mask"] = synthesis_input_ids, synthesis_mask
    synthesis_batch["labels"], synthesis_batch["binary_labels"] = synthesis_labels, synthesis_binary
    return synthesis_batch

# def get_two_fold(pos_batch,config):
#     synthesis_batch = {}
#     synthesis_input_ids_list, synthesis_mask_list, synthesis_labels_list, synthesis_binary_list = [], [], [], []
#     pos_batch_size = len(pos_batch["input_ids"])
#     seq_len = len(pos_batch["input_ids"][0])
#     while len(synthesis_input_ids_list) < int(config["neg_multiple"]) * len(pos_batch["input_ids"]):
#         cdt = np.random.choice(config["batch_size"], 2, replace=False)
#         if np.sum((cdt>=pos_batch_size).astype(int))>0:
#             continue
#         labels_cdt = [pos_batch["labels"][i] for i in cdt]
#         labels_cdt = np.array(labels_cdt)
#         max_number = np.max(np.bincount(labels_cdt))
#         if max_number > int(cdt.shape[0]/2):
#             continue
#         random_array = torch.tensor(np.random.randint(0, 2, seq_len))
#         cdt_input0, cdt_input1 = pos_batch["input_ids"][cdt[0]], pos_batch["input_ids"][cdt[1]]
#         cdt_random_arry0, cdt_random_arry1 = (random_array == 0).int(), (random_array == 1).int()
#         synthesis_input = cdt_random_arry0 * cdt_input0 + cdt_random_arry1 * cdt_input1
#         synthesis_input = synthesis_input[(synthesis_input) != 0]
#         synthesis_input = synthesis_input[(synthesis_input) != 102]
#         no_padding_len = synthesis_input.size()[0]
#         synthesis_input = torch.cat([synthesis_input, torch.tensor([0] * (seq_len - no_padding_len))])
#         synthesis_input[no_padding_len - 1] = torch.tensor(102)
#         synthesis_input_ids_list.append(synthesis_input)
#         synthesis_mask_list.append((~(synthesis_input == 0)).int())
#         synthesis_labels_list.append(torch.tensor(config["num_labels"]))
#         synthesis_binary_list.append(torch.tensor(1))
#     synthesis_input_ids = torch.stack(synthesis_input_ids_list, dim=0)
#     synthesis_mask = torch.stack(synthesis_mask_list, dim=0)
#     synthesis_labels = torch.tensor(synthesis_labels_list)
#     synthesis_binary = torch.tensor(synthesis_binary_list)
#     synthesis_batch["input_ids"], synthesis_batch["attention_mask"] = synthesis_input_ids, synthesis_mask
#     synthesis_batch["labels"], synthesis_batch["binary_labels"] = synthesis_labels, synthesis_binary
#     return synthesis_batch

def get_two(pos_batch,config,labels_dict):
    num_labels=len(labels_dict)
    synthesis_batch = {}
    synthesis_input_ids_list, synthesis_mask_list, synthesis_labels_list, synthesis_binary_list = [], [], [], []
    while len(synthesis_input_ids_list) < int(config["neg_multiple"]) * len(pos_batch["input_ids"]):
        cdt = np.random.choice(config["batch_size"], 2, replace=False)
        if len(pos_batch["attention_mask"]) <= cdt[0] or len(pos_batch["attention_mask"]) <= cdt[1]:
            continue
        min_len = min(sum(pos_batch["attention_mask"][cdt[0]]), sum(pos_batch["attention_mask"][cdt[1]]))
        if min_len<6:
            continue
        if pos_batch["labels"][cdt[0]] != pos_batch["labels"][cdt[1]] and config["num_labels"] not in cdt:
            random_array = torch.tensor(np.random.randint(0, 2, min_len.item()))
            cdt_input0, cdt_input1 = pos_batch["input_ids"][cdt[0]], pos_batch["input_ids"][cdt[1]]
            cdt_random_arry0 = torch.cat(
                [random_array, torch.tensor([0] * (len(pos_batch["attention_mask"][cdt[0]]) - min_len))])
            cdt_random_arry1 = torch.cat(
                [1 - random_array, torch.tensor([0] * (len(pos_batch["attention_mask"][cdt[1]]) - min_len))])
            synthesis_input = cdt_random_arry0 * cdt_input0 + cdt_random_arry1 * cdt_input1
            synthesis_input[min_len - 1] = torch.tensor(102)
            if random.random()>1/2:
                b=synthesis_input[1:min_len-1]
                b = b[torch.randperm(b.size(0))]
                synthesis_input[1:min_len-1]=b[:]
            synthesis_input_ids_list.append(synthesis_input)
            synthesis_mask_list.append((~(synthesis_input == 0)).int())
            synthesis_labels_list.append(torch.tensor(num_labels))
            synthesis_binary_list.append(torch.tensor(1))
    synthesis_input_ids = torch.stack(synthesis_input_ids_list, dim=0)
    synthesis_mask = torch.stack(synthesis_mask_list, dim=0)
    synthesis_labels = torch.tensor(synthesis_labels_list)
    synthesis_binary = torch.tensor(synthesis_binary_list)
    synthesis_batch["input_ids"], synthesis_batch["attention_mask"] = synthesis_input_ids, synthesis_mask
    synthesis_batch["labels"], synthesis_batch["binary_labels"] = synthesis_labels, synthesis_binary
    return synthesis_batch