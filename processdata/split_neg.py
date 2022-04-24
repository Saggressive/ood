import csv
if __name__=="__main__":
    filepath="../oos/squad.tsv"
    neg_train_path="../oos/neg_train.tsv"
    neg_val_path="../oos/neg_val.tsv"
    with open(filepath, "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter="\t", quotechar=None)
        lines = []
        for line in reader:
            lines.append(line)
    lines=lines[1:]
    with open(neg_train_path, "w", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t", quotechar=None)
        writer.writerow(['text','label'])
        for line in lines[0:int(len(lines)/2)]:
            if len(line)!=0 and line[0].strip()!='' and line[1].strip()!='':
                writer.writerow(line)
    with open(neg_val_path, "w", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t", quotechar=None)
        writer.writerow(['text','label'])
        for line in lines[int(len(lines)/2):]:
            if len(line)!=0 and line[0].strip()!='' and line[1].strip()!='':
                writer.writerow(line)
