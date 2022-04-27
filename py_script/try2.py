save_dir = "./best_model_idea_binary"
    config = {
        "pretrained_path": "./bert-base-uncased",
        "data_path": "./oos",
        "know_rate": 0.25,
        "train_script_path": "./dataset/oos_data",
        "neg_script_path": "./dataset/neg_data",
        "epoch": 100,
        "accumulation_steps": 1,
        "batch_size": 1,
        "num_labels": 2,
        "save_dir": save_dir,
        "is_save": True,
        "config_save_path": save_dir + os.sep + "ood_config.txt",
        "val_result_path": save_dir + os.sep + "result_val.txt",
        "test_result_path": save_dir + os.sep + "result_test.txt",
        "lr": 1e-5,
        "weight_decay": 1e-4,
        "linear_lr": 1e-4,
        "linear_decay": 1e-4,
        "use_balance": False,
        "tb_folder": "./tb_folder_idea_binary",
        "tmp": 1,
        "val_tmp": 1,
        "use_neg": True,
        "token_length": 64,
        "alpha": 1.0,
        "beta": 1.0,
        "neg_multiple": 4
    }
    labels_dict = get_label_dict(config["data_path"], config["know_rate"])
    dataset = load_dataset(config["train_script_path"], labels_dict=labels_dict, cache_dir="./cache")
    tokenizer = BertTokenizer.from_pretrained(config["save_dir"])
    datacollator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="max_length",
        max_length=config["token_length"],
        pad_to_multiple_of=8,
        return_tensors='pt'
    )
    train_dataset, val_dataset, test_dataset = dataset["train"], dataset["validation"], dataset["test"]
    test_dataset = test_dataset.map(
        lambda batch: tokenizer_process(batch, tokenizer, config["token_length"]),
        batched=True,
        num_proc=1,
        load_from_cache_file=False  # not data_training_args.overwrite_cache,
    )
    test_dataset = test_dataset.remove_columns(['text', 'label', 'binary_label'])
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], collate_fn=datacollator, shuffle=True)


    for k in range(3):
        for index, i in enumerate(test_loader):
            print(tokenizer.batch_decode(i["input_ids"],skip_special_tokens=True))
        print("*"*20)