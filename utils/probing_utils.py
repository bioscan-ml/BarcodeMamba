import logging
import os
import pickle
import resource
import time

import numpy as np
import torch
from tqdm import tqdm

from utils.barcode_mamba import BarcodeMamba


def get_eval_part_dict(clf, X_part, y_part):
    import sklearn

    y_pred = clf.predict(X_part)
    res_part = {}
    res_part["count"] = len(y_part)
    # Note that these evaluation metrics have all been converted to percentages
    res_part["accuracy"] = 100.0 * sklearn.metrics.accuracy_score(y_part, y_pred)
    res_part["accuracy-balanced"] = 100.0 * sklearn.metrics.balanced_accuracy_score(
        y_part, y_pred
    )
    res_part["f1-micro"] = 100.0 * sklearn.metrics.f1_score(
        y_part, y_pred, average="micro"
    )
    res_part["f1-macro"] = 100.0 * sklearn.metrics.f1_score(
        y_part, y_pred, average="macro"
    )
    res_part["f1-support"] = 100.0 * sklearn.metrics.f1_score(
        y_part, y_pred, average="weighted"
    )

    return res_part


def get_resource_info(start_time):
    running_info = resource.getrusage(resource.RUSAGE_SELF)
    dt = time.time() - start_time
    hour = dt // 3600
    minutes = (dt - (3600 * hour)) // 60
    seconds = dt - (hour * 3600) - (minutes * 60)
    memory = running_info.ru_maxrss / 1e6
    logging.info(
        f"This part took: {int(hour)}:{int(minutes):02d}:{seconds:02.0f} (hh:mm:ss)\n"
    )
    logging.info(f"Max memory usage: {memory} (GB)")


def representations_from_df(
    df, target_level, model: BarcodeMamba, tokenizer, tokenizer_name
):
    orders = df["order_name"].to_numpy()
    _label_set, y = np.unique(df[target_level], return_inverse=True)
    dna_embeddings = []
    with torch.no_grad():
        for barcode in tqdm(df["nucleotides"]):
            assert tokenizer_name in ["char", "k_mer"]
            if tokenizer_name == "char":
                tokenizer.pad_token = "N"
                x = tokenizer(
                    barcode,
                    add_special_tokens=False,
                    padding="max_length",
                    max_length=660,
                    truncation=True,
                )["input_ids"]
            else:
                x, att_mask = tokenizer(barcode)
            x = torch.tensor(x, dtype=torch.int64)
            x = x.unsqueeze(0).cuda()
            x = model.get_hidden_states(x)
            x = x.mean(1)
            dna_embeddings.append(x.cpu().numpy())
    logging.info(f"There are {len(df)} points in the dataset")
    latent = np.array(dna_embeddings)
    latent = np.squeeze(latent, 1)
    return latent, y, orders


def get_data_and_target(
    config, t_start, tokenizer, model, target_level, pkl_file, csv_file
):
    if os.path.isfile(pkl_file):
        logging.info(f"representation found after {time.time()-t_start} seconds")
        with open(pkl_file, "rb") as f:
            X, y = pickle.load(f)
            targets = csv_file[target_level].to_list()
            label_set = sorted(set(targets))
            y = [label_set.index(t) for t in targets]
    else:
        X, y, _ = representations_from_df(
            csv_file, target_level, model, tokenizer, config.tokenizer.name
        )
        pickle.dump((X, y), open(pkl_file, "wb"))
        logging.info(
            f"{pkl_file} has been generated after {time.time()-t_start} seconds"
        )

    return X, y


def get_pretrained_barcodemamba(dir_path, ckpt_path=None):
    import os
    from omegaconf import OmegaConf as o
    import torch
    from utils.barcode_mamba import BarcodeMamba

    try:
        o.register_new_resolver("eval", eval)
        o.register_new_resolver("div_up", lambda x, y: (x + y - 1) // y)
    except Exception as e:
        print("registers have been registered")

    logging.info(f'loading pretrained barcodemamba from {dir_path}...')
    if ckpt_path == None:
        ckpt_path = os.path.join(dir_path, "checkpoints", "last.ckpt")
    else:
        ckpt_path = os.path.join(dir_path, "checkpoints", ckpt_path)
    config_yaml = os.path.join(dir_path, ".hydra", "config.yaml")
    try:
        config = o.load(config_yaml)
        model = BarcodeMamba(**config.model, use_head=config.dataset.phase)
        model_dict = torch.load(ckpt_path)["state_dict"]
        model_dict = {k.replace("model.", ""): v for k, v in model_dict.items()}
        model.load_state_dict(model_dict, strict=False)
    except Exception as e:
        print(e)
        config = None
        model = None
    return ckpt_path, config, model
