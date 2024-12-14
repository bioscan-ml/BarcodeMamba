import argparse
import logging
import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
from ssm_baselines import get_ssm_hf_model_tokenizer
from utils.ssm_dataset import get_probe_dataframe
from sklearn.neighbors import KNeighborsClassifier
from utils.probing_utils import (
    get_eval_part_dict,
)
from omegaconf import OmegaConf as o
from transformers.modeling_outputs import BaseModelOutputWithNoAttention


logger = logging.getLogger(__name__)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
try:
    o.register_new_resolver("eval", eval)
    o.register_new_resolver("div_up", lambda x, y: (x + y - 1) // y)
except Exception:
    pass


def get_data_and_target(tokenizer, model, target_level, pkl_file, csv_file):
    if os.path.isfile(pkl_file):
        with open(pkl_file, "rb") as f:
            X, y = pickle.load(f)
            targets = csv_file[target_level].to_list()
            label_set = sorted(set(targets))
            y = [label_set.index(t) for t in targets]
    else:
        X, y, _ = representations_from_df(csv_file, target_level, model, tokenizer)
        pickle.dump((X, y), open(pkl_file, "wb"))

    return X, y


def representations_from_df(df, target_level, model, tokenizer):
    orders = df["order_name"].to_numpy()
    _label_set, y = np.unique(df[target_level], return_inverse=True)
    dna_embeddings = []
    with torch.no_grad():
        for barcode in tqdm(df["nucleotides"]):
            x = tokenizer(barcode)["input_ids"]
            x = torch.tensor(x, dtype=torch.int64)
            x = x.unsqueeze(0).cuda()
            x = model(x)
            if isinstance(x, BaseModelOutputWithNoAttention):
                x = x["last_hidden_state"]
            x = x.mean(1)
            dna_embeddings.append(x.cpu().numpy())
    logging.info(f"There are {len(df)} points in the dataset")
    latent = np.array(dna_embeddings)
    latent = np.squeeze(latent, 1)
    return latent, y, orders


def knn_probe(args, target_level="genus_name"):
    model_name = args["model_name"]
    checkpoint = args["checkpoint"]
    assert target_level in ["species_name", "genus_name"]
    model, tokenizer = get_ssm_hf_model_tokenizer(
        model_name=model_name, checkpoint=checkpoint
    )
    representation_folder = "representation_knn"
    logging.info("pretrain model has been successfully loaded")

    model.cuda()
    model.eval()

    os.makedirs(representation_folder, exist_ok=True)
    train_file = os.path.join(representation_folder, f"train_{target_level}.pkl")
    test_file = os.path.join(representation_folder, f"test_{target_level}.pkl")
    input_path = args['input_path']
    train = get_probe_dataframe(input_path, phase="knn", split="train")
    test = get_probe_dataframe(input_path, phase="knn", split="test")

    X, y = get_data_and_target(tokenizer, model, target_level, train_file, train)
    X_unseen, y_unseen = get_data_and_target(
        tokenizer, model, target_level, test_file, test
    )

    c = 0
    for label in y_unseen:
        if label not in y:
            c += 1
    logging.info(f"There are {c} genus that are not present during training")

    # kNN =====================================================================
    logging.info("Computing Nearest Neighbors")
    # Fit ---------------------------------------------------------------------
    clf = KNeighborsClassifier(n_neighbors=args["n_neighbors"], metric=args["metric"])
    clf.fit(X, y)

    # Evaluate ----------------------------------------------------------------
    # Create results dictionary
    results = {}
    for partition_name, X_part, y_part in [
        ("Train", X, y),
        ("Unseen", X_unseen, y_unseen),
    ]:
        res_part = get_eval_part_dict(clf, X_part, y_part)
        results[partition_name] = res_part
        logging.info(f"\n{partition_name} evaluation results:")
        for k, v in res_part.items():
            if k == "count":
                logging.info(f"  {k + ' ':.<21s}{v:7d}")
            else:
                logging.info(f"  {k + ' ':.<24s} {v:6.2f} %")
    acc = results["Unseen"]["accuracy"]
    logging.info(f"accuracy: {acc}")


def main(args):
    checkpoint = args["checkpoint"]
    working_folder = f"./baseline_probing_outputs_knn/run_{checkpoint}"
    os.makedirs(working_folder, exist_ok=True)
    os.chdir(working_folder)
    logging.basicConfig(filename="knn-probing.log", level=logging.INFO)
    knn_probe(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
    )
    parser.add_argument(
        "--input_path",
        type=str,
    )
    parser.add_argument(
        "-n",
        "--n-neighbors",
        type=int,
        help="The path to checkpoint and config",
        default=1,
    )
    parser.add_argument(
        "-m",
        "--metric",
        default="cosine",
        type=str,
        help="Distance metric to use for kNN. Default: %(default)s",
    )
    args = vars(parser.parse_args())
    main(args)
