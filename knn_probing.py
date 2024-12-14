import argparse
import logging
import os
import time
from sklearn.neighbors import KNeighborsClassifier
from utils.probing_utils import (
    get_data_and_target,
    get_eval_part_dict,
    get_pretrained_barcodemamba,
    get_resource_info,
)
from utils.ssm_dataset import get_probe_dataframe, get_tokenizer
from omegaconf import OmegaConf as o

logger = logging.getLogger(__name__)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
try:
    o.register_new_resolver("eval", eval)
    o.register_new_resolver("div_up", lambda x, y: (x + y - 1) // y)
except Exception:
    print("registers have been registered")


def knn_probe(args, target_level="genus_name"):
    t_start = time.time()
    timing_stats = {}
    assert target_level in ["species_name", "genus_name"]
    dir_path = args["dir_path"]
    ckpt_path = args["ckpt"]
    ckpt_path, config, model = get_pretrained_barcodemamba(dir_path, ckpt_path)
    config.dataset.input_path = args['input_path']
    logging.info(f"Pretrained model is loaded from {ckpt_path}")
    logging.info(f"Config and model are loaded from {dir_path}")
    representation_folder = "representation_knn"
    tokenizer = get_tokenizer(
        tokenizer_name=config.tokenizer.name, tokenizer_config=config.tokenizer
    )
    logging.info(f"tokenizer {config.tokenizer.name} loaded")
    logging.info(
        f"pretrain model has been successfully loaded after {time.time()-t_start} seconds"
    )

    model.cuda()
    model.eval()
    t_start_embed = time.time()

    os.makedirs(representation_folder, exist_ok=True)
    train_file = os.path.join(representation_folder, f"train_{target_level}.pkl")
    test_file = os.path.join(representation_folder, f"test_{target_level}.pkl")

    train = get_probe_dataframe(config.dataset.input_path, phase="knn", split="train")
    test = get_probe_dataframe(config.dataset.input_path, phase="knn", split="test")
    timing_stats["preamble"] = time.time() - t_start

    X, y = get_data_and_target(
        config, t_start, tokenizer, model, target_level, train_file, train
    )
    X_unseen, y_unseen = get_data_and_target(
        config, t_start, tokenizer, model, target_level, test_file, test
    )
    timing_stats["embed"] = time.time() - t_start_embed

    c = 0
    for label in y_unseen:
        if label not in y:
            c += 1
    logging.info(f"There are {c} genus that are not present during training")

    get_resource_info(t_start_embed)

    # kNN =====================================================================
    logging.info("Computing Nearest Neighbors")
    # Fit ---------------------------------------------------------------------
    t_start_train = time.time()
    clf = KNeighborsClassifier(n_neighbors=args["n_neighbors"], metric=args["metric"])
    clf.fit(X, y)
    timing_stats["train"] = time.time() - t_start_train

    # Evaluate ----------------------------------------------------------------
    t_start_test = time.time()
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
    timing_stats["test"] = time.time() - t_start_test

    # Save results -------------------------------------------------------------
    get_resource_info(t_start)
    timing_stats["overall"] = time.time() - t_start
    logging.info(timing_stats)


def main(args):
    working_folder = f'./probing_outputs/run_knn_{args["dir_path"].split("/")[-1]}'
    os.makedirs(working_folder, exist_ok=True)
    os.chdir(working_folder)
    logging.basicConfig(filename=f"knn-probing-{args['metric']}.log", level=logging.INFO)
    knn_probe(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dir-path",
        type=str,
        help="The path to checkpoint and config",
    )
    parser.add_argument(
        "-n",
        "--n-neighbors",
        type=int,
        help="Number of neighbors",
        default=1,
    )
    parser.add_argument(
        "-m",
        "--metric",
        default="cosine",
        type=str,
        help="Distance metric to use for kNN. Default: %(default)s",
    )
    parser.add_argument(
        "-c",
        "--ckpt",
        default=None,
        type=str,
        help="Which ckpt to use for knn probing",
    )    
    parser.add_argument(
        "--input-path",
        default=None,
        type=str,
        help="Path to data",
    )    
    args = vars(parser.parse_args())
    main(args)
