import argparse
import logging
import os
import time
import numpy as np
import torch
from tqdm import tqdm
from utils.probing_utils import get_data_and_target
from utils.ssm_dataset import get_probe_dataframe, get_tokenizer
from torch.utils.data import TensorDataset, DataLoader
from omegaconf import OmegaConf as o
from utils.probing_utils import get_pretrained_barcodemamba

logger = logging.getLogger(__name__)


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
try:
    o.register_new_resolver("eval", eval)
    o.register_new_resolver("div_up", lambda x, y: (x + y - 1) // y)
except Exception:
    print("registers have been registered")


def linear_probe(
    dir_path,
    ckpt_path=None,
    input_path=None,
    target_level="species_name",
    learning_rate=0.01,
    momentum=0.9,
    weight_decay=1e-5,
):
    assert target_level in ["species_name", "genus_name"]
    start = time.time()
    ckpt_path, config, model = get_pretrained_barcodemamba(dir_path, ckpt_path)
    config.dataset.input_path = input_path
    logging.info(f"Pretrained model is loaded from {ckpt_path}")
    logging.info(f"Config and model are loaded from {dir_path}")
    representation_folder = "representation_linear"
    tokenizer = get_tokenizer(
        tokenizer_name=config.tokenizer.name, tokenizer_config=config.tokenizer
    )
    logging.info("tokenizer loaded")

    logging.info(
        f"pretrain model has been successfully loaded after {time.time()-start} seconds"
    )

    model.cuda()
    model.eval()

    os.makedirs(representation_folder, exist_ok=True)
    # target_level = config.dataset.classify_level
    train_file = os.path.join(representation_folder, f"train_{target_level}.pkl")
    test_file = os.path.join(representation_folder, f"test_{target_level}.pkl")

    train = get_probe_dataframe(
        config.dataset.input_path, phase="linear", split="train"
    )
    test = get_probe_dataframe(config.dataset.input_path, phase="linear", split="test")

    X, y = get_data_and_target(
        config, start, tokenizer, model, target_level, train_file, train
    )
    X_test, y_test = get_data_and_target(
        config, start, tokenizer, model, target_level, test_file, test
    )

    mean = X.mean()
    std = X.std()
    X = (X - mean) / std
    X_test = (X_test - mean) / std

    X_train = torch.tensor(X).float()
    X_test = torch.tensor(X_test).float()
    y_train = torch.tensor(y)
    y_test = torch.tensor(y_test)

    logging.info(f"Train shapes: {X_train.shape}, {X_test.shape}")
    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=1024, shuffle=True
    )
    test = torch.utils.data.TensorDataset(X_test, y_test)
    # test_loader = DataLoader(test, batch_size=1024, shuffle=False, drop_last=False)

    # Define the model
    # clf = torch.nn.Sequential(torch.nn.Linear(768, np.unique(y).shape[0]))
    clf = torch.nn.Sequential(
        torch.nn.Linear(config.model.d_model, np.unique(y).shape[0])
    )
    clf.cuda()

    # Train the model
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        clf.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay
    )

    num_epochs = 200
    for epoch in tqdm(range(num_epochs)):

        for X_train, y_train in train_loader:

            X_train = X_train.cuda()
            y_train = y_train.cuda()

            # Forward pass
            y_pred = clf(X_train)
            loss = criterion(y_pred, y_train)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print the loss every 100 epochs
        if (epoch + 1) % 10 == 0:
            logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Evaluate the model
    X_test = X_test.cuda()
    y_test = y_test.cuda()
    with torch.no_grad():
        y_pred = clf(X_test)
        _, predicted = torch.max(y_pred, dim=1)
        accuracy = (predicted == y_test).float().mean()
        logging.info(
            f"Learning rate: {learning_rate}, momentum: {momentum} weight_decay: {weight_decay} Test Accuracy: {accuracy.item():.4f}"
        )

    _time = time.time() - start  # running_info.ru_utime + running_info.ru_stime
    hour = _time // 3600
    minutes = (_time - (3600 * hour)) // 60
    seconds = _time - (hour * 3600) - (minutes * 60)
    logging.info(
        f"The code finished after: {int(hour)}:{int(minutes)}:{round(seconds)} (hh:mm:ss)\n"
    )


def main(dirpath: str, ckpt_path: str, input_path: str):
    working_folder = f'./probing_outputs/run_linear_{dirpath.split("/")[-1]}'
    os.makedirs(working_folder, exist_ok=True)
    os.chdir(working_folder)
    logging.basicConfig(filename="linear-probing.log", level=logging.INFO)
    linear_probe(
        dir_path=dirpath,
        ckpt_path=ckpt_path,
        input_path=input_path,
        learning_rate=1,
        momentum=0.95,
        weight_decay=1e-10,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dir-path",
        type=str,
        help="The path to checkpoint and config",
    )
    parser.add_argument(
        "-c",
        "--ckpt",
        default=None,
        type=str,
        help="Which checkpoint to use for linear probing",
    )
    parser.add_argument(
        "--input-path",
        default=None,
        type=str,
        help="Path to data",
    )
    args = vars(parser.parse_args())
    main(
        dirpath=args["dir_path"], ckpt_path=args["ckpt"], input_path=args["input_path"]
    )
