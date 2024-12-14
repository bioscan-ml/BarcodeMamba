import argparse
import logging
import os
import pickle
import time
import numpy as np
import torch
from tqdm import tqdm
from ssm_baselines import get_ssm_hf_model_tokenizer
from utils.ssm_dataset import get_probe_dataframe
from torch.utils.data import TensorDataset, DataLoader
from omegaconf import OmegaConf as o
from transformers.modeling_outputs import BaseModelOutputWithNoAttention


logger = logging.getLogger(__name__)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
try:
    o.register_new_resolver("eval", eval)
    o.register_new_resolver("div_up", lambda x, y: (x + y - 1) // y)
except Exception as e:
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


def linear_probe(
    model_name,
    checkpoint,
    input_path='./data',
    target_level="species_name",
    learning_rate=0.01,
    momentum=0.9,
    weight_decay=1e-5,
):
    assert target_level in ["species_name", "genus_name"]
    start = time.time()
    model, tokenizer = get_ssm_hf_model_tokenizer(
        model_name=model_name, checkpoint=checkpoint
    )
    representation_folder = "representation_linear"
    logging.info("tokenizer loaded")

    logging.info(
        f"pretrain model has been successfully loaded after {time.time()-start} seconds"
    )

    model.cuda()
    model.eval()

    os.makedirs(representation_folder, exist_ok=True)
    train_file = os.path.join(representation_folder, f"train_{target_level}.pkl")
    test_file = os.path.join(representation_folder, f"test_{target_level}.pkl")
    train = get_probe_dataframe(input_path, phase="linear", split="train")
    test = get_probe_dataframe(input_path, phase="linear", split="test")

    X, y = get_data_and_target(tokenizer, model, target_level, train_file, train)
    X_test, y_test = get_data_and_target(
        tokenizer, model, target_level, test_file, test
    )

    mean = X.mean()
    std = X.std()
    X = (X - mean) / std
    X_test = (X_test - mean) / std

    X_train = torch.tensor(X).float()
    X_test = torch.tensor(X_test).float()
    y_train = torch.tensor(y)
    y_test = torch.tensor(y_test)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=1024, shuffle=True
    )
    test = torch.utils.data.TensorDataset(X_test, y_test)
    # test_loader = DataLoader(test, batch_size=1024, shuffle=False, drop_last=False)

    d_model = 256 if model_name == "hyenadna" or "ph" in checkpoint else 512
    clf = torch.nn.Sequential(torch.nn.Linear(d_model, np.unique(y).shape[0]))
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
        logging.info(f"Test Accuracy: {accuracy.item():.4f}")


def main(args):
    checkpoint = args["checkpoint"]
    working_folder = f"./baseline_probing_outputs_linear/run_{checkpoint}"
    os.makedirs(working_folder, exist_ok=True)
    os.chdir(working_folder)
    logging.basicConfig(filename="linear-probing.log", level=logging.INFO)
    linear_probe(
        model_name=args["model_name"],
        checkpoint=checkpoint,
        input_path=args['input_path'],
        learning_rate=1,
        momentum=0.95,
        weight_decay=1e-10,
    )


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
    args = vars(parser.parse_args())
    main(args)
