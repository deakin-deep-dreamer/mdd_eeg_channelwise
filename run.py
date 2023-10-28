
import os
from datetime import datetime
import logging
import argparse
import traceback

import torch

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, KFold

import datasource, models, training


logger = logging.getLogger(__name__)


def log(msg):
    logger.debug(msg)


def config_logger(log_file):
    r"""Config logger."""
    global logger
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(format)
    logger.addHandler(ch)
    # logging.basicConfig(format=format, level=logging.DEBUG, datefmt="%H:%M:%S")
    # logger = logging.getLogger(__name__)

    # create file handler which logs even debug messages
    fh = logging.FileHandler(log_file)
    fh.setFormatter(format)
    fh.setLevel(logging.DEBUG)
    # add the handlers to the logger
    logger.addHandler(fh)



def train_loso(
    data_path=None, base_path=None, log_path=None, model_path=None, n_classes=None,
    tm_sim_start=None, device=None, max_epoch=None, early_stop_delta=None, 
    early_stop_patience=None, lr_scheduler_patience=None, init_lr=None, w_decay=None, 
    batch_sz=None, seg_sec=None, n_skip_seg=None, hz=None, class_w=None, dann_loss=True, eta_regulariser=1.
):
    if n_classes != 2:
        raise Exception("n_classes greater than 2 not implemented yet!")
    class_map = {
        "W": 0,
        "1": 1,
        "2": 1,
        "3": 1,
        "4": 1,
        "R": 1,
    }
    # ecg_sig_dim = hz * seg_sec
    rr_seg_dim = 100
    input_dim = rr_seg_dim
    feat_dim = 200
    dataset = datasource.EEGDataset(
        data_directory=data_path, ch_names=["FP1"], hz=hz, log=log,
        # filter_records=['H S1 EC', 'H S2 EC', 'MDD S1 EC', 'MDD S2 EC']
    )
    for i_test_rec_name, test_rec_name in enumerate(dataset.record_names):
        test_idx = []
        test_idx.extend(dataset.record_wise_segments[test_rec_name])
        test_dataset = datasource.PartialDataset(
            dataset, seg_index=test_idx, test=True, log=log)

        preds = []
        labels = []
        fold_scores = {"acc": [], "prec": [], "recl": [], "f1": []}

        r"Fold strategy: balanced labels in train/validation."
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2021)

        r"Training sample database."
        train_idx = []
        for train_rec_name in dataset.record_names:
            if train_rec_name in [test_rec_name]:
                continue
            train_idx.extend(dataset.record_wise_segments[train_rec_name])

        for i_fold in range(1):
            r"Next split test/validation."
            train_index, val_index = next(
                skf.split(
                    np.zeros((len(train_idx), 1)),  # dummy signal
                    [dataset.seg_labels[i] for i in train_idx]))
            train_index = [train_idx[i] for i in train_index]
            val_index = [train_idx[i] for i in val_index]
            
            train_dataset = datasource.PartialDataset(
                dataset, seg_index=train_index, shuffle=True, log=log)
            val_dataset = datasource.PartialDataset(
                dataset, seg_index=val_index, log=log)


            net = models.DANN_CNN(5, n_classes, log=log)
            if i_test_rec_name == 0:
                log(net)

            r"Training."
            model_file = (
                f"{model_path}/{tm_sim_start}_test{test_rec_name}_fold{i_fold}.pt"
            )

            training.training(
                net, train_dataset, val_dataset, model_file=model_file, device=device, 
                batch_sz=batch_sz, early_stop_patience=early_stop_patience, 
                early_stop_delta=early_stop_delta, lr_scheduler_patience=lr_scheduler_patience,
                max_epoch=max_epoch, dann_loss=dann_loss, eta_regulariser=eta_regulariser,
                log=log)
            
            t_preds, t_labels = training.predict(net, test_dataset, device=device)
            r"Persist test preds and labels"
            pred_path = f"{log_path}/sleep_preds/{tm_sim_start}"
            if not os.path.exists(pred_path):
                os.makedirs(pred_path)
            pred_file = f"{pred_path}/test_{test_rec_name}_fold{i_fold}_.preds.csv"
            df = pd.DataFrame(
                {"preds": t_preds, "labels": t_labels})
            df.to_csv(pred_file, index=True)
            _prec, _recl, _f1, _acc, _report_dict, _cm = training.score(
                t_labels, t_preds)
            log(
                f"[{test_rec_name}] Prec:{_prec:.02f}, Recal:{_recl:.02f}, "
                f"F1:{_f1:.02f}, acc:{_acc:.02f}, cm:\n{_cm}")



def load_config():
    parser = argparse.ArgumentParser(description="SleepECGNet")
    parser.add_argument("--i_cuda", default=0, help="CUDA")
    parser.add_argument("--class_w", default=False, help="Weighted class")
    parser.add_argument("--n_classes", default=2,
                        choices=['2', '3'], help="No. of sleep stages")
    parser.add_argument("--hz", default=100, help="Hz")
    parser.add_argument("--seg_sec", default=30, help="Segment len in sec")
    parser.add_argument("--max_epoch", default=200, help="Max no. of epoch")
    parser.add_argument("--early_stop_patience", default=30,
                        help="Early stop patience")
    parser.add_argument("--early_stop_delta", default=0.0001,
                        help="Early stop delta")
    parser.add_argument("--lr_scheduler_patience",
                        default=15, help="LR scheduler patience")
    parser.add_argument("--init_lr", default=0.001, help="Initial LR")
    parser.add_argument("--w_decay", default=0, help="LR weight decay")
    parser.add_argument("--base_path", default=None, help="Sim base path")
    parser.add_argument("--data_path", default=None, help="Data dir")
    parser.add_argument("--batch_sz", default=32, help="Batch size")
    parser.add_argument("--dann_loss", default=True, help="DANN loss enable?")
    parser.add_argument("--eta_regulariser", default=1, help="eta_regulariser dann loss")
    

    args = parser.parse_args()

    args.tm_sim_start = f"{datetime.now():%Y%m%d%H%M%S}"
    if args.base_path is None:
        args.base_path = os.getcwd()
    args.log_path = f"{args.base_path}/logs"
    args.model_path = f"{args.base_path}/models"
    if args.data_path is None:
        args.data_path = f"{args.base_path}/data/4244171/"

    # Convert commonly used parameters to integer, if required.
    if isinstance(args.i_cuda, str):
        args.i_cuda = int(args.i_cuda)
    if isinstance(args.n_classes, str):
        args.n_classes = int(args.n_classes)
    if isinstance(args.batch_sz, str):
        args.batch_sz = int(args.batch_sz)
    if isinstance(args.max_epoch, str):
        args.max_epoch = int(args.max_epoch)
    if isinstance(args.dann_loss, str):
        args.dann_loss = (args.dann_loss=='True')
    if isinstance(args.eta_regulariser, str):
        args.eta_regulariser = float(args.eta_regulariser)
    
    # GPU device?
    if args.i_cuda > 0:
        args.device = torch.device(
            f"cuda:{args.i_cuda}" if torch.cuda.is_available() else "cpu")
    else:
        args.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "cpu":
            args.device = torch.device(
                f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
            )
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    return args


def main():
    training.fix_randomness()
    args = load_config()
    config_logger(f"{args.log_path}/{args.tm_sim_start}.log")
    param_dict = vars(args)
    log(param_dict)
    # Exclude non-existing arguments
    #
    param_dict.pop("i_cuda", None)
    # param_dict.pop("n_skip_seg", None)
    try:
        train_loso(**param_dict)
    except Exception as e:
        log(f"Exception in kfold, {str(e)}, caused by - \n{traceback.format_exc()}")
        logger.exception(e)


if __name__ == '__main__':
    main()
