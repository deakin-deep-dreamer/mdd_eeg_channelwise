
import random
import time
from datetime import datetime

import torch
from torch import nn

import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score, \
    accuracy_score, classification_report, confusion_matrix


def fix_randomness():
    r"Fix randomness."
    RAND_SEED = 2021
    random.seed(RAND_SEED)
    torch.manual_seed(RAND_SEED)
    np.random.seed(RAND_SEED)


def score(
    labels, preds
):
    r"""Calculate scores."""
    _preds = preds
    _labels = labels
    # _preds = _preds.argmax(axis=1)
    score_prec = precision_score(_labels, _preds, average='macro')
    score_recall = recall_score(_labels, _preds, average='macro')
    score_f1 = f1_score(_labels, _preds, average='macro')
    score_acc = accuracy_score(_labels, _preds)
    report_dict = classification_report(_labels, _preds, output_dict=True)
    cm = confusion_matrix(_labels, _preds)
    return score_prec, score_recall, score_f1, score_acc, report_dict, cm


class EarlyStopping():
    '''Early stops the training if validation loss doesn't improve after a
        given patience.'''

    def __init__(
            self, patience=7, verbose=False, delta=0,
            path='checkpoint.pt', log=None, extra_meta=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss
                improved.
            verbose (bool): If True, prints a message for each validation loss
                improvement.
            delta (float): Minimum change in the monitored quantity to qualify
                as an improvement.
            path (str): Path for the checkpoint to be saved to.
            log : log function (TAG, msg).
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_extra = None  # Extra best other scores/info
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.log = log
        self.first_iter = False
        r"extra_meta is {'metric_name':'test_acc', 'max_val':99}"
        self.extra_meta = extra_meta

        self.print_(
            f'[{self.__class__.__name__}] patience:{patience}, delta:{delta}, model-path:{path}')

    def print_(self, msg):
        if self.log is None:
            print(msg)
        else:
            self.log(f"...[EarlyStopping] {msg}")

    def __call__(self, val_loss, model, extra=None):
        r"""extra is {'test_acc':90}. The key was passed at c'tor."""
        score = -val_loss

        if self.best_score is None:
            self.first_iter = True
            self.best_score = score
            self.best_extra = extra
            self.save_checkpoint(val_loss, model)
            self.first_iter = False
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.print_(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_extra = extra
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            self.print_(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

        r"If extra is passed in call, and extra_meta exists, terminate \
        training if condition is met."
        if (not self.first_iter
                and self.extra_meta is not None
                and self.best_extra is not None
                and self.extra_meta.get('metric_name') is not None
                and self.extra_meta.get('max_val') is not None
                and self.best_extra.get(self.extra_meta.get('metric_name')) is not None
                and self.best_extra.get(self.extra_meta.get('metric_name')) >= self.extra_meta.get('max_val')
            ):
            self.print_(
                f"{self.extra_meta.get('metric_name')}:"
                f"{self.best_extra.get(self.extra_meta.get('metric_name'))} "
                f">= {self.extra_meta.get('max_val')}")
            self.early_stop = True


def training(
    net, ds_train, ds_val, ds_test=None, model_file=None, class_w=True,
    device=None, batch_sz=32, early_stop_patience=30, early_stop_delta=0.0001,
    weight_decay=0., init_lr=0.001, min_lr=1e-6, lr_scheduler_patience=15,
    max_epoch=200, dann_loss=True, eta_regulariser=1., log=print,
):    
    data_loader_train = torch.utils.data.DataLoader(
        dataset=ds_train, batch_size=batch_sz, shuffle=True, drop_last=True)
    data_loader_val = torch.utils.data.DataLoader(
        dataset=ds_val, batch_size=batch_sz, shuffle=False, drop_last=True)

    early_stopping = EarlyStopping(
        patience=early_stop_patience, path=model_file, delta=early_stop_delta,
        log=log, verbose=True,
        extra_meta={'metric_name': 'test_acc', 'max_val': 0.999}
    )

    net.to(device)
    optimizer = torch.optim.Adam(
            net.parameters(), lr=init_lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', min_lr=min_lr,
            factor=0.5, patience=lr_scheduler_patience, verbose=True)
    crit_classif = nn.CrossEntropyLoss()

    log(
        f"[model_training:fit] model:{net.__class__.__name__}, "
        f"dann_loss:{dann_loss}, eta_regulariser:{eta_regulariser}, train-db:{len(ds_train)}, "
        f"val-db:{len(ds_val)}, max-epoch:{max_epoch}, device:{device}, model_file:{model_file}, "
        f"early_stop_pt/delta:{early_stop_patience}/{early_stop_delta}, "
        f"lr_schd_pt:{lr_scheduler_patience}, batch-sz:{batch_sz}, "
        f"init_lr:{init_lr}, min_lr:{min_lr}, "
        f"criterion:{crit_classif}, optimizer:{optimizer}, lr_scheduler:{lr_scheduler},")

    for epoch in range(max_epoch):
        since = time.time()
        train_loss = 0.
        net.train()
        data_loader_train.dataset.on_epoch_end()
        for i_batch, (inputs, labels) in enumerate(data_loader_train):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # with torch.autograd.detect_anomaly():
            output = net(inputs)
            # train encoder and classifier without ref samples
            # 
            optimizer.zero_grad()
            loss = crit_classif(output, labels)            
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().cpu().numpy()

        train_loss = train_loss / len(data_loader_train)
        time_elapsed = time.time() - since

        # validate model
        val_loss = 0.
        net.eval()
        with torch.no_grad():
            for inputs, labels in data_loader_val:
                inputs = inputs.to(device)
                labels = labels.to(device)
                output = net(inputs)
                loss = crit_classif(output, labels)
                val_loss += loss.detach().item()
            val_loss = val_loss / len(data_loader_val)
        
        lr_scheduler.step(val_loss)
        log(
            f"Epoch:{epoch}, train_loss:{train_loss:.3f}, val_loss:{val_loss:.3f}, "
            f"lr:{optimizer.param_groups[0]['lr']}, "
            f"time:{time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

        early_stopping(val_loss, net)
        if early_stopping.early_stop:
            break
    pass
    net.load_state_dict(torch.load(model_file))  # return best model
    log('Training is done.')


def predict(net, dataset, device="cpu"):
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=1, shuffle=False, drop_last=True)
    out_preds, out_labels = [], []
    net.to(device)
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            output = net(inputs)
            out_preds.extend(output.detach().cpu().numpy().argmax(axis=1))
            out_labels.extend(labels.detach().cpu().numpy())
    return out_preds, out_labels