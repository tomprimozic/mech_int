# Import stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
import tqdm.notebook as tqdm

import random
import time

# from google.colab import drive
from pathlib import Path
import pickle
import os

import matplotlib.pyplot as plt
# %matplotlib inline
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "colab"
import plotly.graph_objects as go

from torch.utils.data import DataLoader

from functools import *
import pandas as pd
import gc

# import comet_ml
import itertools



train_model = True #@param


num_layers = 1
d_vocab = 12
d_vocab_out = 10
d_model = 512 #@param
num_heads = 4
d_head = d_model//num_heads
d_mlp = 4 * d_model
seed = 129000 #@param
#@markdown Data
num_digits =  5#@param
n_ctx = 3*num_digits + 3
act_type = 'ReLU'
batch_size = 64 #@param
is_finite = False #@param
num_data = 750 #@param
#@markdown Optimizer
lr = 1e-4 #@param
weight_decay = 0.1 #@param
num_epochs = 3000 #@param

#@markdown Misc
checkpoint_models = False #@param
checkpoint_every = 50 #@param


PLUS_INDEX = 10
EQUALS_INDEX = 11


from .neel_transformer import Transformer



# This is mostly a bunch of over-engineered mess to hack Plotly into producing
# the pretty pictures I want, I recommend not reading too closely unless you
# want Plotly hacking practice
def to_numpy(tensor, flat=False):
    if type(tensor)!=torch.Tensor:
        return tensor
    if flat:
        return tensor.flatten().detach().cpu().numpy()
    else:
        return tensor.detach().cpu().numpy()


def line(x, y=None, hover=None, xaxis='', yaxis='', **kwargs):
    if type(y)==torch.Tensor:
        y = to_numpy(y, flat=True)
    if type(x)==torch.Tensor:
        x=to_numpy(x, flat=True)
    fig = px.line(x, y=y, hover_name=hover, **kwargs)
    fig.update_layout(xaxis_title=xaxis, yaxis_title=yaxis)
    fig.show()


def lines(lines_list, x=None, mode='lines', labels=None, xaxis='', yaxis='', title = '', log_y=False, hover=None, **kwargs):
    # Helper function to plot multiple lines
    if type(lines_list)==torch.Tensor:
        lines_list = [lines_list[i] for i in range(lines_list.shape[0])]
    if x is None:
        x=np.arange(len(lines_list[0]))
    fig = go.Figure(layout={'title':title})
    fig.update_xaxes(title=xaxis)
    fig.update_yaxes(title=yaxis)
    for c, line in enumerate(lines_list):
        if type(line)==torch.Tensor:
            line = to_numpy(line)
        if labels is not None:
            label = labels[c]
        else:
            label = c
        fig.add_trace(go.Scatter(x=x, y=line, mode=mode, name=label, hovertext=hover, **kwargs))
    if log_y:
        fig.update_layout(yaxis_type="log")
    fig.show()


def data_generator(batch_size, num_digits, seed):
    torch.manual_seed(seed)
    while True:
        batch = torch.zeros((batch_size, 3*num_digits+3)).to(torch.int64)
        x = torch.randint(0, 10, (batch_size, num_digits))
        y = torch.randint(0, 10, (batch_size, num_digits))
        batch[:, :num_digits] = x
        batch[:, num_digits] = PLUS_INDEX
        batch[:, 1+num_digits:1+num_digits*2] = y
        batch[:, 1+num_digits*2] = EQUALS_INDEX
        carries = [torch.zeros((batch_size,)).to(torch.int64)]
        for i in range(num_digits):
            carry = carries[-1]
            digit_sum = (batch[:, num_digits-1-i]+batch[:, 2*num_digits-i]+carry)
            batch[:, -1-i] = (digit_sum % 10)
            carry = (digit_sum>=10).to(torch.int64)
            carries.append(carry)
        batch[:, -1-num_digits] = carries[-1]
        carries = torch.stack(carries, axis=1)
        yield batch.cuda(), carries.cuda()
if is_finite:
    test_ds = data_generator(batch_size, num_digits, seed)
    train_ds = data_generator(num_data, num_digits, seed)
    train_tokens, train_carries = next(train_ds)
else:
    ds = data_generator(batch_size, num_digits, seed)



torch.manual_seed(seed)
model = Transformer(num_layers=num_layers,
                    d_vocab=d_vocab,
                    d_model=d_model,
                    d_mlp=d_mlp,
                    d_head=d_head,
                    num_heads=num_heads,
                    n_ctx=n_ctx,
                    act_type=act_type,
                    d_vocab_out=d_vocab_out)
model.to('cuda')
optimizer = optim.AdamW(model.parameters(),
                        lr=lr,
                        weight_decay=weight_decay,
                        betas=(0.9, 0.98))
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(step/10, 1))



# def get_pred_lps(logits, mask, already_just_last=False):
#     if not already_just_last:
#         logits = logits[:, -1, :]
#     log_probs = F.log_softmax(logits.to(torch.float64), dim=-1)
#     prediction_log_probs = torch.gather(log_probs, dim=-1, index=mask[:, None])
#     # return -(prediction_log_probs*(weight(mask))).mean()/weight(mask).float().mean()
#     return -(prediction_log_probs)


# import copy



def get_pred_log_probs(logits, tokens):
    trunc_logits = logits[:, -(num_digits+2):-1]
    ans_tokens = tokens[:, -(num_digits+1):]
    log_probs = F.log_softmax(trunc_logits.to(torch.float64), dim=-1)
    pred_log_probs = torch.gather(log_probs, -1, ans_tokens[:, :, None])[..., 0]
    return pred_log_probs

def loss_fn(logits, tokens):
    return -get_pred_log_probs(logits, tokens).mean()





if is_finite:
    train_losses = []
    ptl_train_list = []
    test_losses = []
    ptl_test_list = []
    # per_token_losses_list = []
    # sds=[]
    # epochs = [0]
    # sds.append(model.state_dict())
    for epoch in tqdm.tqdm(range(num_epochs)):
        train_logits = model(train_tokens)
        per_token_losses_train = -get_pred_log_probs(train_logits, train_tokens).mean(0)
        ptl_train_list.append(to_numpy(per_token_losses_train))
        train_loss = per_token_losses_train.mean()
        train_loss.backward()
        train_losses.append(train_loss.item())

        test_tokens, _ = next(test_ds)
        test_logits = model(test_tokens)
        per_token_losses_test = -get_pred_log_probs(test_logits, test_tokens).mean(0)
        ptl_test_list.append(to_numpy(per_token_losses_test))
        test_loss = per_token_losses_test.mean()
        test_losses.append(test_loss.item())

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if epoch % 100 == 0:
            print(epoch, train_loss.item(), test_loss.item())
        if epoch%1000 ==0 and epoch>0:
            lines([train_losses, test_losses], labels=['train', 'test'])
            lines([[ptl_train_list[j][i] for j in range(len(ptl_train_list))] for i in range(1+num_digits)]+[train_losses]+[[ptl_test_list[j][i] for j in range(len(ptl_train_list))] for i in range(1+num_digits)]+[test_losses],
            labels = [f'tok train {i}' for i in range(1+num_digits)]+['train_loss']+[f'tok test {i}' for i in range(1+num_digits)]+['test_loss'],
            title='Per-digit Loss Curves for 5 digit addition (Finite Data)',
            xaxis='Step',
            yaxis='Loss')


if not is_finite and train_model:
    train_losses = []
    per_token_losses_list = []
    sds=[]
    epochs = [0]
    sds.append(model.state_dict())
    for epoch in tqdm.tqdm(range(num_epochs)):
        tokens, carry = next(ds)
        logits = model(tokens)
        per_token_losses = -get_pred_log_probs(logits, tokens).mean(0)
        per_token_losses_list.append(to_numpy(per_token_losses))
        loss = per_token_losses.mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        train_losses.append(loss.item())
        if epoch % 100 == 0:
            print(epoch, loss.item())
        if checkpoint_models:
            if (epoch+1) % (checkpoint_every) == 0:
                sds.append(model.state_dict())
                epochs.append(epoch+1)
        # if (epoch+1) % 2000 == 0:
    line(train_losses)


line(train_losses)
per_token_losses = np.stack(per_token_losses_list, axis=0)
lines([per_token_losses[:, i] for i in range(1+num_digits)]+[train_losses],
      labels = [f'tok {i}' for i in range(1+num_digits)]+['train_loss'],
      title='Per-digit Loss Curves for 5 digit addition (Infinite Data)',
      xaxis='Step',
      yaxis='Loss')

lines([per_token_losses[:, i] for i in range(1+num_digits)]+[train_losses],
      labels = [f'tok {i}' for i in range(1+num_digits)]+['train_loss'], log_y=True)

line(train_losses)
per_token_losses = np.stack(per_token_losses_list, axis=0)
lines([per_token_losses[:, i] for i in range(1+num_digits)]+[train_losses],
      labels = [f'tok {i}' for i in range(1+num_digits)]+['train_loss'])

lines([per_token_losses[:, i] for i in range(1+num_digits)]+[train_losses],
      labels = [f'tok {i}' for i in range(1+num_digits)]+['train_loss'], log_y=True)

# if is_finite:
#     train_ptls = []
#     test_ptls = []
#     train_losses = []
#     test_losses = []
#     # losses_forced = []
#     epochs = []
#     for epoch in tqdm.tqdm(range(num_epochs)):
#         logits = model(train_tokens)
#         train_ptl = get_pred_lps(logits, train_mask)
#         train_ptls.append(to_numpy(train_ptl))
#         train_loss = train_ptl.mean()
#         train_loss.backward()
#         optimizer.step()
#         scheduler.step()
#         optimizer.zero_grad()
#         train_losses.append(train_loss.item())

#         test_tokens, test_mask = next(test_ds)
#         test_logits = model(test_tokens)
#         test_pred_lps = get_pred_lps(test_logits, test_mask)
#         test_ptl = [test_pred_lps[test_mask==i].mean().item() for i in range(rand_size)]
#         test_ptls.append(test_ptl)
#         test_loss = test_pred_lps.mean()
#         test_losses.append(test_loss.item())

#         if epoch % 100 == 0:
#             print(epoch, train_losses[-1], test_losses[-1])
#         if (epoch % 1000 == 0) and epoch>0:
#             lines([train_losses, test_losses], labels=['train', 'test'], log_y=True)
#         if epoch%5 == 0:
#             epochs.append(epoch)
#             sds.append(copy.deepcopy(model.state_dict()))