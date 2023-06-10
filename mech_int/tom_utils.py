import os
import copy
import math
import random
import datetime
from pathlib import Path
from typing import Literal

import einops
import numpy as np
import torch as th
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm
import plotly.express as px
import matplotlib.pyplot as plt
import torch.nn.functional as F
import plotly.graph_objects as go
from plotly.subplots import make_subplots


from .neel_digits import DEVICE, ROOT_DIR, NUM_DIGITS, create_model, data_generator


DATA_DIR = ROOT_DIR / 'infinite'


########################    data loading     ########################


def load_data(path):
  assert os.path.exists(path), f'path does not exist: {path}'
  saved_data = th.load(path, map_location=DEVICE)
  assert saved_data['epoch'] + 1 == len(saved_data['train_loss'])
  attn_shape = saved_data['model']['blocks.0.attn.W_K'].shape
  mlp_shape = saved_data['model']['blocks.0.mlp.W_out'].shape
  assert attn_shape[0] == 4 and attn_shape[1] * 4 == attn_shape[2]
  assert attn_shape[2] == mlp_shape[0]
  d_model = mlp_shape[0]
  d_mlp = mlp_shape[1]
  saved_data['d_mlp'] = d_mlp
  saved_data['d_model'] = d_model
  return saved_data


def load_epoch(d, epoch=None):
  if epoch is None:
    fns = os.listdir(DATA_DIR / d)
    assert all(fn.endswith('.pth') for fn in fns)
    fns.sort(key=lambda fn: int(fn[:-len('.pth')]))
    fn = fns[-1]
  else:
    fn = f'{epoch}.pth'
  return load_data(DATA_DIR / d / fn)



MODELS = []
for d in os.listdir(DATA_DIR):
  saved_data = load_epoch(d)
  MODELS.append({'dir': d,
            'epoch': saved_data['epoch'],
            'd_model': saved_data['d_model'],
            'd_mlp': saved_data['d_mlp'],
            'loss': saved_data['train_loss'][-1],
            'accuracy': saved_data['accuracy'][-1],
          })
MODELS = pd.DataFrame(MODELS)


def load(*, path=None, epoch=None, d_model=None, d_mlp=None):
  if path is not None:
    return load_data(path)
  else:
    assert d_model is not None and d_mlp is not None
    ds = MODELS.loc[(MODELS['d_model'] == d_model) & (MODELS['d_mlp'] == d_mlp)].sort_values('epoch')
    d = ds.iloc[-1]['dir']
    return load_epoch(d, epoch=epoch)



def load_model(*, path=None, epoch=None, d_model=None, d_mlp=None):
  saved_data = load(path=path, epoch=epoch, d_model=d_model, d_mlp=d_mlp)
  d_model = saved_data['d_model']
  d_mlp = saved_data['d_mlp']
  model = create_model(d_model=d_model, d_mlp=d_mlp)
  model.load_state_dict(state_dict=saved_data['model'])
  return model.to(DEVICE)



########################    running the model     ########################



def run_model(model, tokens, result: Literal['logits', 'probs', 'log_probs', 'digits']='digits'):
  logits = model(tokens)
  trunc_logits = logits[:, -(NUM_DIGITS+2):-1]
  trunc_logits = trunc_logits.detach().cpu().to(th.float64)
  if result == 'logits':
    return trunc_logits
  elif result == 'log_probs':
    log_probs = F.log_softmax(trunc_logits, dim=-1)
    return log_probs
  else:
    probs = F.softmax(trunc_logits, dim=-1)
    if result == 'probs':
      return probs
    else:
      return np.argmax(probs.numpy(), axis=-1)



tokens_datasource = data_generator(32)

def get_tokens(n=1):
  result = []
  while n > 0:
    tokens, _ = next(tokens_datasource)
    result.append(tokens)
    n -= len(tokens)
  result = th.concat(result)
  if n < 0:
    result = result[:n]
  return result


TOKEN_TO_LABEL_DICT = {i: str(i) for i in range(10)} | {10: '+', 11: '='}
def tokens_to_labels(t: th.Tensor):
  if t.ndim == 2:
    assert t.shape[0] == 1
    t = t[0]
  assert t.ndim == 1
  return [TOKEN_TO_LABEL_DICT[int(i.item())] for i in t]


def show_tokens(t: th.Tensor):
  ''.join(tokens_to_labels(t)).replace('=', ' = ').replace('+', ' + ')


def get_accuracy(model, tokens):
  answer = run_model(model, tokens)
  correct = (answer == tokens[:, -(NUM_DIGITS+1):].detach().cpu().numpy())
  return correct.mean()