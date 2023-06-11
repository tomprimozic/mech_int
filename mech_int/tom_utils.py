import os
import copy
import math
import random
import datetime
import functools
import itertools
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


from .neel_digits import DEVICE, ROOT_DIR, NUM_DIGITS, create_model, data_generator, PLUS_INDEX, EQUALS_INDEX


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
  assert 'output_proj' not in saved_data or saved_data['output_proj'] == ('blocks.0.attn.W_O' in saved_data['model'])
  assert 'mlp_bias' not in saved_data or saved_data['mlp_bias'] == ('blocks.0.mlp.b_in' in saved_data['model'])
  assert ('blocks.0.mlp.b_in' in saved_data['model']) == ('blocks.0.mlp.b_out' in saved_data['model'])
  MODELS.append({'dir': d,
            'epoch': saved_data['epoch'],
            'd_model': saved_data['d_model'],
            'd_mlp': saved_data['d_mlp'],
            'mlp_bias': 'blocks.0.mlp.b_out' in saved_data['model'],
            'output_proj': 'blocks.0.attn.W_O' in saved_data['model'],
            'resid_mlp': saved_data.get('resid_mlp', True),
            'resid_attn': saved_data.get('resid_attn', True),
            'loss': saved_data['train_loss'][-1],
            'accuracy': saved_data['accuracy'][-1],
          })
MODELS = pd.DataFrame(MODELS)


def load(*, path=None, epoch=None, d_model=None, d_mlp=None, mlp_bias=True, output_proj=True, resid_mlp=True, resid_attn=True):
  if path is not None:
    return load_data(path)
  else:
    assert d_model is not None and d_mlp is not None
    models = MODELS.set_index(['d_model', 'd_mlp', 'mlp_bias', 'output_proj', 'resid_mlp', 'resid_attn']).sort_index()
    ds = models.loc[(d_model, d_mlp, mlp_bias, output_proj, resid_mlp, resid_attn)].sort_values('epoch')
    d = ds.iloc[-1]['dir']
    return load_epoch(d, epoch=epoch)



def load_model(*, path=None, epoch=None, d_model=None, d_mlp=None, mlp_bias=True, output_proj=True, resid_mlp=True, resid_attn=True):
  saved_data = load(path=path, epoch=epoch, d_model=d_model, d_mlp=d_mlp, mlp_bias=mlp_bias, output_proj=output_proj, resid_mlp=resid_mlp, resid_attn=resid_attn)
  d_model = saved_data['d_model']
  d_mlp = saved_data['d_mlp']
  model = create_model(d_model=d_model, d_mlp=d_mlp, mlp_bias=mlp_bias, output_proj=output_proj, resid_mlp=resid_mlp, resid_attn=resid_attn )
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


def show_tokens(tokens: th.Tensor):
  assert len(tokens) == 5 + 5 + 6 + 2
  assert tokens[5].item() == PLUS_INDEX
  assert tokens[11].item() == EQUALS_INDEX
  is_carry = [None] * 5
  carry = 0
  for i in range(5):
    a = tokens[4-i].item()
    b = tokens[10-i].item()
    assert 0 <= a <= 9
    assert 0 <= b <= 9
    if a + b > 9:
      is_carry[4-i] = 1
    elif a + b + carry > 9:
      is_carry[4-i] = 2
    else:
      is_carry[4-i] = 0
    carry = 1 if a + b + carry > 9 else 0
  return ''.join(
      [str(tokens[i].item()) + '\u0332' * is_carry[i] for i in range(5)]
    + [' + ']
    + [str(tokens[6+i].item()) + '\u0332' * is_carry[i] for i in range(5)]
    + [' = ']
    + [str(tokens[12].item())]
    + [str(tokens[13+i].item()) + '\u0332' * is_carry[i] for i in range(5)]
  )


def get_accuracy(model, tokens):
  answer = run_model(model, tokens)
  correct = (answer == tokens[:, -(NUM_DIGITS+1):].detach().cpu().numpy())
  return correct.mean()



########################    visualisations     ########################


def plot_attn(attns, labels=None, plotly=True, color_palette='RdBu', color_center:bool=True, title:str|None=None):
  assert attns.ndim == 3
  assert attns.shape[-1] == attns.shape[-2]
  if isinstance(attns, th.Tensor):
    attns = attns.numpy()
  if labels is None:
    labels = list(range(attns.shape[-1]))
  if plotly:
    fig = px.imshow(np.transpose(attns, axes=[0, 2, 1]),
                    facet_col=0, facet_col_wrap=2, facet_col_spacing=0, facet_row_spacing=0, width=600, height=600, title=title,
                    color_continuous_scale=color_palette, color_continuous_midpoint=0.0 if color_center else None,
                    labels={'facet_col': 'attn_head', 'color': 'attn', 'x': 'output<br>\u2192', 'y': 'input<br>\u2190'})
    axis = {'tickmode': 'array', 'tickvals': list(range(len(labels))), 'ticktext': labels, 'fixedrange': True, 'tickangle': 0}
    fig.update_xaxes(axis)
    fig.update_yaxes(axis)
    fig.update_traces({
      'text': [[(x, y) for x in range(18)] for y in range(18)],
      'hovertemplate': 'input: "%{y}" [%{text[1]}]<br>output: "%{x}" [%{text[0]}]<br>attn: %{z:0.4f}<extra></extra>',
    })
    return fig
  else:
      fig, axs = plt.subplots(2, 2, figsize=(10, 8))
      if title is not None:
        fig.suptitle(title)
      axs = [axs[i, j] for i in [0, 1] for j in [0, 1]]
      kwargs = {}
      if color_center:
        kwargs['vmin'] = -1.0
        kwargs['vmax'] = 1.0
      for i in range(4):
        axs[i].set_title(f'attn_head={i}')
        sns.heatmap(attns[i].T,
                    ax=axs[i],
                    xticklabels=labels, yticklabels=labels, cbar=i % 2 == 1 if color_center else True, square=True,
                    cmap=color_palette, **kwargs)


def show_attn(model, a, b, plotly=True):
  assert 0 <= a <= 99999 and type(a) == int
  assert 0 <= b <= 99999 and type(b) == int
  s = a + b
  labels = list(f'{a:05n}+{b:05n}={s:06n}')
  assert len(labels) == 18
  tokens = th.tensor([10 if label == '+' else 11 if label == '=' else int(label) for label in labels])

  cache = {}
  model.cache_all(cache)
  with th.inference_mode():
    logits = model(tokens.reshape(1, -1)).detach().cpu().to(th.float64)
  model.remove_all_hooks()
  answer = np.argmax(F.softmax(logits[:, -(NUM_DIGITS+2):-1], dim=-1).numpy(), axis=-1)
  answer = ''.join(map(str, answer[0]))

  print(f'{a:05n} + {b:05n} = {s:06n} {"| " + answer if int(answer) != s else ""}')
  return plot_attn(attns=cache['blocks.0.attn.hook_attn'][0], labels=labels, plotly=plotly)



########################    analysing attention     ########################


def remove_attn_head(model, tokens, n_correct=5, n_wrong=5):
  saved_state_dict = copy.deepcopy(model.state_dict())

  def show_examples(correct, initial_correct=True):
    correct_tokens = tokens[correct.all(axis=1)]
    wrong_tokens = tokens[~correct.all(axis=1) & initial_correct]
    wrong_results = result[~correct.all(axis=1) & initial_correct]
    print('  some correct examples:                         some wrong examples:     | model answer=')
    for i, j in itertools.zip_longest(
          random.sample(range(len(correct_tokens)), min(n_correct, len(correct_tokens))),
          random.sample(range(len(wrong_tokens)), n_wrong)
        ):
      if j is not None:
        answer = ''.join(f'{w.item()}' if w == c else f'{w.item()}\u0305' for w, c in zip(wrong_results[j, -6:], wrong_tokens[j, -6:]))
      else:
        answer = ''
      print('   ',
            show_tokens(correct_tokens[i]) if i is not None else '                      ',
            '                       ',
            show_tokens(wrong_tokens[j]), '|', answer,
          )

  with th.inference_mode():
    result = run_model(model, tokens)
  correct = (result == tokens[:, -6:].numpy())
  initial_correct = correct.all(axis=1)
  print(f'initial: accuracy={correct.mean():0.4f} correct={correct.all(axis=1).sum()}/{len(tokens)} by_digit={correct.mean(axis=0)}')
  show_examples(correct)

  for i in range(4):
    model.state_dict()['blocks.0.attn.W_V'][:] = saved_state_dict['blocks.0.attn.W_V']
    model.state_dict()['blocks.0.attn.W_V'][i, :] = 0
    with th.inference_mode():
      result = run_model(model, tokens)
    correct = (result == tokens[:, -6:].numpy())
    print(f'\nhead_{i}=0: accuracy={correct.mean():0.4f} correct={correct.all(axis=1).sum()}/{len(tokens)} by_digit={correct.mean(axis=0)}')
    show_examples(correct, initial_correct)

  model.load_state_dict(saved_state_dict)