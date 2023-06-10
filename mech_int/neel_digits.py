import copy
import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from .neel_transformer import Transformer


num_layers = 1
d_vocab = 12
d_vocab_out = 10
NUM_DIGITS = 5
n_ctx = 3*NUM_DIGITS + 3
act_type = 'ReLU'
batch_size = 64
num_data = 750
lr = 1e-4
weight_decay = 0.1


PLUS_INDEX = 10
EQUALS_INDEX = 11


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

ROOT_DIR = Path(__file__).parent.parent / 'data'


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


def show(fig):
  try:
    from IPython import get_ipython
    if get_ipython() is None or "IPKernelApp" not in get_ipython().config:  # pragma: no cover
      raise ImportError("console")
    from IPython.display import display
    import plotly.graph_objects as go
    display(go.FigureWidget(fig))
  except ImportError:
    fig.show()

def line(x, y=None, hover=None, xaxis='', yaxis='', plotly=False, title=None, **kwargs):
  if type(y)==torch.Tensor:
    y = to_numpy(y, flat=True)
  if type(x)==torch.Tensor:
    x=to_numpy(x, flat=True)
  if plotly:
    fig = px.line(x, y=y, hover_name=hover, title=title, **kwargs)
    fig.update_layout(xaxis_title=xaxis, yaxis_title=yaxis)
    show(fig)
  else:
    if y is None:
        y = x
        x = np.arange(len(y))
    plt.plot(x, y, **kwargs)
    plt.title(title)
    plt.xlabel(xaxis)
    plt.ylabel(xaxis)

def lines(lines_list, x=None, mode='lines', labels=None, xaxis='', yaxis='', title = '', log_y=False, hover=None, plotly=False, **kwargs):
  # Helper function to plot multiple lines
  if type(lines_list)==torch.Tensor:
    lines_list = [lines_list[i] for i in range(lines_list.shape[0])]
  if x is None:
    x=np.arange(len(lines_list[0]))
  if plotly:
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
    show(fig)
  else:
    ax = pd.DataFrame({labels[c] if labels else c: to_numpy(line) for c, line in enumerate(lines_list)}, index=x).plot(title=title, logy=log_y, **kwargs)
    ax.set_xlabel(xaxis)
    ax.set_ylabel(xaxis)



def data_generator(batch_size, num_digits=NUM_DIGITS):
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
    yield batch.to(DEVICE), carries.to(DEVICE)



def create_model(*, d_model: int, d_mlp: int, d_head: int | None = None, num_heads:int=4):
  if d_head is None:
    d_head = d_model//num_heads
  assert d_model == d_head * num_heads
  print(f'creating new model {d_model=} {d_mlp=} {d_head=} {num_layers=} {num_heads=}')
  model = Transformer(num_layers=num_layers,
            d_vocab=d_vocab,
            d_model=d_model,
            d_mlp=d_mlp,
            d_head=d_head,
            num_heads=num_heads,
            n_ctx=n_ctx,
            act_type=act_type,
            d_vocab_out=d_vocab_out)
  model.to(DEVICE)
  return model


def get_pred_log_probs(logits, tokens):
  trunc_logits = logits[:, -(NUM_DIGITS+2):-1]
  ans_tokens = tokens[:, -(NUM_DIGITS+1):]
  log_probs = F.log_softmax(trunc_logits.to(torch.float64), dim=-1)
  pred_log_probs = torch.gather(log_probs, -1, ans_tokens[:, :, None])[..., 0]
  return log_probs, pred_log_probs


def train(*, d_model:int=512, d_mlp:int|None=None, num_epochs:int=3_000, d_head:int|None=None, num_heads:int=4, is_finite:bool=False, seed:int=129000, checkpoint_models:bool=True, checkpoint_every:int=50, plot=True):
  global model, optimizer, scheduler, test_ds, train_ds, ds, train_losses, train_carries, per_token_losses, tokens, logits, train_logits, train_tokens, epoch, per_token_losses_train, train_losses, test_tokens, test_logits, per_token_losses_test, test_losses, ptl_train_list, ptl_test_list, epochs, state_dicts, per_token_losses_list

  if d_mlp is None:
    d_mlp = 4 * d_model

  torch.manual_seed(seed)

  run_name = datetime.datetime.utcnow().strftime('digits_%Y-%m-%d_%H-%M-%S')
  data_dir = ROOT_DIR / ('finite' if is_finite else 'infinite') / run_name
  if checkpoint_models:
    data_dir.mkdir(exist_ok=True, parents=True)

  if is_finite:
    test_ds = data_generator(batch_size)
    train_ds = data_generator(num_data)
    train_tokens, train_carries = next(train_ds)
  else:
    ds = data_generator(batch_size)

  state_dicts = []
  epochs = [0]

  model = create_model(d_model=d_model, d_mlp=d_mlp, d_head=d_head, num_heads=num_heads)

  optimizer = optim.AdamW(model.parameters(),
              lr=lr,
              weight_decay=weight_decay,
              betas=(0.9, 0.98))
  scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(step/10, 1))

  if is_finite:
    train_losses = []
    ptl_train_list = []
    test_losses = []
    ptl_test_list = []
    state_dicts.append(copy.deepcopy(model.state_dict()))

    for epoch in tqdm(range(num_epochs)):
      train_logits = model(train_tokens)
      _, all_losses = get_pred_log_probs(train_logits, train_tokens)
      per_token_losses_train = -all_losses.mean(0)
      ptl_train_list.append(to_numpy(per_token_losses_train))
      train_loss = per_token_losses_train.mean()
      train_loss.backward()
      train_losses.append(train_loss.item())

      optimizer.step()
      scheduler.step()

      with torch.inference_mode():
        test_tokens, _ = next(test_ds)
        test_logits = model(test_tokens)
        _, all_loses = get_pred_log_probs(test_logits, test_tokens).mean(0)
        per_token_losses_test = -all_losses.mean(0)
        ptl_test_list.append(to_numpy(per_token_losses_test))
        test_loss = per_token_losses_test.mean()
        test_losses.append(test_loss.item())

      optimizer.zero_grad()
      if epoch % 100 == 0:
        print(epoch, train_loss.item(), test_loss.item())
      if epoch%1000 ==0 and epoch>0:
        if plot:
          lines([train_losses, test_losses], labels=['train', 'test'])
          lines([[ptl_train_list[j][i] for j in range(len(ptl_train_list))] for i in range(1+NUM_DIGITS)]+[train_losses]+[[ptl_test_list[j][i] for j in range(len(ptl_train_list))] for i in range(1+NUM_DIGITS)]+[test_losses],
                labels = [f'tok train {i}' for i in range(1+NUM_DIGITS)]+['train_loss']+[f'tok test {i}' for i in range(1+NUM_DIGITS)]+['test_loss'],
                title='Per-digit Loss Curves for 5 digit addition (Finite Data)',
                xaxis='Step',
                yaxis='Loss')
      if checkpoint_models:
        if (epoch+1) % (checkpoint_every) == 0:
          state_dicts.append(copy.deepcopy(model.state_dict()))
          epochs.append(epoch+1)
          save_dict = {
            'model': model.state_dict(),
            'train_loss': train_losses,
            'test_loss': test_losses,
            'epoch': epoch,
            'train_tokens': train_tokens,
            'test_tokens': test_tokens,
          }
          torch.save(save_dict, data_dir / f"{epoch}.pth")




  if not is_finite:
    train_losses = []
    per_token_losses_list = []
    accuracy_list = []
    per_token_accuracy_list = []
    state_dicts.append(copy.deepcopy(model.state_dict()))
    for epoch in tqdm(range(num_epochs)):
      tokens, carry = next(ds)
      logits = model(tokens)
      log_probs, all_loses = get_pred_log_probs(logits, tokens)
      per_token_losses = -all_loses.mean(0)
      per_token_losses_list.append(to_numpy(per_token_losses))
      loss = per_token_losses.mean()
      loss.backward()
      optimizer.step()
      scheduler.step()
      optimizer.zero_grad()
      train_losses.append(loss.item())
      answer = np.argmax(log_probs.detach().cpu().numpy(), axis=-1)
      correct = (answer == tokens[:, -(NUM_DIGITS+1):].detach().cpu().numpy())
      per_token_accuracy = correct.mean(axis=0)
      accuracy = correct.mean()
      accuracy_list.append(accuracy); per_token_accuracy_list.append(per_token_accuracy)
      if epoch % 100 == 0:
        print(epoch, f'loss={loss.item()} {accuracy=}')
      if checkpoint_models:
        if (epoch+1) % (checkpoint_every) == 0:
          state_dicts.append(copy.deepcopy(model.state_dict()))
          epochs.append(epoch+1)
          save_dict = {
            'seed': seed,
            'model': model.state_dict(),
            'train_loss': train_losses,
            #   'test_loss': test_losses,
            'epoch': epoch,
            'tokens': tokens,
            #   'test_tokens': test_tokens,
            'per_token_losses_list': per_token_losses_list,
            'accuracy': accuracy_list,
            'per_token_accuracy': per_token_accuracy_list,
          }
          torch.save(save_dict, data_dir / f"{epoch}.pth")

    print(f'{epoch=} loss={loss.item()} per_token_losses={per_token_losses.detach().cpu()} {accuracy=:0.4f} {per_token_accuracy=}')

    if plot:
      line(train_losses, title='train loss')
      per_token_losses = np.stack(per_token_losses_list, axis=0)
      lines([per_token_losses[:, i] for i in range(1+NUM_DIGITS)]+[train_losses],
          labels = [f'tok {i}' for i in range(1+NUM_DIGITS)]+['train_loss'],
          title='Per-digit Loss Curves for 5 digit addition (Infinite Data)',
          xaxis='Step',
          yaxis='Loss')

      lines([per_token_losses[:, i] for i in range(1+NUM_DIGITS)]+[train_losses],
          labels = [f'tok {i}' for i in range(1+NUM_DIGITS)]+['train_loss'],
          title='Per-digit Loss Curves for 5 digit addition (Infinite Data) - Log Y Axis',
          xaxis='Step',
          yaxis='log Loss',
          log_y=True)