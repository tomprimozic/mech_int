import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm.auto import tqdm

import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

from .neel_transformer import Transformer


num_layers = 1
d_vocab = 12
d_vocab_out = 10
num_digits = 5
n_ctx = 3*num_digits + 3
act_type = 'ReLU'
batch_size = 64
num_data = 750
lr = 1e-4
weight_decay = 0.1


PLUS_INDEX = 10
EQUALS_INDEX = 11


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



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


def data_generator(batch_size, num_digits):
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



def create_model(*, d_model: int, d_mlp: int, d_head: int, num_heads: int):
    assert d_model == d_head * num_heads
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
    trunc_logits = logits[:, -(num_digits+2):-1]
    ans_tokens = tokens[:, -(num_digits+1):]
    log_probs = F.log_softmax(trunc_logits.to(torch.float64), dim=-1)
    pred_log_probs = torch.gather(log_probs, -1, ans_tokens[:, :, None])[..., 0]
    return pred_log_probs


def train(*, d_model:int=512, d_mlp:int|None=None, num_epochs:int=3_000, d_head:int|None=None, num_heads:int=4, is_finite:bool=False, seed:int=129000, checkpoint_models:bool=True, checkpoint_every:int=50):
    global model, optimizer, scheduler, train_losses, per_token_losses, tokens, logits, train_logits, train_tokens, epoch, per_token_losses_train, train_losses, test_tokens, test_logits, per_token_losses_test, test_losses

    if d_mlp is None:
        d_mlp = 4 * d_model
    if d_head is None:
        d_head = d_model//num_heads

    torch.manual_seed(seed)

    if is_finite:
        test_ds = data_generator(batch_size, num_digits)
        train_ds = data_generator(num_data, num_digits)
        train_tokens, train_carries = next(train_ds)
    else:
        ds = data_generator(batch_size, num_digits)

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
        for epoch in tqdm(range(num_epochs)):
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


    if not is_finite:
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
