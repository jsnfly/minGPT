import pytorch_lightning as pl
import torch
import math
import inspect
import re
import torch.nn as nn
from torch.nn import functional as F
from pytorch_lightning.utilities.parsing import get_init_args


class LitGPT(pl.LightningModule):

    def __init__(
        self,
        vocab,
        block_size=128,
        n_embd=768,
        n_layer=12,
        n_head=4,
        embd_pdrop=0.1,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        learning_rate=3e-4
    ):
        super().__init__()
        # auto creates self.hparams from the method signature
        self.save_hyperparameters()
        # TODO: Document separator?

        self.tok_emb = nn.Embedding(len(vocab), n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.drop = nn.Dropout(embd_pdrop)

        self.blocks = nn.Sequential(*[
            Block(self.hparams) for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, len(vocab), bias=False)

        self.apply(self._init_weights)

    def save_hyperparameters(self, *args, **kwargs):
        frame = inspect.currentframe().f_back
        init_args = get_init_args(frame)
        # Treating the vocab like a hparam makes it look ugly in Tensorboard.
        # It's is also not really a (single) parameter.
        self.vocab = init_args.pop('vocab')
        super().save_hyperparameters(*init_args.keys(), frame=frame)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, targets=None):
        b, t = x.size()
        assert t <= self.hparams.block_size, \
               "Cannot forward, model block size is exhausted."

        # forward the GPT model
        # each index maps to a (learnable) vector
        token_embeddings = self.tok_emb(x)
        # each position maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :]
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        return logits

    def configure_optimizers(self):
        # create the optimizer
        if not hasattr(self, 'param_names_decay'):
            self._group_parameters()

        params_decay, params_no_decay = [], []
        for n, p in self.named_parameters():
            if n in self.param_names_decay:
                params_decay.append(p)
            else:
                params_no_decay.append(p)

        optim_groups = [
            {
                "params": params_decay,
                "weight_decay": self.hparams.weight_decay
            },
            {
                "params": params_no_decay,
                "weight_decay": 0.0
            },
        ]
        optimizer = torch.optim.AdamW(optim_groups,
                                      lr=self.hparams.learning_rate,
                                      betas=self.hparams.betas)
        return optimizer

    def _group_parameters(self):
        # all biases, layernorm weights and embeddings weights should not be
        # decayed
        no_decay = [r".*bias$", r"(?:.*\.)?ln.*\.weight$", r"pos_emb$",
                    r"tok_emb.weight$"]
        parameter_names = [n for n, _ in self.named_parameters()]
        # These are set as instance variables for easier debugging.
        self.param_names_no_decay = [
            n for n in parameter_names
            if any(re.match(nd, n) for nd in no_decay)
        ]
        self.param_names_decay = [n for n in parameter_names
                                  if n not in self.param_names_no_decay]

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        self.log('val_loss', loss)
        return loss

    def on_save_checkpoint(self, checkpoint):
        # Must be set here, to be loaded properly.
        checkpoint['hyper_parameters']['vocab'] = self.vocab


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the
    end. It is possible to use torch.nn.MultiheadAttention here but I am
    including an explicit implementation here to show that there is nothing too
    scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in
        # the input sequence
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask[None, None, ...])
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head
        # forward to be the batch dim
        k = self.separate_heads(self.key(x))
        q = self.separate_heads(self.query(x))
        v = self.separate_heads(self.value(x))

        # causal self-attention; Self-attend:
        # (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

    def separate_heads(self, all_heads_embeddings):
        """
        Args:
            all_heads_embeddings (torch.tensor):
                keys, querries or values (batch_size, block_size, n_embd)
        Returns:
            torch.tensor: (batch_size, n_heads, block_size, n_emd // n_head)
        """
        B, T, n_embd = all_heads_embeddings.size()
        separated = all_heads_embeddings.view(B, T, self.n_head,
                                              n_embd // self.n_head)
        return separated.transpose(1, 2)
