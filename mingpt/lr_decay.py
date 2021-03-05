import math
import pytorch_lightning as pl


# TODO: Does not seem to work properly on TPU. Also is not saved to continue
# Training from current state. Probably _configre_optimizer should return
# a real LRScheduler and a Callback shouldn't be used.
class LearningRateDecayCallback(pl.Callback):

    def __init__(self,
                 learning_rate,
                 warmup_tokens=375e6,
                 final_tokens=260e9,
                 lr_decay=True):
        super().__init__()
        self.learning_rate = learning_rate
        self.tokens = 0
        self.final_tokens = final_tokens
        self.lr_decay = lr_decay
        self.warmup_tokens = warmup_tokens

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx,
                           dataloader_idx):
        optimizer = trainer.optimizers[0]
        _, y = batch

        if self.lr_decay:
            # TODO:'<pad>'-Token with index -100 in dictionary?
            # number of tokens processed this step (i.e. label is not -100)
            self.tokens += (y >= 0).sum()
            if self.tokens < self.warmup_tokens:
                # linear warmup
                lr_mult = float(self.tokens) / float(
                    max(1, self.warmup_tokens)
                )
            else:
                # cosine learning rate decay
                progress = float(self.tokens - self.warmup_tokens) / float(
                    max(1, self.final_tokens - self.warmup_tokens)
                )
                # Prevent progress larger than one (else there would be
                # cosine oscillations.)
                progress = min(1, progress)
                lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
            lr = self.learning_rate * lr_mult
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            pl_module.log('learning_rate', lr,
                          on_step=True, on_epoch=False, prog_bar=True,
                          logger=True)
