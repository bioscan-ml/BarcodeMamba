"""
Utils for training, including loss function, metrics and config printing.
"""

from typing import Any, Dict, Optional
import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Metric

try:
    from flash_attn.losses.cross_entropy import CrossEntropyLoss
except ImportError:
    CrossEntropyLoss = torch.nn.CrossEntropyLoss
try:
    from apex.transformer import parallel_state
except ImportError:
    parallel_state = None
from omegaconf import DictConfig, OmegaConf
from timm.scheduler import CosineLRScheduler
from pytorch_lightning.utilities import rank_zero_only


def cross_entropy(logits, y, ignore_index=-100):
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    return F.cross_entropy(logits, y, ignore_index=ignore_index)


def accuracy(logits, y):
    logits = logits.view(-1, logits.shape[-1])
    preds = torch.argmax(logits, dim=-1)
    if y.numel() > logits.shape[0]:
        # Mixup leads to this case: use argmax class
        y = y.argmax(dim=-1)
    y = y.view(-1)
    return torch.eq(preds, y).float().mean()


# From https://github.com/HazyResearch/hyena-dna/blob/main/src/tasks/torchmetrics.py
class Perplexity(Metric):
    r"""
    Perplexity measures how well a language model predicts a text sample. It's calculated as the average number of bits
    per word a model needs to represent the sample.
    Args:
        kwargs:
            Additional keyword arguments, see :ref:`Metric kwargs` for more info.
    Examples:
        >>> import torch
        >>> preds = torch.rand(2, 8, 5, generator=torch.manual_seed(22))
        >>> target = torch.randint(5, (2, 8), generator=torch.manual_seed(22))
        >>> target[0, 6:] = -100
        >>> metric = Perplexity(ignore_index=-100)
        >>> metric(preds, target)
        tensor(5.2545)
    """

    is_differentiable = True
    higher_is_better = False
    full_state_update = False
    total_log_probs: Tensor
    count: Tensor

    def __init__(self, **kwargs: Dict[str, Any]):
        super().__init__(**kwargs)
        self.add_state(
            "total_log_probs",
            default=torch.tensor(0.0, dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "count", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum"
        )

        self.loss_fn = CrossEntropyLoss()

    def update(self, preds: Tensor, target: Tensor, loss: Optional[Tensor] = None) -> None:  # type: ignore
        """Compute and store intermediate statistics for Perplexity.
        Args:
            preds:
                Probabilities assigned to each token in a sequence with shape [batch_size, seq_len, vocab_size].
            target:
                Ground truth values with a shape [batch_size, seq_len].
        """
        count = target.numel()
        if loss is None:
            loss = self.loss_fn(preds, target)
        self.total_log_probs += loss.double() * count
        self.count += count

    def compute(self) -> Tensor:
        """Compute the Perplexity.
        Returns:
           Perplexity
        """
        return torch.exp(self.total_log_probs / self.count)


class NumTokens(Metric):
    """Keep track of how many tokens we've seen."""

    # TODO: how do we prevent the reset between the epochs? The reset happens on the 1st batch
    # of the next epoch.
    # Right now the hack is that we override reset(), which would mess up the forward method.
    # We then override forward to do the right thing.

    is_differentiable = False
    higher_is_better = False
    full_state_update = False
    count: Tensor

    def __init__(self, **kwargs: Dict[str, Any]):
        super().__init__(**kwargs)
        self.add_state(
            "count",
            default=torch.tensor(0, dtype=torch.int64),
            dist_reduce_fx="sum",
            persistent=True,
        )  # We want the count to be saved to state-dict
        if parallel_state is not None and not parallel_state.is_unitialized():
            self.tensor_parallel_world_size = (
                parallel_state.get_tensor_model_parallel_world_size()
            )
        else:
            self.tensor_parallel_world_size = 1

    def update(self, preds: Tensor, target: Tensor, loss: Optional[Tensor] = None) -> None:  # type: ignore
        self.count += target.numel() // self.tensor_parallel_world_size

    def compute(self) -> Tensor:
        return self.count

    def reset(self):
        count = self.count
        super().reset()
        self.count = count

    # Adapted from https://github.com/Lightning-AI/metrics/blob/master/src/torchmetrics/metric.py
    def _forward_reduce_state_update(self, *args: Any, **kwargs: Any) -> Any:
        """forward computation using single call to `update` to calculate the metric value on the current batch and
        accumulate global state.
        This can be done when the global metric state is a sinple reduction of batch states.
        """
        self.update(*args, **kwargs)
        return self.compute()


# From https://github.com/HazyResearch/hyena-dna/blob/main/src/utils/optim/schedulers.py
class TimmCosineLRScheduler(CosineLRScheduler, torch.optim.lr_scheduler._LRScheduler):
    """Wrap timm.scheduler.CosineLRScheduler so we can call scheduler.step() without passing in epoch.
    It supports resuming as well.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_epoch = -1
        self.step(epoch=0)

    def step(self, epoch=None):
        if epoch is None:
            self._last_epoch += 1
        else:
            self._last_epoch = epoch
        # We call either step or step_update, depending on whether we're using the scheduler every
        # epoch or every step.
        # Otherwise, lightning will always call step (i.e., meant for each epoch), and if we set
        # scheduler interval to "step", then the learning rate update will be wrong.
        if self.t_in_epochs:
            super().step(epoch=self._last_epoch)
        else:
            super().step_update(num_updates=self._last_epoch)


