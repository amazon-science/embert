from typing import Optional, Callable, Tuple, Any, Union

import torch
from torch import Tensor
from torchmetrics import Metric


class TrajectoryPrecision(Metric):
    def __init__(
            self,
            compute_on_step: bool = True,
            dist_sync_on_step: bool = False,
            process_group: Optional[Any] = None,
            dist_sync_fn: Callable = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state("precision", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None):
        """
        Update state with predictions and targets. See :ref:`references/modules:input types` for more information
        on input types.

        Args:
            preds: Predictions from model (probabilities, or labels)
            target: Ground truth labels
        """

        precision = _trajectory_precision_update(
            preds, target, mask
        )

        self.precision += precision
        self.total += preds.shape[0]

    def compute(self) -> torch.Tensor:
        """
        Computes accuracy based on inputs passed in to ``update`` previously.
        """
        if self.total > 1e-12:
            precision = self.precision / self.total
        else:
            precision = self.num_same.new_zeros((1,))

        return precision


class AverageMetric(Metric):
    """Composition of two metrics with a specific operator which will be executed upon metrics compute """

    def __init__(
            self,
            metric_a: Union[Metric, int, float, Tensor],
            metric_b: Union[Metric, int, float, Tensor, None]
    ):
        """
        Args:
            operator: the operator taking in one (if metric_b is None)
                or two arguments. Will be applied to outputs of metric_a.compute()
                and (optionally if metric_b is not None) metric_b.compute()
            metric_a: first metric whose compute() result is the first argument of operator
            metric_b: second metric whose compute() result is the second argument of operator.
                For operators taking in only one input, this should be None
        """
        super().__init__()

        if isinstance(metric_a, Tensor):
            self.register_buffer("metric_a", metric_a)
        else:
            self.metric_a = metric_a

        if isinstance(metric_b, Tensor):
            self.register_buffer("metric_b", metric_b)
        else:
            self.metric_b = metric_b

    def _sync_dist(self, dist_sync_fn: Callable = None) -> None:
        # No syncing required here. syncing will be done in metric_a and metric_b
        pass

    def update(self, *args, **kwargs) -> None:
        metric_a_args = kwargs["a"]
        if isinstance(self.metric_a, Metric):
            self.metric_a.update(*metric_a_args)

        metric_b_args = kwargs["b"]
        if isinstance(self.metric_b, Metric):
            self.metric_b.update(*metric_b_args)

    def compute(self) -> Any:

        # also some parsing for kwargs?
        if isinstance(self.metric_a, Metric):
            val_a = self.metric_a.compute()
        else:
            val_a = self.metric_a

        if isinstance(self.metric_b, Metric):
            val_b = self.metric_b.compute()
        else:
            val_b = self.metric_b

        return (val_a + val_b) / 2

    def reset(self) -> None:
        if isinstance(self.metric_a, Metric):
            self.metric_a.reset()

        if isinstance(self.metric_b, Metric):
            self.metric_b.reset()

    def persistent(self, mode: bool = False) -> None:
        if isinstance(self.metric_a, Metric):
            self.metric_a.persistent(mode=mode)
        if isinstance(self.metric_b, Metric):
            self.metric_b.persistent(mode=mode)

    def __repr__(self) -> str:
        _op_metrics = f"(\n  average(\n    {repr(self.metric_a)},\n    {repr(self.metric_b)}\n  )\n)"
        repr_str = self.__class__.__name__ + _op_metrics

        return repr_str


class Accuracy(Metric):
    def __init__(
            self,
            compute_on_step: bool = True,
            dist_sync_on_step: bool = False,
            process_group: Optional[Any] = None,
            dist_sync_fn: Callable = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None):
        """
        Update state with predictions and targets. See :ref:`references/modules:input types` for more information
        on input types.

        Args:
            preds: Predictions from model (probabilities, or labels)
            target: Ground truth labels
        """

        correct, total = _accuracy_update(
            preds, target, mask
        )

        self.correct += correct
        self.total += total

    def compute(self) -> torch.Tensor:
        """
        Computes accuracy based on inputs passed in to ``update`` previously.
        """
        if self.total > 1e-12:
            accuracy = self.correct / self.total
        else:
            accuracy = self.correct.new_zeros((1,))

        return accuracy


def _accuracy_update(
        preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    if preds.requires_grad:
        preds = preds.detach()
    pred_labels = preds.max(-1)[1].unsqueeze(-1)
    correct = pred_labels.eq(target.unsqueeze(-1)).long()

    if mask is not None:
        correct *= mask.view(-1, 1)
        _total_count = mask.sum()
    else:
        _total_count = torch.tensor(target.numel())
    _correct_count = correct.sum()

    return _correct_count, _total_count


def _trajectory_precision_update(
        preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None
) -> torch.Tensor:
    if preds.requires_grad:
        preds = preds.detach()
    pred_labels = preds.max(-1)[1].unsqueeze(-1)
    # these are the positions that are actually correct
    correct_mask = pred_labels.eq(target.unsqueeze(-1))
    if mask is not None:
        correct_mask = correct_mask * mask.unsqueeze(-1)
        total = mask.sum(-1)
    else:
        total = target.new_full((target.shape[0],), fill_value=target.shape[1])

    correct_mask = correct_mask.bool().squeeze(-1)
    precision = preds.new_zeros((target.shape[0],))

    for i in range(target.shape[0]):
        num_same = torch.bincount(target[i][correct_mask[i]]).sum()
        precision[i] = num_same / total[i]

    return precision.sum()


import re
import string
import collections


def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def test_f1_metric():
    pred = "MoveAhead25 MoveAhead25 MoveAhead25 MoveAhead25 <<stop>>"
    gold = "MoveAhead25 MoveAhead25 Lookdown_30 PickupObject <<stop>>"

    f1 = compute_f1(gold, pred)

    print(f1)


def test_accuracy_metric():
    preds = torch.tensor([[0.2, 0.45, 0.35], [0.5, 0.2, 0.3], [0.3, 0.1, 0.6]])
    targets = torch.tensor([1, 0, 0])
    mask = torch.tensor([1, 1, 0])

    accuracy = Accuracy()

    accuracy(preds, targets, mask)

    val = accuracy.compute()
    print("Accuracy with mask: {}".format(val))
    assert torch.isclose(val, torch.tensor(1.0))

    accuracy = Accuracy()

    accuracy(preds, targets)

    val = accuracy.compute()
    print("Accuracy without mask: {}".format(val))
    assert torch.isclose(val, torch.tensor(0.6666666))


def test_traj_precision_metric():
    preds = torch.tensor([[[0.2, 0.45, 0.35], [0.5, 0.2, 0.3], [0.3, 0.1, 0.6]],
                          [[0.2, 0.6, 0.2], [0.2, 0.4, 0.3], [0.6, 0.1, 0.3]]])
    targets = torch.tensor([[1, 0, 0], [1, 1, 0]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    accuracy = TrajectoryPrecision()

    accuracy(preds, targets, mask)

    val = accuracy.compute()
    print("Accuracy with mask: {}".format(val))
    # assert torch.isclose(val, torch.tensor(1.0))
