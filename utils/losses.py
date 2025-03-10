import warnings
import sys
from typing import Callable, Optional

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.modules import Module
from torch.nn import _reduction as _Reduction


__all__ = ['MarginRankingLossWithTies']


class _Loss(Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class _WeightedLoss(_Loss):
    def __init__(self, weight: Optional[Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)
        self.weight: Optional[Tensor]


class MarginRankingLossWithTies(_Loss):
    r"""

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Target: :math:`(*)`, same shape as the input.
        - Output: scalar. If :attr:`reduction` is ``'none'``, then
          :math:`(*)`, same shape as the input.

    Examples::
        >>> loss = MarginRankingLossWithTies(margin=1)
        >>> input1 = torch.randn(3, requires_grad=True)
        >>> input2 = torch.randn(3, requires_grad=True)
        >>> target = torch.randn(3).sign()
        >>> output = loss(input1, input2, target)
        >>> output.backward()
    """
    __constants__ = ['margin', 'reduction']
    margin: float

    def __init__(self, margin: float = 0., size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
        self.margin = margin

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        ties_loss_valid = torch.abs(input1 - input2) - self.margin
        zeros = torch.zeros_like(ties_loss_valid)
        loss = torch.max(ties_loss_valid, zeros)

        if self.reduction == 'mean':
            avg_loss = loss.mean()
        else:
            raise Exception("Reduction type not valid. Currently, only allows for 'mean'.")
        return avg_loss


def compute_ranking_loss(network_output_dict, labels, criterion_ranking, ties=False, criterion_ties=None, ties_w=1):
    """
    Computes the ranking loss between network outputs and labels using a loss criterion
        Parameters:
             network_output_dict (dict): Network output
             labels (nn.Tensor): Ground truth class labels
             criterion_ranking (nn.Loss): Loss function (criterion) for non-ties in pairwise comparisons
             ties (bool): Flag to indicate whether to compute loss for ties
             criterion_ties (nn.Loss): Loss function (criterion) for ties in pairwise comparisons
             ties_w (float): Weight to be used in loss sum between ties and non-ties loss

        Returns:
            loss_rank (nn.Tensor): Loss values
    """
    if ties and criterion_ties is None:
        raise Exception('If including ties, criterion (loss) for ties must be included')

    # Forward pass data output
    output_rank_left = network_output_dict['left']['output']
    output_rank_right = network_output_dict['right']['output']
    output_left = output_rank_left.view(output_rank_left.size()[0])
    output_right = output_rank_right.view(output_rank_right.size()[0])

    # Ground truth label data
    label = -1 * labels['label_r']  # switch label, such that -1 corresponds to right winning and 1 to left winning

    # Non-ties loss
    index_mask_nontie = label != 0
    label_nonties = label[index_mask_nontie]
    loss_nonties = criterion_ranking(output_left[index_mask_nontie],
                                     output_right[index_mask_nontie],
                                     label_nonties)

    # If ties, include a L1Loss for ties
    if ties:
        # Ties loss
        index_mask_tie = label == 0
        loss_ties = criterion_ties(output_left[index_mask_tie],
                                   output_right[index_mask_tie])

        # Combination of non-ties and ties rank losses
        w1 = 1
        w2 = ties_w
        loss_rank = w1 * loss_nonties + w2 * loss_ties

    # Otherwise just use MarginRankLoss
    else:
        loss_rank = loss_nonties

    return loss_rank


def compute_loss_classification(network_output_dict, labels, criterion_classification):
    """
    Computes the classification loss between network outputs and labels using a loss criterion

        Parameters:
             network_output_dict (dict): Network output
             labels (nn.Tensor): Ground truth class labels
             criterion_classification (nn.Loss): Loss function (criterion)

        Returns:
            loss_class (nn.Tensor): Loss values
    """

    # Forward pass data output
    logits = network_output_dict['logits']['output']

    # Ground truth label data
    label = labels['label_c']

    # Classification loss
    loss_class = criterion_classification(logits, label.long())
    return loss_class


def compute_loss(args, network_output_dict, labels):
    criterion_ranking = nn.MarginRankingLoss(reduction='mean', margin=args.ranking_margin)
    criterion_classification = nn.CrossEntropyLoss()
    if args.ties:
        # criterion_ties = nn.L1Loss()
        criterion_ties = MarginRankingLossWithTies(reduction='mean', margin=args.ranking_margin_ties)
    else:
        criterion_ties = None

    # Ranking Model Only
    if args.model == 'rcnn':
        loss_rank = compute_ranking_loss(network_output_dict,
                                         labels, criterion_ranking,
                                         ties=args.ties, criterion_ties=criterion_ties, ties_w=args.ties_w)
        return loss_rank

    # Classification Model Only
    elif args.model == 'sscnn':
        loss_class = compute_loss_classification(network_output_dict,
                                                 labels,
                                                 criterion_classification)
        return loss_class

    # Classification + Ranking Model
    elif args.model == 'rsscnn':
        w1 = 1  # Weight for classification loss
        w2 = args.rank_w  # Weight for ranking loss

        loss_class = compute_loss_classification(network_output_dict,
                                                 labels,
                                                 criterion_classification)

        loss_rank = compute_ranking_loss(network_output_dict,
                                         labels, criterion_ranking,
                                         ties=args.ties, criterion_ties=criterion_ties, ties_w=args.ties_w)

        # Return combination of 2 losses
        return w1 * loss_class + w2 * loss_rank

    else:
        print('Invalid model specification. Aborting.')
        sys.quit(-1)


if __name__ == '__main__':

    torch.manual_seed(8)

    loss = MarginRankingLossWithTies(margin=1)
    input1 = torch.randn(3, requires_grad=True)
    print(input1)
    input2 = torch.randn(3, requires_grad=True)
    print(input2)
    print(input1 - input2)
    output = loss(input1, input2)
    output.backward()

    print(output)
