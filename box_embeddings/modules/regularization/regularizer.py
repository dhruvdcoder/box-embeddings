from typing import List, Tuple, Union, Dict, Any, Optional
from box_embeddings.common.registrable import Registrable
import torch
from box_embeddings.parameterizations.box_tensor import BoxTensor


class BoxRegularizer(torch.nn.Module, Registrable):

    """Base box-regularizer class"""

    def __init__(
        self, weight: float, log_scale: bool = True, **kwargs: Any
    ) -> None:
        """
        Args:
            weight: Weight (hyperparameter) given to this regularization in the overall loss.
            log_scale: Whether the output should be in log scale or not.
                Should be true in almost any practical case where box_dim>5.
            kwargs: Unused
        """
        super().__init__()  # type:ignore
        self.weight = weight
        self.log_scale = log_scale

    def forward(self, box_tensor: BoxTensor) -> Union[float, torch.Tensor]:
        """Calls the _forward and multiplies the weight

        Args:
            box_tensor: Input box tensor

        Returns:
            scalar regularization loss
        """

        return self.weight * self._forward(box_tensor)

    def _forward(self, box_tensor: BoxTensor) -> Union[float, torch.Tensor]:
        """The method that does all the work and needs to be overriden

        Args:
            box_tensor: Input box tensor

        Returns:
            0
        """

        return 0.0
