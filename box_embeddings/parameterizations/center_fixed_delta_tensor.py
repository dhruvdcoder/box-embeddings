"""
    Implementation of center-delta box parameterization with fixed delta.
"""
from typing import List, Tuple, Union, Dict, Any, Optional, Type
from box_embeddings.parameterizations.box_tensor import (
    BoxTensor,
    BoxFactory,
    TBoxTensor,
)
from box_embeddings.common.utils import softplus_inverse
import torch
from torch import Tensor
import warnings

FIXED_DELTA = 1e-7


@BoxFactory.register_box_class("center_fixed_delta")
class CenterFixedDeltaBoxTensor(BoxTensor):

    """Unconstrained min-delta box tensor.

    For input of the shape (..., 2, box_dim), this parameterization
    defines z=w, and Z=w + delta, where w and delta come from the -2th dimension
    of the input. It uses softplus to keep the delta positive.

    """
    w2z_ratio: int = 2  #: number of parameters required per dim
    
    def __init__(
        self,
        data: torch.Tensor,
        delta: float = FIXED_DELTA,
    ):
        """

        Args:
            data: The weight that have to be of shape (..., num_dims).
                This is different from rest of the box parameterizations
                because we have a fixed delta. This class' constructor does not
                accept (z,Z) tuple because that is same as the base-class.
            delta: The fixed delta for the box
        """
        # DP: Currently we are re-creating this tensor
        # in every forward pass. Because we do not require grad
        # this should work but might be slow. Not doing anything
        # about it because we don't know if it is a problem.
        self.delta_tensor = torch.ones_like(data, requires_grad=False) * delta
        delta_2 = self.delta_tensor / 2.0
        z = data - delta_2
        Z = data + delta_2
        super().__init__((z, Z))
        self.delta = delta

    @property
    def kwargs(self) -> Dict:
        return {"delta": self.delta}

    @property
    def args(self) -> Tuple:
        return tuple()

    @property
    def Z(self) -> torch.Tensor:
        """Top right coordinate as Tensor

        Returns:
            Tensor: top right corner
        """
        assert self._Z is not None, "Not created correctly"

        return self._Z  # type:ignore

    @property
    def z(self) -> torch.Tensor:
        """Top right coordinate as Tensor

        Returns:
            Tensor: bottom left corner
        """
        assert self._z is not None, "Not created correctly"

        return self._z  # type:ignore

    @classmethod
    def W(  # type:ignore
        cls: Type[TBoxTensor],
        z: torch.Tensor,
        Z: torch.Tensor,
        delta: float = FIXED_DELTA,
    ) -> torch.Tensor:
        """Given (z,Z), it returns one set of valid box weights W, such that
        Box(W) = (z,Z).

        Args:
            z: Lower left coordinate of shape (..., hidden_dims)
            Z: Top right coordinate of shape (..., hidden_dims)
            delta: We do not need this parameter but
                have if for consistent API.

        Returns:
            Tensor: Parameters of the box. In this class, this
                will have shape (..., hidden_dims).
        """
        cls.check_if_valid_zZ(z, Z)

        return (z + Z) / 2.0

    @classmethod
    def from_vector(  # type:ignore
        cls, vector: torch.Tensor, delta: float = FIXED_DELTA
    ) -> BoxTensor:
        """Creates a box from a vector. In this class the vector acts as the center
        and we have a fixed delta.

        Args:
            vector: Tensor of shape (..., hidden_dim)
            delta: Fixed delta.

        Returns:
            A BoxTensor
        """

        return cls(vector, delta=delta)

    @classmethod
    def from_zZ( # type: ignore
        cls: Type[TBoxTensor], z: Tensor, Z: Tensor, *args: Any, **kwargs: Any
    ) -> BoxTensor:
        """Creates a box for the given min-max coordinates (z,Z).

        In the this base implementation we do this by
        stacking z and Z along -2 dim to form W.

        Args:
            z: lower left
            Z: top right
            *args: extra arguments for child class
            **kwargs: extra arguments for child class

        Returns:
            A BoxTensor

        """

        return BoxTensor.from_zZ(z, Z)

    @property
    def box_shape(self) -> Tuple:
        """Shape of z, Z and center.

        Returns:
            Shape of z, Z and center.

        Note:
            This is *not* the shape of the `data` attribute.
        """

        assert self._z.shape == self._Z.shape  # type:ignore

        return self._z.shape  # type: ignore


BoxFactory.register_box_class("center_fixed_delta_from_vector", "from_vector")(
    CenterFixedDeltaBoxTensor
)
BoxFactory.register_box_class("center_fixed_delta_from_zZ", "from_zZ")(
    CenterFixedDeltaBoxTensor
)
