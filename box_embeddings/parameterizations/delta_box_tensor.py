"""
    Implementation of min-delta box parameterization.
"""
from typing import List, Tuple, Union, Dict, Any, Optional, Type
from box_embeddings.parameterizations.box_tensor import (
    BoxTensor,
    BoxFactory,
    TBoxTensor,
)
from box_embeddings.common.utils import softplus_inverse
import torch
import warnings


@BoxFactory.register_box_class("mindelta")
class MinDeltaBoxTensor(BoxTensor):

    """Unconstrained min-delta box tensor.

    For input of the shape (..., 2, box_dim), this parameterization
    defines z=w, and Z=w + delta, where w and delta come from the -2th dimension
    of the input. It uses softplus to keep the delta positive.

    """

    def __init__(
        self,
        data: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        beta: float = 1.0,
        threshold: float = 20,
        minimum_delta: float = 0,
    ):
        """

        Args:
            data: The weights for the box
            beta: beta parameter for softplus for delta. Depending on the
                universe box and your inputs ranges, you might want to change this.
                Higher values of beta will make softplus harder and bring it close to ReLU.
            threshold: parameter for the softplus for delta
            minimum_delta: Minimum detla
        """
        super().__init__(data)
        self.beta = beta
        self.threshold = threshold
        self.minimum_delta = minimum_delta

    @property
    def kwargs(self) -> Dict:
        return {
            "beta": self.beta,
            "threshold": self.threshold,
            "minimum_delta": self.minimum_delta,
        }

    @property
    def args(self) -> Tuple:
        return tuple()

    @property
    def Z(self) -> torch.Tensor:
        """Top right coordinate as Tensor

        Returns:
            Tensor: top right corner
        """

        if self.data is not None:
            return (
                self.z
                + self.minimum_delta
                + torch.nn.functional.softplus(
                    self.data[..., 1, :],
                    beta=self.beta,
                    threshold=self.threshold,
                )
            )
        else:
            return self._Z  # type:ignore

    @classmethod
    def W(  # type:ignore
        cls: Type[TBoxTensor],
        z: torch.Tensor,
        Z: torch.Tensor,
        beta: float = 1.0,
        threshold: float = 20.0,
        minimum_delta: float = 0,
    ) -> torch.Tensor:
        """Given (z,Z), it returns one set of valid box weights W, such that
        Box(W) = (z,Z).

        The min coordinate is stored as is:
        W[...,0,:] = z
        W[...,1,:] = softplus_inverse(Z-z)

        The max coordinate is transformed


        Args:
            z: Lower left coordinate of shape (..., hidden_dims)
            Z: Top right coordinate of shape (..., hidden_dims)
            beta: TODO
            threshold: TODO
            minimum_delta: TODO

        Returns:
            Tensor: Parameters of the box. In base class implementation, this
                will have shape (..., 2, hidden_dims).

        Raises:
            ValueError: When given delta (i.e. Z-z) is less than minimum_delta
        """
        cls.check_if_valid_zZ(z, Z)

        if ((Z - z) < 0).any():
            warnings.warn(
                "W() method for MinDeltaBoxTensor is numerically unstable."
                " It can produce high error when input Z-z is < 0."
            )

        if ((Z - z) < minimum_delta).any():
            raise ValueError(f"Z-z should not be less than {minimum_delta}")

        return torch.stack(
            (
                z,
                softplus_inverse(
                    (Z - z) - minimum_delta, beta=beta, threshold=threshold
                ),
            ),
            -2,
        )

    #    @classmethod
    #    def from_zZ(  # type:ignore
    #        cls,
    #        z: torch.Tensor,
    #        Z: torch.Tensor,
    #        beta: float = 1.0,
    #        threshold: float = 20,
    #    ) -> BoxTensor:
    #        """Creates a box for the given min-max coordinates (z,Z).
    #
    #        In the this base implementation we do this by
    #        stacking z and Z along -2 dim to form W.
    #
    #        Args:
    #            z: lower left
    #            Z: top right
    #            beta: beta parameter for softplus for delta. Depending on the
    #                universe box and your inputs ranges, you might want to change this.
    #                Higher values of beta will make softplus harder and bring it close to ReLU.
    #            threshold: parameter for the softplus for delta
    #
    #
    #        Returns:
    #            A BoxTensor
    #
    #        """
    #
    #        return cls((z, Z), beta=beta, threshold=threshold)

    #    def like_this_from_zZ(
    #        self, z: torch.Tensor, Z: torch.Tensor,
    #    ) -> "BoxTensor":
    #        """Creates a box for the given min-max coordinates (z,Z).
    #        This is similar to the class method :method:`from_zZ`, but
    #        uses the attributes on self and not external args, kwargs.
    #
    #        Args:
    #            z: lower left
    #            Z: top right
    #
    #        Returns:
    #            A BoxTensor
    #
    #        """
    #
    #        return self.from_zZ(z, Z, beta=self.beta, threshold=self.threshold)

    @classmethod
    def from_vector(  # type:ignore
        cls,
        vector: torch.Tensor,
        beta: float = 1.0,
        threshold: float = 20,
        minimum_delta: float = 0,
    ) -> BoxTensor:
        """Creates a box for a vector. In this base implementation the vector is split
        into two pieces and these are used as z and delta.

        Args:
            vector: tensor
            beta: beta parameter for softplus for delta. Depending on the
                universe box and your inputs ranges, you might want to change this.
                Higher values of beta will make softplus harder and bring it close to ReLU.
            threshold: parameter for the softplus for delta
            minimum_delta: delta = minimum_delta + delta


        Returns:
            A BoxTensor

        Raises:
            ValueError: if last dimension is not even
        """
        len_dim = vector.shape[-1]
        dim = -1

        if vector.shape[-1] % 2 != 0:
            raise ValueError(
                f"The last dimension of vector should be even but is {vector.shape[-1]}"
            )

        split_point = int(len_dim / 2)
        z = vector.index_select(
            dim,
            torch.tensor(
                list(range(split_point)),
                dtype=torch.int64,
                device=vector.device,
            ),
        )

        delta = vector.index_select(
            dim,
            torch.tensor(
                list(range(split_point, len_dim)),
                dtype=torch.int64,
                device=vector.device,
            ),
        )

        return cls(
            torch.stack((z, delta), -2),
            beta=beta,
            threshold=threshold,
            minimum_delta=minimum_delta,
        )  # type:ignore


BoxFactory.register_box_class("mindelta_from_zZ", "from_zZ")(MinDeltaBoxTensor)
BoxFactory.register_box_class("mindelta_from_vector", "from_vector")(
    MinDeltaBoxTensor
)

BoxFactory.register_box_class("mindelta_from_center", "from_center_vector")(
    MinDeltaBoxTensor
)
