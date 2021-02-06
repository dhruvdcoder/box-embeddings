from box_embeddings.parameterizations.box_tensor import BoxTensor
from .pooling import BoxPooler
import torch


def gumbel_intersection_pooler(
    boxes: BoxTensor,
    beta: float = 1e-4,
    mask: torch.BoolTensor = None,
    dim: int = 0,
    keepdim: bool = False,
) -> BoxTensor:
    box_z = boxes.z
    box_Z = boxes.Z

    if mask is not None:
        box_z[mask] -= float("inf")
        box_Z[mask] += float("inf")
    z = beta * torch.logsumexp(box_z / beta, dim=dim, keepdim=keepdim)
    Z = -beta * torch.logsumexp(-box_Z / beta, dim=dim, keepdim=keepdim)

    return BoxTensor.from_zZ(z, Z)


@BoxPooler.register("gumbel-intersection")
class GumbelIntersectionBoxPooler(BoxPooler):

    """Pools a box tensor using hard intersection operation"""

    def __init__(
        self,
        beta: float = 1e-4,
        dim: int = 0,
        keepdim: bool = False,
    ):
        super().__init__()  # type:ignore
        self.beta = beta
        self.dim = dim
        self.keepdim = keepdim

    def forward(  # type:ignore
        self, box_tensor: BoxTensor, mask: torch.BoolTensor = None
    ) -> BoxTensor:
        return gumbel_intersection_pooler(
            box_tensor,
            beta=self.beta,
            mask=mask,
            dim=self.dim,
            keepdim=self.keepdim,
        )
