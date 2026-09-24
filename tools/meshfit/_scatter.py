"""The three torch_scatter calls the remesher makes, in plain PyTorch
(scatter_reduce_), so nothing has to be compiled against the installed torch."""
import torch


class torch_scatter:  # noqa: N801  (stands in for the module)
    @staticmethod
    def scatter_max(src, index, dim=0, out=None):
        idx = index.expand_as(src) if index.shape != src.shape else index
        out.scatter_reduce_(dim, idx, src, reduce="amax", include_self=True)
        return out, None

    @staticmethod
    def scatter_mean(src, index, dim=0, out=None):
        idx = index.expand_as(src) if index.shape != src.shape else index
        out.zero_()
        out.scatter_add_(dim, idx, src)
        cnt = torch.zeros(out.shape[0], device=src.device, dtype=src.dtype)
        cnt.scatter_add_(0, idx[:, 0], torch.ones(src.shape[0], device=src.device, dtype=src.dtype))
        out /= cnt.clamp(min=1)[:, None]
        return out
