"""Fixed GT geometry pooling without CUDA sparse matrix multiplication."""

import torch


# Bound each gathered feature temporary to ~4 MiB in FP32, including for
# 24 prompts x 128 dimensions. Never materialize (all nonzeros, features).
_CHUNK_ELEMENTS = 1 << 20


def _weighted_sum(features, source, destination, weights, output_rows):
    output = features.new_zeros((output_rows, features.shape[1]))
    chunk = max(1, _CHUNK_ELEMENTS // max(1, features.shape[1]))
    for start in range(0, source.numel(), chunk):
        end = start + chunk
        gathered = features.index_select(0, source[start:end])
        gathered = gathered * weights[start:end, None]
        output.index_add_(0, destination[start:end], gathered)
    return output


class _FixedGeometryPool(torch.autograd.Function):
    """Y = P X; backward is P.T grad, both using bounded indexed sums.

    Only geometry is saved, not X, gathered features or per-chunk autograd
    graphs. Backward accumulates into one feature-gradient buffer across
    chunks. GT coverage is fixed and must never require gradients.
    """

    @staticmethod
    def forward(ctx, features, rows, columns, weights, output_rows):
        ctx.save_for_backward(rows, columns, weights)
        ctx.input_rows = features.shape[0]
        return _weighted_sum(features, columns, rows, weights, output_rows)

    @staticmethod
    def backward(ctx, grad_output):
        rows, columns, weights = ctx.saved_tensors
        grad_features = _weighted_sum(
            grad_output, rows, columns, weights, ctx.input_rows)
        return grad_features, None, None, None, None


def pool_instances(pool, features):
    """Apply a detached coalesced COO GT pool to dense feature rows.

    COO is used only to store/coalesce mask geometry. Forward and backward
    use index_select, multiplication and index_add_, for every instance count
    (including zero). Empty/invalid rows are zero and retain a gradient path.
    Accumulation uses the caller's feature dtype (FP32 in the fusion neck).
    CUDA indexed sums may differ in their last bits due to addition order.
    """
    if (pool.layout != torch.sparse_coo or pool.ndim != 2
            or not pool.is_coalesced()):
        raise ValueError('Instance geometry must be a coalesced 2-D COO tensor.')
    if pool.requires_grad:
        raise ValueError('GT pooling geometry must not require gradients.')
    if features.ndim != 2 or features.shape[0] != pool.shape[1]:
        raise ValueError('Feature rows must match the GT pool columns.')
    if pool.device != features.device or pool.dtype != features.dtype:
        raise ValueError('GT pool and features must share device and dtype.')
    rows, columns = pool.indices()
    return _FixedGeometryPool.apply(
        features, rows, columns, pool.values(), pool.shape[0])


class _FixedLogGeometryPool(torch.autograd.Function):
    """Log(P exp(X)) without underflow or saved nonzeros x prompts tensors."""

    @staticmethod
    def forward(ctx, features, rows, columns, log_weights, output_rows):
        width = features.shape[1]
        chunk = max(1, _CHUNK_ELEMENTS // max(1, width))
        maximum = features.new_full((output_rows, width), float('-inf'))
        for start in range(0, columns.numel(), chunk):
            end = start + chunk
            terms = features.index_select(0, columns[start:end]) + log_weights[start:end, None]
            index = rows[start:end, None].expand(-1, width)
            maximum.scatter_reduce_(0, index, terms, reduce='amax', include_self=True)
        occupied = torch.zeros(output_rows, dtype=torch.bool, device=features.device)
        occupied[rows] = True
        maximum.masked_fill_(~occupied[:, None], 0.)
        total = features.new_zeros(maximum.shape)
        for start in range(0, columns.numel(), chunk):
            end = start + chunk
            terms = features.index_select(0, columns[start:end]) + log_weights[start:end, None]
            terms = (terms - maximum.index_select(0, rows[start:end])).exp()
            total.index_add_(0, rows[start:end], terms)
        total.masked_fill_(~occupied[:, None], 1.)
        output = maximum + total.log()
        ctx.save_for_backward(features, rows, columns, log_weights, maximum, total)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        features, rows, columns, log_weights, maximum, total = ctx.saved_tensors
        width = features.shape[1]
        chunk = max(1, _CHUNK_ELEMENTS // max(1, width))
        gradient = torch.zeros_like(features)
        for start in range(0, columns.numel(), chunk):
            end = start + chunk
            source, destination = rows[start:end], columns[start:end]
            terms = features.index_select(0, destination) + log_weights[start:end, None]
            # Keep the shift and normalizer separate: subtracting the final
            # large log mean would lose small normalization terms in FP32.
            terms = (terms - maximum.index_select(0, source)).exp()
            terms = terms / total.index_select(0, source)
            terms = terms * grad_output.index_select(0, source)
            gradient.index_add_(0, destination, terms)
        return gradient, None, None, None, None


def pool_log_instances(pool, log_features):
    """Log arithmetic GT mean with bounded forward and backward temporaries.

    Requires finite log features and positive stored geometry weights, as
    produced by the fusion neck. Empty GT rows return zero log placeholders.
    Only the small dense input/output and fixed geometry are saved. The
    original linear pool and existing class-budget log pool are unchanged.
    """
    if (pool.layout != torch.sparse_coo or pool.ndim != 2
            or not pool.is_coalesced() or pool.requires_grad):
        raise ValueError('Log pooling requires detached coalesced 2-D COO geometry.')
    if (log_features.ndim != 2 or log_features.shape[0] != pool.shape[1]
            or pool.device != log_features.device or pool.dtype != log_features.dtype):
        raise ValueError('Log features and GT pool must have matching rows, device and dtype.')
    rows, columns = pool.indices()
    return _FixedLogGeometryPool.apply(
        log_features, rows, columns, pool.values().log(), pool.shape[0])
