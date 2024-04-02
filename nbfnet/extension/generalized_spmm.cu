#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/ParallelOpenMP.h>

#include <tuple>
#include <iostream>

namespace at {

using namespace at::sparse;

// tuned on ctd chemical-gene, disease-gene
const int kWarpWorkload = 64;

__global__ void divide_sparse_by_row_out_cuda(const int64_t *indices_p, int64_t *row_index_p, int64_t nnz) {
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_thread = gridDim.x * blockDim.x;

    for (int64_t i = thread_id; i < nnz / 2; i += num_thread)
        if (i == 0 || indices_p[i] > indices_p[i - 1])
             row_index_p[indices_p[i]] = i;
}

template <class scalar_t>
__global__ void spmm_max_forward_out_cuda(
        const int64_t *indices_p, const scalar_t *values_p, const scalar_t *input_p, const int64_t *row_index_p,
        scalar_t *max_p, int64_t *max_indices_p,
        int64_t nnz, int64_t dim, int64_t num_row) {
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_group = (dim + kWarpWorkload - 1) / kWarpWorkload;
    const int group_size = num_group * warpSize;
    const int group_id = thread_id % group_size / warpSize;
    const int lane_id = thread_id % warpSize;
    const int num_thread = gridDim.x * blockDim.x;

    if (thread_id / group_size >= num_thread / group_size)
        return;

    for (int64_t row = thread_id / group_size; row < num_row; row += num_thread / group_size) {
        if (row_index_p[row] == -1)
            continue;
        for (int64_t i = row_index_p[row]; indices_p[i] == row && i < nnz; i++) {
            int64_t column = indices_p[i + nnz];
            scalar_t value = values_p[i];

            for (int64_t j = lane_id; j < kWarpWorkload; j += warpSize) {
                int64_t k = group_id * kWarpWorkload + j;
                if (k >= dim)
                    break;
                scalar_t product = input_p[column * dim + k] * value;
                scalar_t &max = max_p[row * dim + k];
                int64_t &max_indice = max_indices_p[row * dim + k];
                if (max_indice < 0 || product > max) {
                    max_indice = column;
                    max = product;
                }
            }
        }
    }
}

template <class scalar_t>
__global__ void spmm_max_backward_out_cuda(
        const int64_t * indices_p, const scalar_t *values_p, const int64_t *max_indices_p, const int64_t *row_index_p,
        const scalar_t *output_grad_p,
        scalar_t *input_grad_p,
        int64_t nnz, int64_t dim, int64_t num_row) {
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_group = (dim + kWarpWorkload - 1) / kWarpWorkload;
    const int group_size = num_group * warpSize;
    const int group_id = thread_id % group_size / warpSize;
    const int lane_id = thread_id % warpSize;
    const int num_thread = gridDim.x * blockDim.x;

    if (thread_id / group_size >= num_thread / group_size)
        return;

    for (int64_t row = thread_id / group_size; row < num_row; row += num_thread / group_size) {
        if (row_index_p[row] == -1)
            continue;
        for (int64_t i = row_index_p[row]; indices_p[i] == row && i < nnz; i++) {
            int64_t column = indices_p[i + nnz];
            scalar_t value = values_p[i];

            for (int64_t j = lane_id; j < kWarpWorkload; j += warpSize) {
                int64_t k = group_id * kWarpWorkload + j;
                if (k >= dim)
                    break;
                if (max_indices_p[row * dim + k] == column)
                    atomicAdd(&input_grad_p[column * dim + k], output_grad_p[row * dim + k] * value);
            }
        }
    }
}

/*
 * mask: m * n, nnz elements
 * input: n * d
 * max, max_indices: m * d
 * serial computation: O(nnz * d)
 * expected parallel computation: O(nnz * d / min(group_size * row, num_thread))
 * worst parallel computation: O(column * d / group_size)
 * for degree unbalanced graph, it is close to worst parallel computation
 */
std::tuple<Tensor, Tensor> spmm_max_forward_cuda(const SparseTensor &mask, const Tensor &input_) {
    TORCH_CHECK(mask.is_coalesced(), "spmm_max_forward: mask is uncoalesced");
    TORCH_CHECK(mask.sparse_dim() == 2 && mask.dense_dim() == 0,
                "spmm_max_forward: mask should be a sparse 2D tensor");
    TORCH_CHECK(mask.dtype() == input_.dtype(),
                "spmm_max_forward: operands have incompatible dtype; mask has dtype ", mask.dtype(),
                ", but input has dtype ", input_.dtype());
    TORCH_CHECK(input_.dim() <= 2, "spmm_max_forward: input should be either a 1D or 2D tensor");
    TORCH_CHECK(mask.size(1) == input_.size(0),
                "spmm_max_forward: operands have incompatible sizes; mask has size ", mask.sizes(),
                ", but input has size ", input_.sizes());
    TORCH_CHECK(mask.is_cuda(), "spmm_max_forward: expected `mask` to be CUDA, but got CPU");
    TORCH_CHECK(input_.is_cuda(), "spmm_max_forward: expected `input` to be CUDA, but got CPU");

    const Tensor indices_ = mask.indices();
    const Tensor values_ = mask.values();
    const Tensor input = input_.contiguous();

    Tensor flatten_indices = indices_.select(0, 0) * mask.size(1) + indices_.select(0, 1);
    Tensor order = flatten_indices.argsort();
    Tensor indices = indices_.gather(1, order.unsqueeze(0).expand_as(indices_)).contiguous();
    Tensor values = values_.gather(0, order).contiguous();

    int64_t nnz = values.size(0);
    int64_t num_row = mask.size(0);
    int64_t dim = input.numel() / input.size(0);

    Tensor row_index = -at::ones({num_row}, indices.options());
    std::vector<int64_t> sizes = input.sizes().vec();
    sizes[0] = num_row;
    Tensor max = at::full(sizes, -std::numeric_limits<float>::infinity(), values.options());
    Tensor max_indices = -at::ones(sizes, indices.options());

    cudaSetDevice(input.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    divide_sparse_by_row_out_cuda<<<512, 512, 0, stream>>>(
        indices.data_ptr<int64_t>(),
        row_index.data_ptr<int64_t>(),
        nnz);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "spmm_max_forward_out_cuda", [&] {
        spmm_max_forward_out_cuda<scalar_t><<<512, 512, 0, stream>>>(
            indices.data_ptr<int64_t>(),
            values.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            row_index.data_ptr<int64_t>(),
            max.data_ptr<scalar_t>(),
            max_indices.data_ptr<int64_t>(),
            nnz,
            dim,
            num_row
        );
    });

    return std::make_tuple(max, max_indices);
}

/*
 * mask: m * n, nnz elements
 * input: n * d
 * max, max_indices: m * d
 * serial computation: O(nnz * d)
 * expected parallel computation: O(nnz * d / min(group_size * row, num_thread))
 * worst parallel computation: O(column * d / group_size)
 * for degree unbalanced graph, it is close to worst parallel computation
 */
std::tuple<Tensor, Tensor> spmm_max_backward_cuda(const SparseTensor &mask, const Tensor &max_indices_,
                                                  const Tensor &output_grad_) {
    TORCH_CHECK(mask.is_coalesced(), "spmm_max_backward: mask is uncoalesced");
    TORCH_CHECK(mask.sparse_dim() == 2 && mask.dense_dim() == 0,
                "spmm_max_forward: mask should be a sparse 2D tensor");
    TORCH_CHECK(mask.dtype() == output_grad_.dtype(),
                "spmm_max_backward: operands have incompatible dtype; mask has dtype ", mask.dtype(),
                ", but output_grad has dtype ", output_grad_.dtype());
    TORCH_CHECK(max_indices_.sizes() == output_grad_.sizes(),
                "spmm_max_backward: operands have incompatible sizes; max_indices has size ",
                max_indices_.sizes(), ", but output_grad has size ", output_grad_.sizes());
    TORCH_CHECK(mask.size(0) == output_grad_.size(0),
                "spmm_max_backward: operands have incompatible sizes; mask has size ", mask.sizes(),
                ", but output_grad has size ", output_grad_.sizes());
    TORCH_CHECK(mask.is_cuda(), "spmm_max_backward: expected `mask` to be CUDA, but got CPU");
    TORCH_CHECK(max_indices_.is_cuda(), "spmm_max_backward: expected `max_indices` to be CUDA, but got CPU");
    TORCH_CHECK(output_grad_.is_cuda(), "spmm_max_backward: expected `output_grad` to be CUDA, but got CPU");

    const Tensor indices_ = mask.indices();
    const Tensor values_ = mask.values();
    const Tensor max_indices = max_indices_.contiguous();
    const Tensor output_grad = output_grad_.contiguous();
    Tensor flatten_indices = indices_.select(0, 0) * mask.size(1) + indices_.select(0, 1);
    Tensor order = flatten_indices.argsort();
    Tensor indices = indices_.gather(1, order.unsqueeze(0).expand_as(indices_)).contiguous();
    Tensor values = values_.gather(0, order).contiguous();

    int64_t nnz = values.size(0);
    int64_t num_row = mask.size(0);
    int64_t dim = output_grad.numel() / output_grad.size(0);

    Tensor row_index = -at::ones({num_row}, indices.options());
    std::vector<int64_t> sizes = output_grad.sizes().vec();
    sizes[0] = mask.size(1);
    Tensor mask_grad = at::zeros({1}, values.options());
    Tensor input_grad = at::zeros(sizes, values.options());

    cudaSetDevice(output_grad.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    divide_sparse_by_row_out_cuda<<<512, 512, 0, stream>>>(
        indices.data_ptr<int64_t>(),
        row_index.data_ptr<int64_t>(),
        nnz);

    AT_DISPATCH_FLOATING_TYPES(input_grad.type(), "spmm_max_backward", [&] {
        spmm_max_backward_out_cuda<scalar_t><<<512, 512, 0, stream>>>(
            indices.data_ptr<int64_t>(),
            values.data_ptr<scalar_t>(),
            max_indices.data_ptr<int64_t>(),
            row_index.data_ptr<int64_t>(),
            output_grad.data_ptr<scalar_t>(),
            input_grad.data_ptr<scalar_t>(),
            nnz,
            dim,
            num_row
        );
    });

    return std::make_tuple(mask_grad, input_grad);
}

} // namespace at
