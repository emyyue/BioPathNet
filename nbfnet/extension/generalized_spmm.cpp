#include <torch/extension.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/ParallelOpenMP.h>
#include <c10/core/ScalarType.h>

#include <tuple>
#include <thread>

namespace at {

using namespace at::sparse;

// forward declaration
// a tricky macro to detect whether it is built for CPU or CUDA
#ifdef CUDA_OP
std::tuple<Tensor, Tensor> spmm_max_forward_cuda(const SparseTensor &mask, const Tensor &input_);

std::tuple<Tensor, Tensor> spmm_max_backward_cuda(const SparseTensor &mask, const Tensor &max_indices_,
                                                         const Tensor &output_grad_);
#endif

template <class scalar_t>
void spmm_max_forward_out_cpu(
        const int64_t *indices_p, const scalar_t *values_p, const scalar_t *input_p,
        scalar_t *max_p, int64_t *max_indices_p,
        int64_t nnz, int64_t dim) {
    at::parallel_for(0, dim, 0, [&](int64_t start, int64_t end){
        for (int64_t i = 0; i < nnz; i++) {
            int64_t row = indices_p[i];
            int64_t column = indices_p[i + nnz];
            scalar_t value = values_p[i];

            for (int64_t j = start; j < end; j++) {
                scalar_t product = input_p[column * dim + j] * value;
                scalar_t &max = max_p[row * dim + j];
                int64_t &max_indice = max_indices_p[row * dim + j];
                if (max_indice < 0 || product > max) {
                    max_indice = column;
                    max = product;
                }
            }
        }
    });
}

template <class scalar_t>
void spmm_max_forward_out_cpu_2(
        const int64_t *indices_p, const scalar_t *values_p, const scalar_t *input_p,
        scalar_t *max_p, int64_t *max_indices_p,
        int64_t nnz, int64_t dim, int64_t start, int64_t end) {
    for (int64_t i = 0; i < nnz; i++) {
        int64_t row = indices_p[i];
        int64_t column = indices_p[i + nnz];
        scalar_t value = values_p[i];

        for (int64_t j = start; j < end; j++) {
            scalar_t product = input_p[column * dim + j] * value;
            scalar_t &max = max_p[row * dim + j];
            int64_t &max_indice = max_indices_p[row * dim + j];
            if (max_indice < 0 || product > max) {
                max_indice = column;
                max = product;
            }
        }
    }
}

template <class scalar_t>
void spmm_max_backward_out_cpu(
        const int64_t * indices_p, const scalar_t *values_p, const int64_t *max_indices_p,
        const scalar_t *output_grad_p,
        scalar_t *input_grad_p,
        int64_t nnz, int64_t dim) {
    at::parallel_for(0, dim, 0, [&](int64_t start, int64_t end){
        for (int64_t i = 0; i < nnz; i++) {
            int64_t row = indices_p[i];
            int64_t column = indices_p[i + nnz];
            scalar_t value = values_p[i];

            for (int64_t j = start; j < end; j++) {
                if (max_indices_p[row * dim + j] == column)
                    // TODO: CPU atomic addition?
                    input_grad_p[column * dim + j] += output_grad_p[row * dim + j] * value;
            }
        }
    });
}

/*
 * mask: m * n, nnz elements
 * input: n * d
 * max, max_indices: m * d
 * O(nnz * d)
 */
std::tuple<Tensor, Tensor> spmm_max_forward_cpu(const SparseTensor &mask, const Tensor &input_) {
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
    TORCH_CHECK(!mask.is_cuda(), "spmm_max_forward: expected `mask` to be CPU, but got CUDA");
    TORCH_CHECK(!input_.is_cuda(), "spmm_max_forward: expected `input` to be CPU, but got CUDA");

    const Tensor indices = mask.indices().contiguous();
    const Tensor values = mask.values().contiguous();
    const Tensor input = input_.contiguous();
    int64_t nnz = values.size(0);
    int64_t dim = input.numel() / input.size(0);

    std::vector<int64_t> sizes = input_.sizes().vec();
    sizes[0] = mask.size(0);
    Tensor max = at::full(sizes, -std::numeric_limits<float>::infinity(), values.options());
    Tensor max_indices = -at::ones(sizes, indices.options());

//    AT_DISPATCH_FLOATING_TYPES(input.type(), "spmm_max_forward", [&] {
//        spmm_max_forward_out_cpu<scalar_t>(
//            indices.data_ptr<int64_t>(),
//            values.data_ptr<scalar_t>(),
//            input.data_ptr<scalar_t>(),
//            max.data_ptr<scalar_t>(),
//            max_indices.data_ptr<int64_t>(),
//            nnz,
//            dim
//        );
//    });

    int64_t num_thread = std::thread::hardware_concurrency();
    num_thread = std::min(num_thread, dim);
    std::vector<std::thread> threads(num_thread);
    int64_t workload = (dim + num_thread - 1) / num_thread;
    TORCH_CHECK(input.type().scalarType() == at::ScalarType::Float, "Only implement for float32");
    for (int64_t i = 0; i < num_thread; i++) {
        int64_t start = workload * i;
        int64_t end = std::min(workload * (i + 1), dim);
        threads[i] = std::thread(spmm_max_forward_out_cpu_2<float_t>,
                          indices.data_ptr<int64_t>(),
                          values.data_ptr<float_t>(),
                          input.data_ptr<float_t>(),
                          max.data_ptr<float_t>(),
                          max_indices.data_ptr<int64_t>(),
                          nnz, dim,
                          start, end
                          );
    }
    for (auto &thread : threads)
        thread.join();

    return std::make_tuple(max, max_indices);
}

/*
 * mask: m * n, nnz elements
 * max_indices, output_grad: m * d
 * input_grad: n * d
 * O(nnz * d)
 */
std::tuple<Tensor, Tensor> spmm_max_backward_cpu(const SparseTensor &mask, const Tensor &max_indices_,
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
    TORCH_CHECK(!mask.is_cuda(), "spmm_max_backward: expected `mask` to be CPU, but got CUDA");
    TORCH_CHECK(!max_indices_.is_cuda(), "spmm_max_backward: expected `max_indices` to be CPU, but got CUDA");
    TORCH_CHECK(!output_grad_.is_cuda(), "spmm_max_backward: expected `output_grad` to be CPU, but got CUDA");

    const Tensor indices = mask.indices().contiguous();
    const Tensor values = mask.values().contiguous();
    const Tensor max_indices = max_indices_.contiguous();
    const Tensor output_grad = output_grad_.contiguous();
    int64_t nnz = values.size(0);
    int64_t dim = output_grad.numel() / output_grad.size(0);

    std::vector<int64_t> sizes = output_grad.sizes().vec();
    sizes[0] = mask.size(1);
    Tensor mask_grad = at::zeros({1}, values.options());
    Tensor input_grad = at::zeros(sizes, values.options());

    AT_DISPATCH_FLOATING_TYPES(input_grad.type(), "spmm_max_backward", [&] {
        spmm_max_backward_out_cpu<scalar_t>(
            indices.data_ptr<int64_t>(),
            values.data_ptr<scalar_t>(),
            max_indices.data_ptr<int64_t>(),
            output_grad.data_ptr<scalar_t>(),
            input_grad.data_ptr<scalar_t>(),
            nnz,
            dim
        );
    });

    return std::make_tuple(mask_grad, input_grad);
}

} // namespace at

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("spmm_max_cpu_forward", &at::spmm_max_forward_cpu, "sparse mask max cpu forward");
    m.def("spmm_max_cpu_backward", &at::spmm_max_backward_cpu, "sparse mask max cpu backward");
#ifdef CUDA_OP
    m.def("spmm_max_cuda_forward", &at::spmm_max_forward_cuda, "sparse mask max cuda forward");
    m.def("spmm_max_cuda_backward", &at::spmm_max_backward_cuda, "sparse mask max cuda backward");
#endif
}
