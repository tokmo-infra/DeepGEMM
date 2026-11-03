#pragma once

#include "../runtime/mega_moe_rdma.hpp"

namespace deep_gemm::rdma {
inline int64_t buffer_size(int64_t ranks, int64_t experts, int64_t tokens, int64_t hidden, int64_t topk) {
    DG_HOST_ASSERT(ranks > 0 and ranks <= layout::kNumMaxRanks);
    DG_HOST_ASSERT(experts > 0 and experts <= layout::MegaMoESignals::kNumMaxExperts and experts % ranks == 0);
    DG_HOST_ASSERT(tokens > 0 and tokens <= (1 << 20) and tokens % 4 == 0);
    DG_HOST_ASSERT(hidden > 0 and hidden <= (1 << 20) and hidden % 128 == 0);
    DG_HOST_ASSERT(topk > 0 and topk <= 32);
    return layout::MegaMoERdmaBuffer(nullptr, ranks, experts, tokens, hidden, topk).get_num_bytes();
}
}

STABLE_TORCH_LIBRARY_FRAGMENT(deep_gemm, m) {
    m.def("mega_moe_rdma_abi_version() -> int");
    m.def("mega_moe_rdma_buffer_size(int ranks, int experts, int tokens, int hidden, int topk) -> int");
#ifdef DG_ENABLE_NVSHMEM
    m.def("mega_moe_rdma_unique_id() -> int[]");
    m.def("mega_moe_rdma_init(int[] uid, int rank, int size) -> ()");
    m.def("mega_moe_rdma_allocate(int bytes) -> (Tensor, int[])");
#endif
}
STABLE_TORCH_LIBRARY_IMPL(deep_gemm, CompositeExplicitAutograd, m) {
    m.impl("mega_moe_rdma_abi_version", TORCH_BOX(&deep_gemm::rdma::abi_version));
    m.impl("mega_moe_rdma_buffer_size", TORCH_BOX(&deep_gemm::rdma::buffer_size));
#ifdef DG_ENABLE_NVSHMEM
    m.impl("mega_moe_rdma_unique_id", TORCH_BOX(&deep_gemm::rdma::unique_id));
    m.impl("mega_moe_rdma_init", TORCH_BOX(&deep_gemm::rdma::init));
    m.impl("mega_moe_rdma_allocate", TORCH_BOX(&deep_gemm::rdma::allocate));
#endif
}
