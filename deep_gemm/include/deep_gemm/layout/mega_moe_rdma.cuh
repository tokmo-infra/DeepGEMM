#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define DG_RDMA_HD __host__ __device__
#else
#define DG_RDMA_HD
#endif

namespace deep_gemm::layout {

// Separate allocation: native MegaMoE's workspace and tensor offsets stay unchanged.
struct MegaMoERdmaBuffer {
    uint8_t* base;
    uint64_t num_ranks, num_experts, max_tokens, hidden, topk;

    DG_RDMA_HD
    MegaMoERdmaBuffer(void* base, uint64_t ranks, uint64_t experts,
                     uint64_t tokens, uint64_t hidden, uint64_t topk):
        base(static_cast<uint8_t*>(base)), num_ranks(ranks), num_experts(experts),
        max_tokens(tokens), hidden(hidden), topk(topk) {}

    DG_RDMA_HD uint64_t input_bytes() const {
        return max_tokens * (hidden + hidden / 32 + topk * 12);
    }
    DG_RDMA_HD uint64_t output_bytes() const {
        return topk * max_tokens * hidden * 2;
    }
    DG_RDMA_HD uint64_t route_offset() const { return 1024; }
    DG_RDMA_HD uint64_t input_offset() const {
        return ((route_offset() + num_experts * max_tokens * 4 + 127) / 128 * 128);
    }
    DG_RDMA_HD uint64_t output_offset() const {
        return ((input_offset() + num_ranks * input_bytes() + 127) / 128 * 128);
    }
    DG_RDMA_HD uint64_t get_num_bytes() const {
        return output_offset() + num_ranks * output_bytes();
    }
    DG_RDMA_HD uint64_t* peer_tokens(uint32_t peer) const {
        return reinterpret_cast<uint64_t*>(base) + peer;
    }
    DG_RDMA_HD uint32_t* routes(uint32_t expert) const {
        return reinterpret_cast<uint32_t*>(base + route_offset()) + expert * max_tokens;
    }
    DG_RDMA_HD void* inputs(uint32_t peer) const {
        return base + input_offset() + peer * input_bytes();
    }
    DG_RDMA_HD void* outputs(uint32_t peer) const {
        return base + output_offset() + peer * output_bytes();
    }
};

} // namespace deep_gemm::layout

#undef DG_RDMA_HD
