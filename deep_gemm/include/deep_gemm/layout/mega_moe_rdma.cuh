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
    DG_RDMA_HD uint32_t* peer_visible(uint32_t peer) const {
        // Per-launch GPU visibility cache, disjoint from counts and epoch.
        // At the supported maximum EP72 its end is byte 864 (< epoch at 1016).
        return reinterpret_cast<uint32_t*>(peer_tokens(num_ranks)) + peer;
    }
    DG_RDMA_HD uint64_t* launch_epoch() const {
        return reinterpret_cast<uint64_t*>(base + route_offset() - sizeof(uint64_t));
    }
    DG_RDMA_HD uint32_t* routes(uint32_t expert) const {
        return reinterpret_cast<uint32_t*>(base + route_offset()) + expert * max_tokens;
    }
    // The dispatch route slab is dead after its collective completion. Reuse
    // it for return-row tile counts only when every possible row fits.
    DG_RDMA_HD bool can_stream_returns() const { return num_ranks * topk <= num_experts; }
    DG_RDMA_HD uint32_t* return_counter(uint32_t peer, uint32_t slot, uint32_t token) const {
        return routes(0) + (peer * topk + slot) * max_tokens + token;
    }
    DG_RDMA_HD uint64_t return_rows() const { return num_ranks * topk * max_tokens; }
    DG_RDMA_HD uint64_t return_queue_capacity(uint32_t senders) const {
        return (return_rows() + senders - 1) / senders;
    }
    DG_RDMA_HD uint64_t return_queue_words(uint32_t senders) const {
        return return_rows() + senders * return_queue_capacity(senders) + senders + 2;
    }
    DG_RDMA_HD bool can_queue_returns(uint32_t senders) const {
        return senders != 0 and return_queue_words(senders) <= num_experts * max_tokens;
    }
    DG_RDMA_HD uint32_t* return_queue_entries(uint32_t senders, uint32_t shard) const {
        return routes(0) + return_rows() + shard * return_queue_capacity(senders);
    }
    DG_RDMA_HD uint32_t* return_queue_tail(uint32_t senders, uint32_t shard) const {
        return return_queue_entries(senders, senders) + shard;
    }
    DG_RDMA_HD uint32_t* return_queue_closed(uint32_t senders) const {
        return return_queue_tail(senders, senders);
    }
    DG_RDMA_HD uint32_t* return_queue_done(uint32_t senders) const {
        return return_queue_closed(senders) + 1;
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
