#pragma once

#include <deep_gemm/layout/mega_moe_rdma.cuh>

#if defined(__CUDACC__)
#define DG_RETURN_QUEUE_DEVICE __device__ __forceinline__
#else
#define DG_RETURN_QUEUE_DEVICE inline
#endif

namespace deep_gemm::comm {

// Append-only for one launch: every possible row has a slot in its shard.
// Producers never wait for network progress or recycle an in-flight entry.
// Atomic supplies GPU acquire/release operations, or CPU equivalents in tests.
template <uint32_t kSenders, typename Atomic>
struct MegaMoEReturnQueue {
    static_assert(kSenders > 0);
    const layout::MegaMoERdmaBuffer& staging;

    DG_RETURN_QUEUE_DEVICE void publish_tile(uint32_t row, uint32_t tiles) const {
        if (Atomic::add(staging.routes(0) + row, 1) != tiles - 1) return;
        const auto shard = row % kSenders;
        const auto ticket = Atomic::add(staging.return_queue_tail(kSenders, shard), 1);
        // Reservation is not publication: another producer may publish a later
        // ticket first. Zero entries must never be interpreted as valid rows.
        Atomic::store(staging.return_queue_entries(kSenders, shard) + ticket, row + 1);
    }

    DG_RETURN_QUEUE_DEVICE bool peek(uint32_t shard, uint32_t cursor, uint32_t& row) const {
        if (cursor >= Atomic::load(staging.return_queue_tail(kSenders, shard))) return false;
        const auto value = Atomic::load(staging.return_queue_entries(kSenders, shard) + cursor);
        if (value == 0) return false;
        row = value - 1;
        return true;
    }

    // Close only after the existing epilogue grid rendezvous joins producers.
    DG_RETURN_QUEUE_DEVICE void close() const { Atomic::store(staging.return_queue_closed(kSenders), 1); }

    DG_RETURN_QUEUE_DEVICE bool closed_and_empty(uint32_t shard, uint32_t cursor) const {
        // Acquire close BEFORE reading tail, so a stale tail cannot cause exit.
        return Atomic::load(staging.return_queue_closed(kSenders)) != 0 and
               cursor >= Atomic::load(staging.return_queue_tail(kSenders, shard));
    }

    // Called by one lane after all lanes have returned from their put calls.
    DG_RETURN_QUEUE_DEVICE void finish_sender() const { Atomic::add(staging.return_queue_done(kSenders), 1); }
    DG_RETURN_QUEUE_DEVICE bool all_senders_done() const {
        return Atomic::load(staging.return_queue_done(kSenders)) == kSenders;
    }
};

} // namespace deep_gemm::comm

#undef DG_RETURN_QUEUE_DEVICE
