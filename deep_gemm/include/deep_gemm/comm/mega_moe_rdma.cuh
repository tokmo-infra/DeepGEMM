#pragma once

#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/layout/mega_moe_rdma.cuh>

#ifdef DG_MEGAMOE_NVSHMEM
#include <nvshmem.h>
#include <nvshmemx.h>
#endif

namespace deep_gemm::comm {

// Only instantiated in the opt-in SM100 FP8 x MXFP4 SwiGLU kernel.
template <uint32_t kNumRanks>
struct MegaMoETransport {
    const layout::SymBuffer<kNumRanks>& sym;
    const layout::MegaMoEBuffer& buffer;
    layout::MegaMoERdmaBuffer staging;

    CUTLASS_DEVICE bool remote(uint32_t peer) const {
#ifdef DG_MEGAMOE_NVSHMEM
        return not sym.is_mapped(peer);
#else
        return false;
#endif
    }

    CUTLASS_DEVICE void route(uint32_t expert, uint32_t slot, uint32_t token_topk) const {
        const auto local_experts = buffer.workspace.num_experts_per_rank;
        const auto peer = expert / local_experts;
        auto* dst = buffer.workspace.get_src_token_topk_idx_ptr(expert % local_experts, sym.rank_idx, slot);
        if (remote(peer))
            staging.routes(expert)[slot] = token_topk;
        else
            *sym.map(dst, peer) = token_topk;
    }

    CUTLASS_DEVICE void publish_epoch(uint32_t peer, uint64_t epoch, uint32_t tokens) const {
        auto* dst = buffer.workspace.get_peer_grid_idx_ptr(sym.rank_idx);
#ifdef DG_MEGAMOE_NVSHMEM
        // Token counts also describe the valid prefix of the return routing table.
        nvshmem_uint64_p(staging.peer_tokens(sym.rank_idx), tokens, peer);
        nvshmem_uint64_p(dst, epoch, peer);
#else
        *sym.map(dst, peer) = epoch;
#endif
    }

    CUTLASS_DEVICE void publish_count(uint32_t expert, uint64_t status) const {
        const auto& ws = buffer.workspace;
        const auto peer = expert / ws.num_experts_per_rank;
        const auto local_expert = expert % ws.num_experts_per_rank;
        auto* recv = ws.get_expert_recv_count_ptr(sym.rank_idx, local_expert);
        auto* sum = ws.get_expert_recv_count_sum_ptr(local_expert);
#ifdef DG_MEGAMOE_NVSHMEM
        if (remote(peer)) {
            const auto count = static_cast<uint32_t>(status);
            if (count != 0)
                nvshmem_putmem_nbi(ws.get_src_token_topk_idx_ptr(local_expert, sym.rank_idx),
                                  staging.routes(expert), count * sizeof(uint32_t), peer);
            nvshmem_uint64_p(recv, count, peer);
        } else {
            *sym.map(recv, peer) = status & 0xffffffff;
            ptx::atomic_add_sys(sym.map(sum, peer), status);
        }
#else
        *sym.map(recv, peer) = status & 0xffffffff;
        ptx::atomic_add_sys(sym.map(sum, peer), status);
#endif
    }

    template <uint32_t kNumSMs, uint32_t kThreads, uint32_t kSync, uint32_t kTag, typename Sync>
    CUTLASS_DEVICE void barrier(const layout::Workspace& ws, uint32_t sm, uint32_t thread,
                                const Sync& sync, bool prologue, bool epilogue) const {
#ifdef DG_MEGAMOE_NVSHMEM
        // Every issuer drains its queue before the single collective participant.
        nvshmem_quiet();
        grid_sync<kNumSMs, kSync>(ws, sm, thread, sync);
        if (sm == 0 and thread == 0)
            nvshmem_barrier_all();
        // The dispatch barrier additionally prefetches complete peer input slabs.
        if constexpr (kTag == 1) {
            sync();
            if (sm == 0 and thread < kNumRanks and remote(thread)) {
                nvshmem_getmem(staging.inputs(thread), buffer.input_token_buffer.get_base_ptr(),
                               staging.input_bytes(), thread);
                nvshmem_quiet();
                // Independent source slots avoid mixed NIC/GPU AMOs. Preserve
                // the scheduler's kNumSMs * kNumRanks contribution count.
                for (uint32_t e = 0; e < ws.num_experts_per_rank; ++e)
                    ptx::atomic_add_sys(ws.get_expert_recv_count_sum_ptr(e),
                        (static_cast<uint64_t>(kNumSMs) << 32) | *ws.get_expert_recv_count_ptr(thread, e));
            }
        }
        if (epilogue)
            grid_sync<kNumSMs, kSync>(ws, sm, thread, sync);
#else
        nvlink_barrier<kNumRanks, kNumSMs, kThreads, kSync, kTag>(
            ws, sym, sm, thread, sync, prologue, epilogue);
#endif
    }

    template <typename Ptr>
    CUTLASS_DEVICE Ptr input(Ptr ptr, uint32_t peer) const {
        if (not remote(peer))
            return sym.map(ptr, peer);
        const auto offset = reinterpret_cast<const uint8_t*>(ptr) - buffer.input_token_buffer.template get_base_ptr<uint8_t>();
        return reinterpret_cast<Ptr>(static_cast<uint8_t*>(staging.inputs(peer)) + offset);
    }

    template <typename Ptr>
    CUTLASS_DEVICE Ptr output(Ptr ptr, uint32_t peer) const {
        if (not remote(peer))
            return sym.map(ptr, peer);
        const auto offset = reinterpret_cast<const uint8_t*>(ptr) - buffer.combine_token_buffer.template get_base_ptr<uint8_t>();
        return reinterpret_cast<Ptr>(static_cast<uint8_t*>(staging.outputs(peer)) + offset);
    }

    CUTLASS_DEVICE void return_and_signal(uint32_t peer, uint64_t epoch) const {
        auto* ready = buffer.workspace.get_combine_ready_grid_idx_ptr(sym.rank_idx);
#ifdef DG_MEGAMOE_NVSHMEM
        if (remote(peer)) {
            const auto* indices = input(buffer.input_topk_idx_buffer.template get_base_ptr<int64_t>(), peer);
            const auto tokens = *staging.peer_tokens(peer);
            DG_DEVICE_ASSERT(tokens <= staging.max_tokens);
            // Combine layout is [top-k][token][hidden]. Coalesce consecutive
            // tokens owned by this expert rank, never overwrite another rank's slots.
            for (uint32_t slot = 0; slot < staging.topk; ++slot) {
                uint32_t token = 0;
                while (token < tokens) {
                    const auto owned = [&](uint32_t t) {
                        const auto expert = indices[t * staging.topk + slot];
                        return expert >= 0 and expert / buffer.workspace.num_experts_per_rank == sym.rank_idx;
                    };
                    if (not owned(token)) { ++token; continue; }
                    const auto begin = token++;
                    while (token < tokens and owned(token)) ++token;
                    auto* dst = buffer.combine_token_buffer.get_rank_buffer(slot).get_data_buffer(begin).get_base_ptr();
                    nvshmem_putmem_nbi(dst, output(dst, peer), (token - begin) * staging.hidden * 2, peer);
                }
            }
        }
        // Called after the epilogue grid sync: all CTA stores precede this drain.
        nvshmem_quiet();
        nvshmemx_signal_op(ready, epoch, NVSHMEM_SIGNAL_SET, peer);
        nvshmem_quiet();
#else
        ptx::st_rel_sys(sym.map(ready, peer), epoch);
#endif
    }

    CUTLASS_DEVICE bool ready(uint64_t* ptr, uint64_t epoch) const {
#ifdef DG_MEGAMOE_NVSHMEM
        // NVSHMEM test orders target reads, unlike a raw acquire of a NIC-written flag.
        return nvshmem_uint64_test(ptr, NVSHMEM_CMP_EQ, epoch);
#else
        return ptx::ld_acq_sys(ptr) == epoch;
#endif
    }
};

} // namespace deep_gemm::comm
