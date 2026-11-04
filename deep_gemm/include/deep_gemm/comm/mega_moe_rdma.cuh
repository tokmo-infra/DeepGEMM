#pragma once

#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/layout/mega_moe_rdma.cuh>
#include <deep_gemm/ptx/utils.cuh>

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

    CUTLASS_DEVICE void begin_launch() const {
#ifdef DG_MEGAMOE_NVSHMEM
        // CUDA Graph replays can reuse %gridid. Only one thread advances this
        // process-lifetime buffer counter, before the dispatch grid barrier.
        ++*staging.launch_epoch();
        DG_DEVICE_ASSERT(*staging.launch_epoch() != 0);
#endif
    }

    CUTLASS_DEVICE uint64_t epoch() const {
#ifdef DG_MEGAMOE_NVSHMEM
        return *staging.launch_epoch();
#else
        return ptx::get_grid_idx() + 1;
#endif
    }

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
        // Only CTA 0 issues dispatch RMA. Other CTAs must not flood the
        // transport with quiet requests. Return issuers drain before the
        // dispatch/epilogue rendezvous, so cleanup has no pending dispatch RMA.
        if constexpr (kTag == 1) {
            if (sm == 0)
                nvshmemx_quiet_warp();
        }
        grid_sync<kNumSMs, kSync>(ws, sm, thread, sync);
        if (sm == 0 and thread == 0)
            nvshmem_barrier_all();
        // Preserve the capacity-based offsets, but transfer only each peer's
        // live token prefixes. The barrier publishes peer_tokens with the data.
        if constexpr (kTag == 1) {
            sync();
            // The dispatch barrier has consumed the route send slab. Reuse it
            // for row completion counters until the next launch's dispatch.
            if (sm == 0 and streaming_returns()) {
                const auto count = kNumRanks * staging.topk * staging.max_tokens;
                for (uint64_t i = thread; i < count; i += kThreads)
                    staging.routes(0)[i] = 0;
            }
            if (sm == 0 and thread < kNumRanks and remote(thread)) {
                const auto tokens = *staging.peer_tokens(thread);
                DG_DEVICE_ASSERT(tokens <= staging.max_tokens);
                if (input_proxy(thread) == sym.rank_idx and tokens != 0) {
                    const auto pull = [&](const layout::Buffer& part) {
                        auto* src = part.template get_base_ptr<uint8_t>();
                        nvshmem_getmem_nbi(input(src, thread), src,
                                          tokens * part.data_layout.template get_num_bytes<uint64_t>(), thread);
                    };
                    pull(buffer.input_token_buffer);
                    pull(buffer.input_sf_buffer);
                    pull(buffer.input_topk_idx_buffer);
                    pull(buffer.input_topk_weights_buffer);
                    nvshmem_quiet();
                }
            }
            // One cache per directly accessible team; different source PEs
            // select different cache owners to keep all local NICs active.
            if (sm == 0) {
                sync();
                if (thread == 0 and nvshmem_team_n_pes(NVSHMEM_TEAM_SHARED) > 1 and
                    nvshmem_team_n_pes(NVSHMEM_TEAM_SHARED) < kNumRanks)
                    nvshmem_barrier(NVSHMEM_TEAM_SHARED);
                sync();
            }
            if (sm == 0 and thread < kNumRanks and remote(thread)) {
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

    CUTLASS_DEVICE uint32_t input_proxy(uint32_t peer) const {
#ifdef DG_MEGAMOE_NVSHMEM
        const auto peers = nvshmem_team_n_pes(NVSHMEM_TEAM_SHARED);
        return nvshmem_team_translate_pe(NVSHMEM_TEAM_SHARED, peer % peers, NVSHMEM_TEAM_WORLD);
#else
        return sym.rank_idx;
#endif
    }

    template <typename Ptr>
    CUTLASS_DEVICE Ptr input(Ptr ptr, uint32_t peer) const {
        if (not remote(peer))
            return sym.map(ptr, peer);
        const auto offset = reinterpret_cast<const uint8_t*>(ptr) - buffer.input_token_buffer.template get_base_ptr<uint8_t>();
#ifdef DG_MEGAMOE_NVSHMEM
        auto* base = static_cast<uint8_t*>(nvshmem_ptr(staging.inputs(peer), input_proxy(peer)));
        DG_DEVICE_ASSERT(base != nullptr);
        return reinterpret_cast<Ptr>(base + offset);
#else
        return nullptr;
#endif
    }

    template <typename Ptr>
    CUTLASS_DEVICE Ptr output(Ptr ptr, uint32_t peer) const {
        if (not remote(peer))
            return sym.map(ptr, peer);
        const auto offset = reinterpret_cast<const uint8_t*>(ptr) - buffer.combine_token_buffer.template get_base_ptr<uint8_t>();
        return reinterpret_cast<Ptr>(static_cast<uint8_t*>(staging.outputs(peer)) + offset);
    }

    CUTLASS_DEVICE bool streaming_returns() const {
#ifdef DG_MEGAMOE_NVSHMEM
        return staging.can_stream_returns();
#else
        return false;
#endif
    }

    CUTLASS_DEVICE void return_tile(uint32_t peer, uint32_t token, uint32_t slot,
                                    uint32_t lane, uint32_t num_tiles) const {
#ifdef DG_MEGAMOE_NVSHMEM
        if (not streaming_returns() or not remote(peer)) return;
        // Each half warp stores one row fragment. Release/acquire RMWs join
        // fragments produced by different CTAs; the last one sends the row
        // while the original MegaMoE pipeline continues computing other tiles.
        __syncwarp(lane < 16 ? 0x0000ffff : 0xffff0000);
        if (lane % 16 == 0) {
            auto* counter = staging.return_counter(peer, slot, token);
            uint32_t old;
            asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;"
                         : "=r"(old) : "l"(counter) : "memory");
            if (old == num_tiles - 1) {
                auto* dst = buffer.combine_token_buffer.get_rank_buffer(slot).get_data_buffer(token).get_base_ptr();
                nvshmem_putmem_nbi(dst, output(dst, peer), staging.hidden * 2, peer);
            }
        }
#endif
    }

    template <uint32_t kThreads, typename Sync>
    CUTLASS_DEVICE void return_and_signal(uint32_t thread, uint64_t epoch, const Sync& sync) const {
#ifdef DG_MEGAMOE_NVSHMEM
        // Spread independent return rows across the epilogue CTA instead of
        // serializing every row for a destination in one GPU lane.
        for (uint32_t peer = 0; not streaming_returns() and peer < kNumRanks; ++peer) {
            if (not remote(peer)) continue;
            const auto* indices = input(buffer.input_topk_idx_buffer.template get_base_ptr<int64_t>(), peer);
            const auto tokens = *staging.peer_tokens(peer);
            DG_DEVICE_ASSERT(tokens <= staging.max_tokens);
            for (uint32_t row = thread; row < tokens * staging.topk; row += kThreads) {
                const auto slot = row / tokens, token = row % tokens;
                const auto expert = indices[token * staging.topk + slot];
                if (expert >= 0 and expert / buffer.workspace.num_experts_per_rank == sym.rank_idx) {
                    auto* dst = buffer.combine_token_buffer.get_rank_buffer(slot).get_data_buffer(token).get_base_ptr();
                    nvshmem_putmem_nbi(dst, output(dst, peer), staging.hidden * 2, peer);
                }
            }
        }
        // The existing epilogue grid sync orders all streaming submissions
        // before this CTA. One elected warp can complete the PE's operations;
        // a quiet in every issuing warp would repeatedly drain the same QPs.
        sync();
        if (thread < 32)
            nvshmemx_quiet_warp();
        sync();
        if (thread < kNumRanks) {
            auto* ready = buffer.workspace.get_combine_ready_grid_idx_ptr(sym.rank_idx);
            nvshmemx_signal_op(ready, epoch, NVSHMEM_SIGNAL_SET, thread);
        }
        sync();
        if (thread < 32)
            nvshmemx_quiet_warp();
#else
        if (thread < kNumRanks) {
            auto* ready = buffer.workspace.get_combine_ready_grid_idx_ptr(sym.rank_idx);
            ptx::st_rel_sys(sym.map(ready, thread), epoch);
        }
#endif
    }

    CUTLASS_DEVICE bool ready(uint64_t* ptr, uint64_t epoch) const {
        // RDMA target consistency is established once by finish_returns, then
        // distributed by the epilogue grid sync before any combine loads.
        return ptx::ld_acq_sys(ptr) == epoch;
    }

    template <uint32_t kNumSMs, uint32_t kSync, typename Sync>
    CUTLASS_DEVICE void finish_returns(const layout::Workspace& ws, uint32_t sm,
                                      uint32_t thread, const Sync& sync) const {
#ifdef DG_MEGAMOE_NVSHMEM
        // Each peer signals even when it has no output. A single test_all orders
        // the NIC writes before the grid release/acquire and subsequent TMA
        // fences. Per-token tests would repeat the costly target-consistency op.
        if (sm == 0 and thread == 0)
            wait_until([&]() {
                return nvshmem_uint64_test_all(ws.get_combine_ready_grid_idx_ptr(),
                                              kNumRanks, nullptr, NVSHMEM_CMP_EQ, epoch());
            }, [&]() { printf("DeepGEMM RDMA combine timeout: rank=%u\n", sym.rank_idx); });
        grid_sync<kNumSMs, kSync>(ws, sm, thread, sync);
#endif
    }
};

} // namespace deep_gemm::comm
