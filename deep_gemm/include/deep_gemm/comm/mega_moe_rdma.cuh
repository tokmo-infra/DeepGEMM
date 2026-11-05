#pragma once

#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/comm/mega_moe_peer_ready.cuh>
#include <deep_gemm/comm/mega_moe_return_queue.cuh>
#include <deep_gemm/layout/mega_moe_rdma.cuh>
#include <deep_gemm/ptx/utils.cuh>

#ifdef DG_MEGAMOE_NVSHMEM
#include <nvshmem.h>
#include <nvshmemx.h>
#endif

namespace deep_gemm::comm {

#ifdef DG_MEGAMOE_NVSHMEM
struct MegaMoEReturnQueueAtomic {
    CUTLASS_DEVICE static uint32_t load(const uint32_t* ptr) {
        uint32_t value;
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(value) : "l"(ptr) : "memory");
        return value;
    }
    CUTLASS_DEVICE static void store(uint32_t* ptr, uint32_t value) {
        asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(ptr), "r"(value) : "memory");
    }
    CUTLASS_DEVICE static uint32_t add(uint32_t* ptr, uint32_t value) {
        uint32_t old;
        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                     : "=r"(old) : "l"(ptr), "r"(value) : "memory");
        return old;
    }
    CUTLASS_DEVICE static uint32_t compare_exchange(uint32_t* ptr, uint32_t expected, uint32_t desired) {
        uint32_t old;
        asm volatile("atom.acq_rel.gpu.global.cas.b32 %0, [%1], %2, %3;"
                     : "=r"(old) : "l"(ptr), "r"(expected), "r"(desired) : "memory");
        return old;
    }
};
#endif

// Only instantiated in the opt-in SM100 FP8 x MXFP4 SwiGLU kernel.
template <uint32_t kNumRanks, uint32_t kReturnSenders = 0, bool kOverlap = false>
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

    template <uint32_t kThreads, typename Sync>
    CUTLASS_DEVICE void fetch_inputs(uint32_t sm, uint32_t thread, const Sync& sync) const {
#ifdef DG_MEGAMOE_NVSHMEM
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
                // Streaming/queued returns use dispatch metadata, not the
                // source top-k array. The post-grid scan fallback still needs it.
                if (not kOverlap or not streaming_returns())
                    pull(buffer.input_topk_idx_buffer);
                pull(buffer.input_topk_weights_buffer);
                nvshmem_quiet();
            }
        }
        // One cache per directly accessible team; different source PEs select
        // different owners. This also orders NIC writes before local peer reads.
        if (sm == 0) {
            sync();
            if (thread == 0 and nvshmem_team_n_pes(NVSHMEM_TEAM_SHARED) > 1 and
                nvshmem_team_n_pes(NVSHMEM_TEAM_SHARED) < kNumRanks)
                nvshmem_barrier(NVSHMEM_TEAM_SHARED);
            sync();
        }
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
            if constexpr (kOverlap) {
                // Use only reserved header space; neither native offsets nor
                // allocation ABI change. This reset precedes any combine work.
                DG_STATIC_ASSERT(kNumRanks * 12 <= 1016, "RDMA peer cache exceeds the staging header");
                if (sm == 0 and thread < kNumRanks)
                    *staging.peer_visible(thread) = 0;
            }
            // The dispatch barrier has consumed the route send slab. Reuse it
            // for row completion counters until the next launch's dispatch.
            if ((kOverlap or sm == 0) and streaming_returns()) {
                const auto count = queued_returns() ? staging.return_queue_words(kReturnSenders) : staging.return_rows();
                // The preceding grid joins every dispatch issuer's quiet, so
                // other CTAs can clear their source slab while CTA 0 is in the
                // world barrier. The final grid joins all clears before returns.
                const uint64_t first = kOverlap ? sm * kThreads + thread : thread;
                const uint64_t stride = kOverlap ? kNumSMs * kThreads : kThreads;
                for (uint64_t i = first; i < count; i += stride)
                    staging.routes(0)[i] = 0;
            }
            if constexpr (not kOverlap)
                fetch_inputs<kThreads>(sm, thread, sync);
            if (sm == 0 and thread < kNumRanks and remote(thread)) {
                // Independent source slots avoid mixed NIC/GPU AMOs. Preserve
                // the scheduler's kNumSMs * kNumRanks contribution count. With
                // overlap, publish metadata before fetching activation payloads;
                // activation TMA still waits on the original l1_full_count.
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

    template <uint32_t kNumSMs, uint32_t kThreads, uint32_t kSync, typename Sync>
    CUTLASS_DEVICE void prepare_inputs(const layout::Workspace& ws, uint32_t sm,
                                       uint32_t thread, const Sync& sync) const {
#ifdef DG_MEGAMOE_NVSHMEM
        if constexpr (kOverlap) {
            // Called AFTER the dispatch/epilogue rendezvous: shared-expert
            // epilogues and routed weight preparation may now run during GETs.
            // Only dispatch warps wait here; do not release incomplete ring rows.
            fetch_inputs<kThreads>(sm, thread, sync);
            grid_sync<kNumSMs, kSync>(ws, sm, thread, sync);
            asm volatile("fence.proxy.async.global;" ::: "memory");
        }
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

    CUTLASS_DEVICE bool queued_returns() const {
#ifdef DG_MEGAMOE_NVSHMEM
        if constexpr (kReturnSenders > 0)
            return staging.can_queue_returns(kReturnSenders);
#endif
        return false;
    }

    CUTLASS_DEVICE void return_tile(uint32_t peer, uint32_t token, uint32_t slot,
                                    uint32_t lane, uint32_t num_tiles) const {
#ifdef DG_MEGAMOE_NVSHMEM
        if (not streaming_returns() or not remote(peer)) return;
        // Each half warp stores one row fragment. Release/acquire RMWs join
        // fragments produced by different CTAs. In queued mode, the last one
        // publishes a descriptor; only retired dispatch warps call NVSHMEM.
        __syncwarp(lane < 16 ? 0x0000ffff : 0xffff0000);
        if (lane % 16 == 0) {
            if constexpr (kReturnSenders > 0) {
                if (queued_returns()) {
                    const auto row = (peer * staging.topk + slot) * staging.max_tokens + token;
                    MegaMoEReturnQueue<kReturnSenders, MegaMoEReturnQueueAtomic>{staging}.publish_tile(row, num_tiles);
                    return;
                }
            }
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

    CUTLASS_DEVICE void progress_returns(uint32_t sm, uint32_t warp, uint32_t lane) const {
#ifdef DG_MEGAMOE_NVSHMEM
        if constexpr (kReturnSenders > 0) {
            if (not queued_returns() or sm >= kReturnSenders or warp != 0) return;
            const auto queue = MegaMoEReturnQueue<kReturnSenders, MegaMoEReturnQueueAtomic>{staging};
            uint32_t cursor = lane;
            // All lanes participate, including those with no rows. Later lanes
            // can submit while an earlier producer's reserved slot is unready.
            wait_until([&]() {
                uint32_t row;
                const bool submitted = queue.peek(sm, cursor, row);
                if (submitted) {
                    const auto token = row % staging.max_tokens;
                    const auto slot = row / staging.max_tokens % staging.topk;
                    const auto peer = row / (staging.max_tokens * staging.topk);
                    auto* dst = buffer.combine_token_buffer.get_rank_buffer(slot).get_data_buffer(token).get_base_ptr();
                    nvshmem_putmem_nbi(dst, output(dst, peer), staging.hidden * 2, peer);
                    cursor += 32;
                }
                const bool done = __all_sync(0xffffffff, queue.closed_and_empty(sm, cursor));
                if (not __any_sync(0xffffffff, submitted) and not done) __nanosleep(64);
                return done;
            }, [&]() { printf("DeepGEMM RDMA sender timeout: rank=%u shard=%u lane=%u cursor=%u\n",
                              sym.rank_idx, sm, lane, cursor); });
            __syncwarp();
            if (lane == 0) queue.finish_sender();
        }
#endif
    }

    template <uint32_t kThreads, typename Sync>
    CUTLASS_DEVICE void return_and_signal(uint32_t thread, uint64_t epoch, const Sync& sync) const {
#ifdef DG_MEGAMOE_NVSHMEM
        if constexpr (kOverlap) {
            // The existing epilogue grid already joined direct stores. Local
            // destinations need neither the remote sender join nor NIC quiet.
            if (thread < kNumRanks and not remote(thread))
                ptx::st_rel_sys(sym.map(buffer.workspace.get_combine_ready_grid_idx_ptr(sym.rank_idx), thread), epoch);
        }
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
        if constexpr (kReturnSenders > 0) {
            if (queued_returns() and thread == 0) {
                const auto queue = MegaMoEReturnQueue<kReturnSenders, MegaMoEReturnQueueAtomic>{staging};
                queue.close();
                wait_until([&]() { return queue.all_senders_done(); }, [&]() {
                    printf("DeepGEMM RDMA sender join timeout: rank=%u\n", sym.rank_idx);
                });
            }
        }
        // The epilogue grid sync joins producers. Queued mode additionally
        // joins all senders before quiet can observe their submissions.
        // One elected warp can complete the PE's operations;
        // a quiet in every issuing warp would repeatedly drain the same QPs.
        sync();
        if (thread < 32)
            nvshmemx_quiet_warp();
        sync();
        if (thread < kNumRanks and (not kOverlap or remote(thread))) {
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

    CUTLASS_DEVICE void wait_selected_returns(uint32_t peer, uint32_t lane, uint64_t epoch,
                                              MegaMoEPeerReadiness<kNumRanks>& cache) const {
#ifdef DG_MEGAMOE_NVSHMEM
        // All lanes participate; kNumRanks is the sentinel for masked/shared
        // slots. Each lane probes its own peers, once successfully per launch,
        // instead of repeating target consistency for every token/chunk.
        constexpr auto kGroups = MegaMoEPeerReadiness<kNumRanks>::kGroups;
        uint32_t required[kGroups];
        #pragma unroll
        for (uint32_t group = 0; group < kGroups; ++group)
            required[group] = __reduce_or_sync(0xffffffff,
                peer < kNumRanks and peer / 32 == group ? 1u << (peer % 32) : 0u);
        wait_until([&]() {
            bool complete = true;
            #pragma unroll
            for (uint32_t group = 0; group < kGroups; ++group) {
                const bool done = cache.poll(lane, group, required[group], [&](uint32_t source) {
                    auto* ptr = buffer.workspace.get_combine_ready_grid_idx_ptr(source);
                    // A plain acquire is sufficient only for mapped GPU writes.
                    // Remote success must include NVSHMEM target consistency.
                    if (not remote(source)) return ptx::ld_acq_sys(ptr) == epoch;
                    return MegaMoEPeerVisibility<MegaMoEReturnQueueAtomic>::test(
                        staging.peer_visible(source),
                        [&]() { return ptx::ld_acq_sys(ptr) == epoch; },
                        [&]() { return nvshmem_uint64_test(ptr, NVSHMEM_CMP_EQ, epoch) != 0; });
                });
                complete = complete and done;
            }
            return __all_sync(0xffffffff, complete);
        }, [&]() { printf("DeepGEMM RDMA selected-peer timeout: rank=%u peer=%u\n", sym.rank_idx, peer); });
        // Publish each successful lane's acquire to the warp before its TMA
        // issuer executes the caller's async-proxy fence and combine loads.
        __syncwarp();
#endif
    }

    template <uint32_t kNumSMs, uint32_t kSync, typename Sync>
    CUTLASS_DEVICE void finish_returns(const layout::Workspace& ws, uint32_t sm,
                                      uint32_t thread, const Sync& sync) const {
#ifdef DG_MEGAMOE_NVSHMEM
        if constexpr (kOverlap) return;
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
