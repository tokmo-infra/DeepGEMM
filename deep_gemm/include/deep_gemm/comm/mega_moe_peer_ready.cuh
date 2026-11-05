#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define DG_PEER_READY_DEVICE __device__ __forceinline__
#else
#define DG_PEER_READY_DEVICE inline
#endif

namespace deep_gemm::comm {

// Share one successful target-consistency probe across all combine warps.
// State is reset only after the previous launch has retired. A losing lane
// returns to the warp poll immediately; it must never spin inside divergence
// while the winning lane still needs to execute its probe.
template <typename Atomic>
struct MegaMoEPeerVisibility {
    template <typename Arrived, typename Probe>
    DG_PEER_READY_DEVICE static bool test(uint32_t* state, const Arrived& arrived, const Probe& probe) {
        if (Atomic::load(state) == 2) return true;
        // Observing the flag is only a cheap hint. It does not expose payloads.
        if (not arrived() or Atomic::compare_exchange(state, 0, 1) != 0) return false;
        const bool visible = probe();
        Atomic::store(state, visible ? 2 : 0);
        return visible;
    }
};

// One instance per lane, constructed afresh for each launch. Lane l owns peers
// l, l + 32, ...; successful probes are cached only for this launch's epoch.
// Probe must establish payload visibility, not merely observe a ready flag.
template <uint32_t kNumRanks>
struct MegaMoEPeerReadiness {
    static_assert(kNumRanks > 0 and kNumRanks <= 128);
    static constexpr uint32_t kGroups = (kNumRanks + 31) / 32;
    uint32_t visible_groups = 0;

    template <typename Probe>
    DG_PEER_READY_DEVICE bool poll(uint32_t lane, uint32_t group,
                                   uint32_t required_peers, const Probe& probe) {
        const auto peer = group * 32 + lane;
        if (peer >= kNumRanks or (required_peers & (1u << lane)) == 0)
            return true;
        const auto bit = 1u << group;
        if ((visible_groups & bit) == 0) {
            if (not probe(peer)) return false;
            visible_groups |= bit;
        }
        return true;
    }
};

} // namespace deep_gemm::comm

#undef DG_PEER_READY_DEVICE
