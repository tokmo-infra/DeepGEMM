// CPU coverage of the actual per-lane cache. This does not emulate NIC/GPU
// memory ordering; the GPU matrix must validate the NVSHMEM probe and warp join.
#include <deep_gemm/comm/mega_moe_peer_ready.cuh>

#include <array>
#include <atomic>
#include <cassert>
#include <cstdlib>
#include <random>
#include <thread>
#include <vector>

struct Atomic {
    static uint32_t load(const uint32_t* ptr) {
        return std::atomic_ref(*const_cast<uint32_t*>(ptr)).load(std::memory_order_acquire);
    }
    static void store(uint32_t* ptr, uint32_t value) {
        std::atomic_ref(*ptr).store(value, std::memory_order_release);
    }
    static uint32_t compare_exchange(uint32_t* ptr, uint32_t expected, uint32_t desired) {
        std::atomic_ref(*ptr).compare_exchange_strong(expected, desired, std::memory_order_acq_rel);
        return expected;
    }
};

void shared_visibility() {
    using Visibility = deep_gemm::comm::MegaMoEPeerVisibility<Atomic>;
    uint32_t state = 0;
    assert(not Visibility::test(&state, [] { return false; }, [] { std::abort(); return true; }));
    // A failed confirmation must release ownership for another probe. A loser
    // must return, even if the owner is suspended inside its confirm callback.
    assert(not Visibility::test(&state, [] { return true; }, [&] {
        assert(not Visibility::test(&state, [] { return true; }, [] { std::abort(); return true; }));
        return false;
    }));
    assert(state == 0);

    std::array<uint32_t, 24> states{};
    std::array<uint32_t, 24> visible_payload{};
    std::array<std::atomic<uint32_t>, 24> successes{};
    for (uint32_t epoch = 1; epoch <= 12; ++epoch) {
        // Retire all consumers before resetting; leave old payload in place.
        states.fill(0);
        for (auto& count : successes) count = 0;
        std::atomic<bool> arrived = false;
        std::vector<std::thread> threads;
        for (uint32_t worker = 0; worker < 8; ++worker) {
            threads.emplace_back([&, worker] {
                for (uint32_t i = 0; i < states.size(); ++i) {
                    const auto peer = (i + worker) % states.size();
                    while (not Visibility::test(&states[peer],
                        [&] { return arrived.load(std::memory_order_relaxed); }, [&] {
                            if (not arrived.load(std::memory_order_acquire)) return false;
                            std::this_thread::yield();
                            visible_payload[peer] = epoch * 1000 + peer;
                            ++successes[peer];
                            return true;
                        })) std::this_thread::yield();
                    // Successful publication must carry the owner's writes to
                    // every consumer, including those that never ran a probe.
                    assert(visible_payload[peer] == epoch * 1000 + peer);
                }
            });
        }
        arrived.store(true, std::memory_order_release);
        for (auto& thread : threads) thread.join();
        for (auto& count : successes) assert(count == 1);
    }
}

template <uint32_t kRanks>
void run() {
    using Cache = deep_gemm::comm::MegaMoEPeerReadiness<kRanks>;
    std::array<uint64_t, kRanks> arrived{};
    std::array<uint64_t, kRanks> payload{};
    std::mt19937 random(20261105 + kRanks);
    for (uint64_t epoch = 1; epoch <= 20; ++epoch) {
        // Cache lifetime must end at each kernel boundary, even when buffers and
        // route patterns repeat. Old flags/payload remain in place deliberately.
        std::array<Cache, 32> lanes{};
        std::array<uint64_t, kRanks> visible{};
        std::array<uint32_t, kRanks> successful_probes{};
        std::array<uint32_t, kRanks> probes{};
        auto poll = [&](const std::array<bool, kRanks>& needed) {
            bool complete = true;
            for (uint32_t group = 0; group < Cache::kGroups; ++group) {
                uint32_t mask = 0;
                for (uint32_t peer = group * 32; peer < kRanks and peer < (group + 1) * 32; ++peer)
                    if (needed[peer]) mask |= 1u << (peer % 32);
                for (uint32_t lane = 0; lane < 32; ++lane) {
                    const bool done = lanes[lane].poll(lane, group, mask, [&](uint32_t peer) {
                        ++probes[peer];
                        if (arrived[peer] != epoch) return false;
                        ++successful_probes[peer];
                        // Model the probe's visibility contract separately from
                        // its flag: only a successful probe exposes the payload.
                        visible[peer] = payload[peer];
                        return true;
                    });
                    complete = complete and done;
                }
            }
            bool expected = true;
            for (uint32_t peer = 0; peer < kRanks; ++peer) {
                expected = expected and (not needed[peer] or arrived[peer] == epoch);
                if (complete and needed[peer]) assert(visible[peer] == epoch * 1000 + peer);
                assert(successful_probes[peer] <= 1);
            }
            assert(complete == expected);
            return complete;
        };

        // Shared-only/all-masked tokens do not wait for or probe any peer.
        std::array<bool, kRanks> needed{};
        assert(poll(needed));
        for (auto count : probes) assert(count == 0);

        // A stale previous epoch must not satisfy a new launch. An unrelated
        // slow peer must not hold up tokens that require only the last peer.
        needed[kRanks - 1] = true;
        assert(not poll(needed));
        payload[kRanks - 1] = epoch * 1000 + kRanks - 1;
        arrived[kRanks - 1] = epoch;
        assert(poll(needed));
        const auto before = probes;
        for (uint32_t chunk = 0; chunk < 100; ++chunk) assert(poll(needed));
        assert(before == probes);
        for (uint32_t peer = 0; peer + 1 < kRanks; ++peer) assert(probes[peer] == 0);

        // Changing dependencies and out-of-order arrivals exercise lane 31,
        // multiple groups and the partial last group (e.g. EP33/EP72).
        for (uint32_t step = 0; step < 300; ++step) {
            const auto source = random() % kRanks;
            payload[source] = epoch * 1000 + source;
            arrived[source] = epoch;
            for (auto& value : needed) value = random() % 4 == 0;
            poll(needed);
        }
        for (uint32_t peer = 0; peer < kRanks; ++peer) {
            payload[peer] = epoch * 1000 + peer;
            arrived[peer] = epoch;
            needed[peer] = true;
        }
        assert(poll(needed));
        for (auto count : successful_probes) assert(count == 1);
    }
}

int main(int argc, char** argv) {
    assert(argc == 2);
    switch (std::atoi(argv[1])) {
        case 0: shared_visibility(); break;
        case 1: run<1>(); break;
        case 8: run<8>(); break;
        case 24: run<24>(); break;
        case 33: run<33>(); break;
        case 72: run<72>(); break;
        case 128: run<128>(); break;
        default: return 1;
    }
}
