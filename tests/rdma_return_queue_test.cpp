// CPU execution of the real queue protocol. This does not validate PTX or NIC visibility.
#include <deep_gemm/comm/mega_moe_return_queue.cuh>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

struct Atomic {
    static inline std::atomic<bool> delay{false}, reserved{false}, resume{false};
    static uint32_t load(const uint32_t* p) {
        return std::atomic_ref(*const_cast<uint32_t*>(p)).load(std::memory_order_acquire);
    }
    static void store(uint32_t* p, uint32_t v) {
        if (delay.load() and v == 1) {
            reserved.store(true);
            while (not resume.load()) std::this_thread::yield();
        }
        std::atomic_ref(*p).store(v, std::memory_order_release);
    }
    static uint32_t add(uint32_t* p, uint32_t v) {
        return std::atomic_ref(*p).fetch_add(v, std::memory_order_acq_rel);
    }
};

using deep_gemm::layout::MegaMoERdmaBuffer;
template <uint32_t S> using Queue = deep_gemm::comm::MegaMoEReturnQueue<S, Atomic>;

static void holes() {
    std::vector<uint32_t> storage(8192, 0);
    MegaMoERdmaBuffer layout(storage.data(), 4, 64, 64, 128, 3);
    Queue<4> queue{layout};
    Atomic::delay.store(true);
    std::thread producer([&] { queue.publish_tile(0, 1); });
    while (not Atomic::reserved.load()) std::this_thread::yield();
    queue.publish_tile(4, 1); // Same shard, later ticket published first.
    uint32_t row = 999;
    assert(not queue.peek(0, 0, row) and row == 999);
    assert(queue.peek(0, 1, row) and row == 4);
    assert(not queue.closed_and_empty(0, 2)); // Empty is not end-of-stream.
    Atomic::resume.store(true);
    producer.join();
    Atomic::delay.store(false);
    queue.close();
    assert(queue.peek(0, 0, row) and row == 0); // Row zero is not the sentinel.
    assert(not queue.closed_and_empty(0, 1));
    assert(queue.closed_and_empty(0, 2));
    for (unsigned s = 1; s < 4; ++s) assert(queue.closed_and_empty(s, 0));
    for (unsigned s = 0; s < 4; ++s) {
        assert(not queue.all_senders_done());
        queue.finish_sender();
    }
    assert(queue.all_senders_done());
}

template <uint32_t S> static void stress(bool late_senders) {
    constexpr uint32_t tiles = 7, capacity = 20;
    MegaMoERdmaBuffer layout(nullptr, 4, 64, capacity, 128, 3);
    assert(layout.can_queue_returns(S));
    std::vector<uint32_t> storage(layout.input_offset() / 4 + 32, 0xfeedbeef);
    layout.base = reinterpret_cast<uint8_t*>(storage.data());
    Queue<S> queue{layout};
    const auto rows = static_cast<uint32_t>(layout.return_rows());
    std::vector<uint32_t> payload(rows * tiles), seen(rows);
    for (uint32_t replay = 0; replay < 12; ++replay) {
        std::fill_n(layout.routes(0), layout.return_queue_words(S), 0);
        std::fill(payload.begin(), payload.end(), 0);
        std::fill(seen.begin(), seen.end(), 0);
        // Full capacity, changing masked/local routes, and entirely empty replays.
        const auto active = [=](uint32_t row) {
            return replay % 3 == 0 or (replay % 3 == 1 and (row + replay) % 5 < 2);
        };
        std::vector<std::thread> consumers;
        const auto start_senders = [&] {
            for (uint32_t shard = 0; shard < S; ++shard) consumers.emplace_back([&, shard] {
                uint32_t cursors[32];
                for (uint32_t lane = 0; lane < 32; ++lane) cursors[lane] = lane;
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
                for (;;) {
                    bool done = true;
                    for (uint32_t lane = 0; lane < 32; ++lane) {
                        uint32_t row;
                        if (queue.peek(shard, cursors[lane], row)) {
                            assert(row < rows and row % S == shard and active(row));
                            assert(seen[row]++ == 0);
                            // All fragments must be visible after the queue acquire,
                            // including fragments written by other producer threads.
                            for (uint32_t t = 0; t < tiles; ++t)
                                assert(payload[row * tiles + t] == replay + t + 1);
                            cursors[lane] += 32;
                        }
                        done &= queue.closed_and_empty(shard, cursors[lane]);
                    }
                    if (done) break;
                    assert(std::chrono::steady_clock::now() < deadline);
                    std::this_thread::yield();
                }
                queue.finish_sender();
            });
        };
        if (not late_senders) start_senders();
        std::vector<std::thread> producers;
        for (uint32_t t = 0; t < tiles; ++t) producers.emplace_back([&, t] {
            for (uint32_t i = 0; i < rows; ++i) {
                const auto row = (i + t * 31) % rows;
                if (not active(row)) continue;
                payload[row * tiles + t] = replay + t + 1;
                queue.publish_tile(row, tiles);
                if (i % 11 == 0) std::this_thread::yield();
            }
        });
        for (auto& producer : producers) producer.join(); // Models epilogue grid join.
        queue.close();
        if (late_senders) start_senders(); // Producers can fill every slot without a consumer.
        while (not queue.all_senders_done()) std::this_thread::yield();
        for (uint32_t row = 0; row < rows; ++row) assert(seen[row] == uint32_t(active(row)));
        for (auto& consumer : consumers) consumer.join();
        const auto* end = layout.routes(0) + layout.return_queue_words(S);
        for (auto* p = end; p < storage.data() + storage.size(); ++p) assert(*p == 0xfeedbeef);
        for (uint32_t i = 0; i < layout.route_offset() / 4; ++i) assert(storage[i] == 0xfeedbeef);
    }
}

int main(int argc, char** argv) {
    assert(argc == 2);
    const std::string test = argv[1];
    if (test == "holes") holes();
    else if (test == "concurrent") { stress<1>(false); stress<3>(false); stress<16>(false); stress<32>(false); }
    else if (test == "late") { stress<1>(true); stress<3>(true); stress<16>(true); stress<32>(true); }
    else return 1;
}
