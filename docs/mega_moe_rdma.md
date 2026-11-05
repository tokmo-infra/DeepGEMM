# Experimental native MegaMoE RDMA

This branch implements a correctness-first NVSHMEM transport for the SM100
FP8 activation / MXFP4 weight SwiGLU kernel, including FP8 shared experts.
Both build modes and GPU kernels have been compiled on B300 with CUDA 13.0.
Single-node tests include forced remote staging with P2P disabled. Cross-node
EP8 (four GPUs per node) also passed exact correctness checks, with measured
layer timings. EP24 synthetic correctness was subsequently exercised; model
serving validation remains pending.
The default remains `nvlink`.
Historical GPU tables below describe the streaming baseline (`67f08c5`). The
2026-11-05 section records the later queue experiments (`8e5b4be`). The new
phase-overlap candidate has only CPU checks; CUDA/GPU validation is pending.

## Protocol (default overlap mode 0)

The main symmetric allocation uses NVSHMEM. `nvshmem_ptr` determines which peers
are directly accessible; inaccessible peers have a zero pointer and cannot pass
`SymBuffer.map`. Both sides must agree on direct reachability. EP rank order is
the NVSHMEM PE order, and the bootstrap Gloo group must have that same order.
NVSHMEM selects NICs and connections; deployments must configure the intended
RDMA transport and HCA selection before starting workers.

1. Local peers retain direct route stores. Remote routes are written into local
   `[global expert][token slot]` staging and sent as one contiguous row per
   populated expert. Each source writes a separate receive-count slot, including
   zero counts. A nonzero, GPU-incremented counter in each RDMA buffer is exchanged
   for every invocation. CUDA Graph replays can reuse `%gridid`, so it cannot
   identify a fresh RDMA return. One dispatch thread advances the counter before
   the existing grid barrier publishes it to the other threads.
2. CTA 0's dispatch warps drain their NVSHMEM operations collectively, synchronize
   locally, and one thread participates in `nvshmem_barrier_all`. Only live input
   prefixes are fetched; capacity offsets remain unchanged. Within each
   `NVSHMEM_TEAM_SHARED` clique, one cache owner fetches each remote source's
   FP8 tokens, scales, top-k IDs and weights. Source PE modulo shared-team size
   selects the owner, spreading traffic across local NICs. A shared-team barrier
   publishes completed caches for direct NVLink reads by other local GPUs.
   NVSHMEM reduces this team to self when it cannot establish a shared clique;
   that configuration retains one independent cache per GPU.
   Remote counts are accumulated **locally**, preserving the scheduler's
   `num_sms * num_ranks` readiness condition without mixing NIC and GPU atomics.
   The existing final grid sync publishes inputs before the ring pipeline runs.
3. Remote GEMM2 stores retain `[peer][top-k][token][hidden]` staging. Each half
   warp synchronizes after storing a row fragment, then its elected lane uses
   an acquire/release atomic tile counter. The last tile submits the complete
   hidden row immediately, while the original pipeline continues other tiles.
   After dispatch completes, its dead route send slab holds these counters;
   there is no new allocation or layout ABI change. If `ranks * topk > experts`,
   the counters would not fit, so the epilogue CTA instead sends owned rows after
   its grid sync. Other expert ranks' slots are untouched; shared output stays
   local. GEMM scheduling, TMA stages, ring credits and reduction order are kept.
4. The existing epilogue grid sync orders all streaming submissions. One elected
   warp completes the PE's outstanding RMA, a CTA rendezvous precedes publishing
   the **destination** epochs, and another collective quiet completes signals.
   NVSHMEM permits one thread/warp to perform quiet after synchronization with
   all issuers; repeating it in every issuing warp is unnecessary.
   Dispatch cleanup starts after outgoing completion, overlapping receipt checks
   and combine. One destination thread uses NVSHMEM `test_all` on all readiness
   signals, then an epilogue grid sync distributes completion to combine readers.
   TMA reads retain their async-proxy fence. Send-buffer consumption, target
   visibility and ring consumer credit remain separate events.
5. Stream serialization and the next dispatch barrier prevent a later epoch
   from overtaking a previous combine or overwriting a shared input cache still
   in use. No collective free is run from a destructor. Allocations and loaded
   modules live for the worker process lifetime; reuse the same `SymmBuffer`
   to bound memory use.

The native path keeps its original barriers and direct operations. RDMA reserves
full-capacity input and output staging, while transferring only valid inputs
and owned return rows.
With `N` ranks, `M` token capacity, `E` experts, hidden size `H`, and top-k `K`,
extra bytes are approximately `1024 + 4*E*M + N*M*(H + H/32 + 12*K + 2*K*H)`
plus alignment. Slots for directly accessible peers are also reserved in this
initial layout. The default staging limit is 4 GiB **per distinct buffer**;
`DG_MEGAMOE_RDMA_MAX_STAGING_BYTES` explicitly raises it. This is additional to
the native workspace and does not configure NVSHMEM's symmetric heap size.

## Build and API

Use the exact DeepGEMM branch paired with the vLLM branch. Requirements:

- Linux, SM100-family CUDA devices, CUDA >=12.9, and the baseline's PyTorch ABI.
- NVSHMEM headers and `libnvshmem_host.so` / `libnvshmem_device.a` providing
  `nvshmemx_collective_launch_attr` and `nvshmemx_quiet_warp`.
  API source inspected during implementation:
  NVIDIA/nvshmem commit `eaf955fd938eefbdb17ec607ee11b71493f95959`.
  Older installations without that API are rejected by setup.
- Build with `DG_ENABLE_NVSHMEM=1`, `DG_FORCE_BUILD=1`, and `NVSHMEM_HOME`.
  Keep that NVSHMEM installation at the same path on the runtime workers.
  JIT compilation uses `nvcc -dc`, an explicit device link, a separate cache
  signature including the device-library contents, and `nvshmemx_cumodule_init`.
  Both compilation and linking use explicit family-specific `--generate-code`
  targets; RDC must not introduce generic PTX lacking the kernel's instructions.
  Unclamped activation is encoded with a separate boolean because NVCC's RDC
  host stub cannot represent an infinite floating-point template argument.
- The collective launch carries the required 2-SM cluster attributes, checks
  resident cluster capacity, and disables PDL. Warm up before graph capture.

The runtime must load the same NVSHMEM host library used at build time. Images
may include an older library through PyTorch or other packages. For the tested
image, select the new installation before importing PyTorch:

```sh
export LD_LIBRARY_PATH="$NVSHMEM_HOME/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$NVSHMEM_HOME/lib/libnvshmem_host.so${LD_PRELOAD:+:$LD_PRELOAD}"
```

```python
buffer = deep_gemm.SymmBuffer(
    ep_group, num_experts, max_tokens, topk, hidden, intermediate,
    num_shared_experts=1,
    transport="nvshmem", bootstrap_group=ep_gloo_group,
)
# Prepare inputs exactly as for native MegaMoE, then use fp8_fp4_mega_moe.
```

All ranks must allocate matching configurations in the same order. One CUDA
device and one EP ordering per process are supported. An already initialized
NVSHMEM world (for example DeepEP in the same process) is rejected. Overlapping
launches on a shared buffer, base-buffer aliasing, other precision/activation
paths, and `DG_COMM_KERNEL_DEBUG` are unsupported in this first version.

## Validation

### Phase overlap: CPU-checked candidate, CUDA/GPU validation pending

Set `DG_MEGAMOE_RDMA_OVERLAP=1` to test the candidate. The default `0` retains
the preceding protocol. Set the same value on every worker before allocation
and warmup; restart workers when changing it. Rebuild the paired DeepGEMM
extension, which forwards the option into the JIT cache key. Bootstrap rejects
invalid values and cross-rank disagreement. Native NVLink ignores the option.
It composes with `DG_MEGAMOE_RDMA_RETURN_SENDERS`; start A/B with senders `0`.

The candidate changes transport dependencies, while keeping the scheduler,
GEMM arithmetic, TMA pipeline, ring credit, shared-expert math and top-k
reduction order:

1. Dispatch RMA completion and the world barrier still precede consumption of
   routing metadata. Remote counts are accumulated before bulk input GETs, so
   routed task/weight preparation no longer waits for activation staging.
2. The original dispatch/epilogue rendezvous runs after metadata and return
   state initialization, before input GETs. Shared epilogues can progress during
   prefetch. A separate **dispatch-only** grid rendezvous and async-proxy fence
   still complete all input caches before any ring producer reads them. This
   adds one dispatch grid synchronization; its cost needs measurement.
3. Dispatch CTAs clear disjoint portions of the dead route slab after the
   existing grid joins all route issuers' quiet calls. The metadata grid joins
   those clears before any return producer. Streaming/queued modes skip the
   unused top-k-ID input GET; the scan fallback still fetches it.
4. After the existing epilogue grid joins direct output stores, mapped peers
   receive their native release-store ready signal before the remote sender
   join/quiet. Remote readiness still follows the full sender join and PE-wide
   quiet. Signal completion and the cleanup rendezvous remain in place.
5. Combine removes the all-peer `test_all` and its extra epilogue grid barrier.
   Each warp waits only for source peers selected by its current token. Shared
   and masked slots add no dependency. Mapped sources use native acquire;
   remote sources require a successful NVSHMEM `test`, preserving target
   consistency before the existing TMA async-proxy fence.

Remote visibility is shared across the GPU: a per-peer state transitions from
unknown to claimed to visible. One winning lane executes the consistency probe
and publishes success with GPU release ordering; other readers acquire that
publication. A failed probe releases the claim. Losing lanes return to the
warp-wide poll without spinning inside divergence, so they cannot prevent a
winning lane from running. A per-lane register cache then avoids repeat probes
for later tokens/chunks in the same launch. This limits **successful remote
consistency probes to one per required peer per receiving GPU per launch**,
rather than one per token or warp. Failed probes can retry.

The shared visibility states occupy previously unused staging-header bytes
after peer token counts: at EP72 they end at byte 864, before the epoch at 1016.
They reset before the dispatch metadata grid and are never reused across
launches. Allocation sizes, native offsets and RDMA ABI 1 are unchanged.
The retained outgoing quiet, signal completion, cleanup world barrier and
serialized launch order still protect send slabs, input caches and epochs;
removing the receive grid does not remove those lifetime requirements.

Local checks on 2026-11-05: 56 CPU contract tests passed, including compiled
header tests of selected/late peers, masked dependencies, repeated epochs,
header bounds and concurrent visibility publication. The shared-cache test
also passed ThreadSanitizer. These tests do not emulate CUDA/NIC ordering.
This revision has **not** been CUDA compiled, run on GPUs, or benchmarked;
no machines or GPUs were accessed during this change. Earlier compilation and
GPU results apply to their recorded revisions, not this candidate.

Remaining bottlenecks are explicit: input GETs still transfer entire live
prefixes before routed ring pulls, return tiles still use per-row atomics,
remote ready signals still follow a PE-wide quiet, and cleanup still has a
world barrier. This candidate does not claim to restore all native overlap.
The next hardware run must first compile/link both modes, then compare:

- Native NVLink, local NVSHMEM and cross-node overlap `0/1`, with identical
  shapes/seeds, GPU/NIC mappings and input hashes; primary tokens `128/512`.
- Sender `0/8`, shared expert off/on, empty/uneven ranks, all-masked routes,
  P2P disabled, counter/queue storage fallbacks, long changing-route graph
  replay and ring reuse; require exact output and expert-count checks.
- Delayed selected and unselected sources: verify unrelated peers do not gate
  combine and the visibility claim never blocks another lane's progress.
- Count publication, shared epilogue start, input readiness, epilogue/sender
  completion, receive/combine and cleanup timestamps; report total layer time,
  register/spill changes and native regression checks before enabling it.

### Queued returns: opt-in experiment

`DG_MEGAMOE_RDMA_RETURN_SENDERS=16` selects a sharded return queue for RDMA.
The default `0` keeps the validated direct streaming path; values `1..32`
select that many sender warps, capped by the kernel's CTA count. Set the same
value on all workers **before allocating buffers and warming up**, and restart
workers when changing it. Rebuild the matching DeepGEMM extension: an older
extension does not forward this option to JIT. The option is included in the
allocation agreement and JIT cache key. Native NVLink ignores it.

Each selected CTA's first dispatch warp becomes a sender only after finishing
all its original input pulls. No extra CTA, warp, shared-memory allocation or
SM reservation is introduced. GEMM/TMA scheduling, ring credits, register
budgets, shared-expert computation and top-k reduction order are unchanged.
The last tile of a remote row publishes a descriptor with release ordering;
the epilogue never waits for a sender or calls NVSHMEM to submit that row.
Senders acquire descriptors and submit complete BF16 rows while remaining
GEMM work proceeds. All SMs remain available to the original compute pipeline.

Queues are append-only within a launch. For `R = ranks * topk * capacity` rows
and `S` senders, row `r` belongs to shard `r % S`, with `ceil(R / S)` entries.
Counters, entries, tails, close and sender-completion state occupy
`R + S * ceil(R / S) + S + 2` words of the completed dispatch route slab.
Every possible row fits without recycling entries or waiting for queue space,
even if senders start after all producers finish. With EP8/E384/K6/capacity1920
and 16 senders this uses 737,352 bytes of the existing 2,949,120-byte route slab.
Main/staging allocation sizes and RDMA ABI 1 remain unchanged. If the queue does
not fit, execution retains direct streaming; if counters also do not fit, it
retains the existing post-grid scan fallback.

A reserved ticket is not a ready descriptor: readers wait for a nonzero entry
and can advance other lanes past a delayed publication. The existing epilogue
grid rendezvous joins all producers before CTA 0 closes the queues. Senders
drain them and publish completion only after every lane returns from its put
calls. CTA 0 acquires all sender completions before the existing elected quiet
warp, peer epoch signals and receive checks. This preserves the required
ordering between different issuing threads and `nvshmem_quiet`. Workspace
cleanup and combine retain their existing overlap; no per-row quiet is added.
The next dispatch completion resets the queue before any new row publication.

CPU checks exercise the real queue protocol with C++ atomics, including delayed
publication, full-capacity producers preceding consumers, concurrent fragment
visibility, empty/masked routes and repeated resets. These do not prove CUDA
memory ordering, NIC progress, graph replay correctness or performance.
On 2026-11-04, all 38 CPU contract tests and the queue tests under
ThreadSanitizer passed. CUDA 13.0.88 compiled/linked 16 offline cases without
GPU access: both primary block shapes, sender counts 8/16/32, both storage
fallbacks, shared expert off and a sender-count clamp. The native NVLink kernel
instruction bytes matched the preceding revision for both primary shapes.
Queued fusion bodies reported 116 registers and no spill loads/stores; NVSHMEM
library functions still have their own stack/spill usage. These compiler
figures do not measure runtime register pressure or latency.
Keep the default `0`; the latest hardware results below do not establish a
consistent large-batch advantage. Further A/B runs compare
`0/8/16/32` with identical seeded inputs, 100 changing-route graph replays and
the seven-batch timing checks below. Include queue-storage fallback, no shared
expert, uneven/empty ranks, all-masked routes, P2P disabled, local NVSHMEM and
native NVLink controls. Record sender start times, epilogue completion, sender
join, quiet, register spills and end-to-end layer latency; a shorter epilogue
alone is insufficient to establish an improvement.

### 2026-11-05 queue results (before the phase-overlap candidate)

DeepGEMM `8e5b4be` passed exact eager/reference checks, 100 changing-route
graph replays and checks after every timed batch in the main EP8 and EP24
matrix. EP8 used 55/89 with four GPUs per node; EP24 used 55/89/91 with eight
per node. Shape: E384/K6/H5120/I2304/shared1, aligned capacity 1920.

| Tokens/rank | EP8 direct ms | EP8 queue8 ms | EP24 direct ms | EP24 queue8 ms |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 0.658 | 0.619 | 2.065 | 2.044 |
| 512 | 0.979 | 0.956 | 8.353 | 9.859 |

EP24 and EP8's larger queued case were noisy; sender counts 16/32 regressed
the EP8 large case. Native EP8 on node 91 measured 0.195/0.227 ms with matching
EP8 input hashes (a different-host control). Queue8 remained about 3.17/4.22x
slower. These are synthetic layer measurements, not model TPOT. Only the
direct EP8 phase trace finished before testing was paused; there is no complete
queue8 phase trace. Defaults remain unchanged, and GPU work is paused.

### CPU contract checks

CPU-only checks, run locally with no CUDA runtime:

```sh
.venv/bin/python -m pytest tests/test_mega_moe_rdma_contract.py -q
```

This compiles the real staging-layout header with a host C++ compiler and checks
its offsets against Python, including capacities beyond 4 GiB and counter-slab
exact-fit/fallback boundaries. It also checks allocation disagreement, invalid
shapes and asymmetric peer reachability.
It does not prove device code or network correctness.

Run `tests/test_mega_moe.py --transport nvshmem`.
Its RDMA branch compares eager output and changing-route CUDA Graph replays
against native EP1 with the same quantized weights; expert-count statistics must
also agree exactly. It does not initialize a second NVSHMEM transport or silently
skip numerical verification. The reference supports at most 512 total experts.

On 2026-11-04, single-node B300 validation used CUDA 13.0.88,
PyTorch 2.13.0+cu130 and NVSHMEM 3.9.0.dev0 built from the revision above:

- Native EP1/EP8 launch smoke tests and both C++ extension build modes passed.
- NVSHMEM EP1/EP8 eager output and 20 changing-route graph replays matched the
  native EP1 reference exactly; EP8 included one shared expert.
- Forced remote EP2 passed eager output, expert counts and 100 changing-route
  replays after the per-buffer epoch fix. Each rank mapped only itself.
- Forced remote EP8 also passed 100 replays with one shared expert, rank 0 empty
  and 50% masked routes. Outputs and expert counts matched exactly.
- A larger forced remote EP8 case passed 20 replays with 256 experts, top-k 8,
  hidden 4096, intermediate 2048, 128 tokens per rank, one shared expert and no
  activation clamp. This also exercises the RDC unclamped-template fix.
- The vLLM EP1 wrapper passed eager output and 20 changing-route replays with
  default unclamped activation and separate transport buffer caches.

To exercise remote staging on a single machine, disable P2P and require the
harness to verify that a peer is unmapped. For example, with this machine's HCA
and RoCE GID selection (adjust them to the target host):

```sh
NVSHMEM_DISABLE_P2P=1 NVSHMEM_REMOTE_TRANSPORT=ibrc \
NVSHMEM_IB_GID_INDEX=3 NVSHMEM_ENABLE_NIC_PE_MAPPING=1 \
NVSHMEM_HCA_LIST=mlx5_0:1 NVSHMEM_SYMMETRIC_SIZE=2G \
  .venv/bin/python tests/test_mega_moe.py --transport nvshmem \
  --require-remote-peers --num-processes 2 --num-experts 32 --num-topk 2 \
  --hidden 1024 --intermediate-hidden 1024 --num-max-tokens-per-rank 64 \
  --num-tokens 32 --num-shared-experts 0 --num-correctness-tests 100
```

This validates the remote protocol through a local HCA; it does not validate
inter-node routing, fabric behavior or multi-node performance. The native buffer
rounds token capacity up to its required alignment; staging uses that capacity.

### Cross-node correctness and layer timing

Two B300 nodes with four GPUs each passed eager and 100 changing-route graph
replays, including a shared expert, an empty rank and 50% masked routes. Each
rank directly mapped only its four node-local peers. The same test harness can
also benchmark both transports against its native EP1 reference:

```sh
WORLD_SIZE=2 RANK=0 MASTER_ADDR=<node0-address> MASTER_PORT=8361 \
  .venv/bin/python tests/test_mega_moe.py --num-processes 4 \
  --transport nvshmem --require-remote-peers \
  --num-experts 384 --num-topk 6 --hidden 5120 --intermediate-hidden 2304 \
  --num-shared-experts 1 --num-max-tokens-per-rank 512 --num-tokens 128 \
  --num-correctness-tests 100 --benchmark-iterations 20 --benchmark-repeats 7
```

Run `RANK=1` on the second node with matching NIC/bootstrap settings. For the
single-node EP8 controls, use `WORLD_SIZE=1`, eight local processes, and omit
`--require-remote-peers`; test both `nvlink` and `nvshmem`.

Each timed graph contains 20 fused calls including input copies and statistics
resets. Three graphs warm up before seven timed batches. Reference checks,
quantization, inter-batch barriers and JIT are outside the timing. Outputs and
counts are checked after each timed batch. The original seeded inputs and routes
are restored after correctness replays; `BENCHMARK_JSON` includes their SHA-256
per rank so different transports can verify an identical workload. Report the
median of each batch's maximum rank-average latency. These are synthetic MoE
layer measurements, not serving throughput or request latency percentiles.

NVSHMEM IBGDA can be enabled with `NVSHMEM_IB_ENABLE_IBGDA=1`. Initially,
`cudaHostRegister(IoMemory)` returned error 800 and the requested GPU NIC handler
fell back to CPU host memory. Adding `SYS_RAWIO` did not resolve it. The tested
NVIDIA driver checks `CAP_SYS_ADMIN` for administrator access; adding that
capability to the isolated test containers resolved GPU NIC buffer mapping on
both hosts, without reloading or reconfiguring the host drivers. Follow the
deployment's approved driver/container configuration for peer mappings.
Set `NVSHMEM_IBGDA_NIC_HANDLER=gpu` and `NVSHMEM_DEBUG=INFO`, then verify
`NIC handler will be GPU` in every worker's log. The requested handler alone
does not prevent fallback in this NVSHMEM build.

Use a GPU-local HCA per worker, and match the fabric's traffic class. These
tests used `NVSHMEM_IB_GID_INDEX=3`, `NVSHMEM_IB_TRAFFIC_CLASS=106`,
`NVSHMEM_ENABLE_NIC_PE_MAPPING=1`, `NVSHMEM_MAX_NICS_PER_PE=1`, and explicit
`NVSHMEM_HCA_PE_MAPPING`. Do not also set `NVSHMEM_HCA_LIST` when using that
mapping. Each mapping entry is `HCA:port:local-worker-count`; for example:

```sh
# Host 55, local GPUs 0 through 3:
export NVSHMEM_HCA_PE_MAPPING=mlx5_0:1:1,mlx5_2:1:1,mlx5_6:1:1,mlx5_8:1:1
# Host 90 instead uses mlx5_0, mlx5_6, mlx5_8, mlx5_10, each with :1:1.
```

These names and traffic class are specific to the measured fabric. Keep the
tested IBGDA defaults (two RC QPs per peer and request batch 32); increasing RC
QPs to eight and reducing the batch to one regressed this workload. NIC PFC,
trust mode, host sysctls and driver settings were not changed by the tests.

Measured on 2026-11-04 with B300, CUDA 13.0.88, PyTorch 2.13.0+cu130 and the
NVSHMEM revision above. All cases use EP8, 384 experts, top-k 6, hidden 5120,
intermediate 2304, one shared expert and SwiGLU clamp 10. Requested capacity 512
aligns to 1920. Native and NVSHMEM local controls use eight GPUs on one node.
The two nodes use drivers 580.105.08 and 580.196.07. The initial baseline used
four GPUs and one shared RoCE HCA per node. Its median layer latency in
milliseconds was:

| Tokens per rank | Native NVLink | Local NVSHMEM | Cross-node IBRC | Cross-node IBGDA, CPU handler |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 0.191 | 0.226 | 84.277 | 11.719 |
| 512 | 0.225 | 0.258 | 217.581 | 16.398 |

All eight baseline cases passed exact checks and used matching per-rank input
hashes. Both IBGDA cases also passed 100 changing-route replays. Their
batch-average ranges were 9.206–15.102 ms and 15.520–25.122 ms.

The previous revision added live-prefix prefetch, parallel return submission,
scoped collective completion and one target-consistency check per layer. With
the runtime corrections above, its cross-node results were:

| Tokens per rank | GPU-handler IBGDA median ms | Min–max ms | Speedup over initial IBGDA |
| ---: | ---: | ---: | ---: |
| 128 | 1.072 | 1.064–1.077 | 10.9x |
| 512 | 2.800 | 2.736–2.839 | 5.9x |

Both cases passed 100 changing-route replays and exact checks after every timed
batch, with input hashes matching the baseline. The final kernel also passed
100 replays each for no shared expert, shared plus an empty rank and 50% masked
routes, uneven token counts, and an IBRC fallback case. All eight IBGDA workers
used the GPU NIC handler. The speedups above include both kernel and runtime
changes; switching only the NIC handler left the 128-token median at 11.730 ms.
An additional A/B run of the original kernel with the final GPU handler, HCA
mapping and traffic class measured 6.480 ms (6.479–6.496 ms) at 128 tokens/rank,
versus 1.072 ms with the final kernel: a 6.0x kernel speedup under the same
runtime configuration and input hashes.
That revision transfers 10,960,896 input bytes per node at 128 tokens instead of
164,413,440 (layout-derived, excluding routes and returns). Additional RDMA
staging remains 1,028,875,264 bytes per rank, beyond the 355,896,704-byte native
buffer. These results still do not establish a full-model deployment benefit.

The previous single-node EP8 controls passed 20 changing-route replays per
case and the same seven timed-batch checks. Native NVLink measured 0.192/0.225 ms
and local NVSHMEM measured 0.217/0.249 ms at 128/512 tokens per rank. Those
cross-node results were about 5.6/12.4 times slower than native NVLink.

The validated streaming baseline adds shared-team input caching and sends each remote
result row as soon as GEMM2 finishes its last tile. One elected warp drains
posted returns after the existing grid sync, while workspace cleanup overlaps
incoming completion and combine. Native NVLink retains its original ordering;
GEMM/TMA scheduling, ring credits and top-k reduction order are unchanged.
No extra staging is allocated. At four GPUs per node, input payload falls by
4x to 2,740,224 / 10,960,896 bytes per node at 128 / 512 tokens (layout-derived,
excluding routes and returns).

With the same runtime configuration and seeded workloads, that baseline's cross-node
medians are **0.846 / 1.986 ms** (ranges 0.837–0.849 / 1.980–2.061 ms), reducing
latency by **21% / 29%** versus 1.072 / 2.800 ms. Each case passed 100 changing-route
replays and exact checks after all seven timed batches. Packet gathering into
larger messages regressed performance and was discarded. RC=8 did not clearly
improve the streaming implementation, RC=16 regressed, and splitting prefetch
into 64 KiB requests provided no measurable benefit; defaults remain RC=2 and
request batch 32. The cross-node protocol still has dispatch/prefetch barriers
and receive waits; preserving the native compute pipeline does not remove
these network stalls.

The streaming baseline passed nine cross-node cases with 100 changing-route replays
each: both performance shapes, shared disabled, empty rank plus 50% masked
routes, uneven token counts, IBRC fallback, insufficient counter space
(32 experts / top-k 5), P2P disabled (each shared team is self), and exact-fit
counter storage (32 experts / top-k 4) with 1920 tokens per rank. CPU layout and
contract checks pass all 18 tests, including the counter boundary conditions.

With the same final headers, single-node EP8 controls passed 20 changing-route
replays each and all timed-batch checks. Native NVLink measured 0.191 / 0.225 ms
at 128 / 512 tokens (previously 0.192 / 0.225 ms); local NVSHMEM measured
0.220 / 0.253 ms (previously 0.217 / 0.249 ms). The native path shows no
measurable regression in these cases. Cross-node execution remains about
4.4 / 8.8 times slower than native NVLink, so further network and protocol
optimization is still required before a deployment performance claim.

Example per-node test command (existing harness uses `WORLD_SIZE` as the number
of nodes and `RANK` as the node index; this command spawns eight local workers):

```sh
WORLD_SIZE=2 RANK=0 MASTER_ADDR=<node0-address> MASTER_PORT=8361 \
  .venv/bin/python tests/test_mega_moe.py --num-processes 8 \
  --transport nvshmem --num-experts 256 --num-topk 8 \
  --num-max-tokens-per-rank 256 --num-tokens 128 \
  --num-shared-experts 1 --num-correctness-tests 100
```

Run the corresponding `RANK=1` command on node 1. Beyond cross-node EP8, validate
EP16, and EP24 with an expert count divisible by 24 (for example 384; 256 is
invalid). Include shared expert off/on, `--empty-rank 0`, masked routing,
capacity/ring wraparound and long replay runs. Still required: deliberate
delayed-arrival/failure injection, multi-node graph stream ordering under the
deployment's NVSHMEM build, sanitizers, and vLLM model evaluation.
Barrier timeout/error propagation and register pressure require hardware review.
End-to-end TPOT/throughput measurement is a separate pending stage.

## API references

- [NVSHMEM ordering](https://docs.nvidia.com/nvshmem/api/latest/gen/api/ordering.html)
- [NVSHMEM initialization and module setup](https://docs.nvidia.com/nvshmem/api/latest/gen/api/setup.html)
- [NVSHMEM collective launch source](https://github.com/NVIDIA/nvshmem/blob/eaf955fd938eefbdb17ec607ee11b71493f95959/src/device/launch/collective_launch.cpp)
- [CUDA dynamic kernel-handle interoperability](https://developer.nvidia.com/blog/dynamic-loading-in-the-cuda-runtime/)
