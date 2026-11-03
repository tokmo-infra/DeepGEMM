# Experimental native MegaMoE RDMA

This branch implements a correctness-first NVSHMEM transport for the SM100
FP8 activation / MXFP4 weight SwiGLU kernel, including FP8 shared experts.
Both build modes and GPU kernels have been compiled on B300 with CUDA 13.0.
Single-node tests include forced remote staging with P2P disabled. Cross-node
EP8 (four GPUs per node) also passed exact correctness checks, with measured
layer timings. Larger EP sizes and model serving validation remain pending.
The default remains `nvlink`.

## Protocol

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
   locally, and one thread participates in `nvshmem_barrier_all`. Remote inputs
   (FP8 tokens, packed scales, top-k IDs and weights) are then pulled into local
   staging. Only the live token prefixes are transferred; capacity-based offsets
   remain unchanged. An empty peer issues no input transfer.
   Counts from remote sources are accumulated **locally**, preserving the
   scheduler's `num_sms * num_ranks` high-word readiness condition. NIC atomics
   never contend with GPU atomics on the expert sum. A final local grid sync
   publishes the staged inputs before the existing ring-buffer pipeline runs.
3. Remote GEMM2 stores write local staging in `[peer][top-k][token][hidden]`
   order. After the epilogue grid sync, the whole epilogue CTA distributes
   independent owned rows across its threads. Each RDMA operation transfers a
   complete hidden row, avoiding a serial row loop in one lane per destination.
   Other ranks' slots are untouched; shared-expert output stays local.
4. Each issuing warp calls collective `quiet`, then a CTA rendezvous permits
   peer threads to publish an NVSHMEM ready signal tagged with the **destination**
   buffer epoch and drain it. One destination thread uses NVSHMEM `test_all` on
   all peers' readiness signals, and an epilogue grid sync distributes completion
   to combine readers. TMA reads retain the async-proxy fence. Per-token polling
   therefore need not repeat the NVSHMEM target-consistency operation.
   Send-buffer consumption, target ordering, and ring consumer credit remain
   separate events.
5. The cleanup barrier follows all outgoing returns. Stream serialization and
   the next dispatch barrier prevent a later epoch from overtaking a previous
   combine. No collective free is run from a destructor. Allocations and loaded
   modules intentionally live for the worker process lifetime; reusing the same
   `SymmBuffer` is required to bound memory use.

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

CPU-only checks, run locally with no CUDA runtime:

```sh
.venv/bin/python -m pytest tests/test_mega_moe_rdma_contract.py -q
```

This compiles the real staging-layout header with a host C++ compiler and checks
its offsets against Python, including capacities beyond 4 GiB. It also checks
allocation disagreement, invalid shapes and asymmetric peer reachability.
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

After live-prefix prefetch, parallel return submission, scoped collective
completion, one target-consistency check per layer, and the runtime corrections
above, the cross-node results were:

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
The smaller batch now transfers 10,960,896 input bytes per node instead of
164,413,440 (layout-derived, excluding routes and returns). Additional RDMA
staging remains 1,028,875,264 bytes per rank, beyond the 355,896,704-byte native
buffer. These results still do not establish a full-model deployment benefit.

Final single-node EP8 controls also passed 20 changing-route replays per case
and the same seven timed-batch checks. Native NVLink measured 0.192/0.225 ms
and local NVSHMEM measured 0.217/0.249 ms at 128/512 tokens per rank. The final
cross-node path remains about 5.6/12.4 times slower than native NVLink here.

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
