# Experimental native MegaMoE RDMA

This branch implements a correctness-first NVSHMEM transport for the SM100
FP8 activation / MXFP4 weight SwiGLU kernel, including FP8 shared experts.
It has **not been CUDA-compiled or GPU-tested**. The default remains `nvlink`.

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
   zero counts. The existing grid ID plus one is exchanged for every invocation.
2. Dispatch issuers drain their NVSHMEM operations, synchronize locally, and one
   thread participates in `nvshmem_barrier_all`. Remote input slabs (FP8 tokens,
   packed scales, top-k IDs and weights) are then pulled into local staging.
   Counts from remote sources are accumulated **locally**, preserving the
   scheduler's `num_sms * num_ranks` high-word readiness condition. NIC atomics
   never contend with GPU atomics on the expert sum. A final local grid sync
   publishes the staged inputs before the existing ring-buffer pipeline runs.
3. Remote GEMM2 stores write local staging in `[peer][top-k][token][hidden]`
   order. After the epilogue grid sync, consecutive tokens owned by this rank in
   each top-k slot are sent in contiguous runs. Other ranks' slots are untouched;
   shared-expert output stays local. No 16-byte lane store becomes an RDMA call.
4. Each peer issuer calls `quiet`, publishes an NVSHMEM ready signal tagged with
   the **destination** grid ID, and drains it. The destination uses NVSHMEM
   `test` before reading the payload, followed by an async-proxy fence for TMA.
   Send-buffer consumption, target ordering, and ring consumer credit remain
   separate events.
5. The cleanup barrier follows all outgoing returns. Stream serialization and
   the next dispatch barrier prevent a later epoch from overtaking a previous
   combine. No collective free is run from a destructor. Allocations and loaded
   modules intentionally live for the worker process lifetime; reusing the same
   `SymmBuffer` is required to bound memory use.

The native path keeps its original barriers and direct operations. RDMA adds
full-capacity input and output staging as a deliberately simple first baseline.
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
  `nvshmemx_collective_launch_attr`. API source inspected during implementation:
  NVIDIA/nvshmem commit `eaf955fd938eefbdb17ec607ee11b71493f95959`.
  Older installations without that API are rejected by setup.
- Build with `DG_ENABLE_NVSHMEM=1`, `DG_FORCE_BUILD=1`, and `NVSHMEM_HOME`.
  Keep that NVSHMEM installation at the same path on the runtime workers.
  JIT compilation uses `nvcc -dc`, an explicit device link, a separate cache
  signature including the device-library contents, and `nvshmemx_cumodule_init`.
- The collective launch carries the required 2-SM cluster attributes, checks
  resident cluster capacity, and disables PDL. Warm up before graph capture.

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

When cards become available, run `tests/test_mega_moe.py --transport nvshmem`.
Its RDMA branch compares eager output and changing-route CUDA Graph replays
against native EP1 with the same quantized weights; expert-count statistics must
also agree exactly. It does not initialize a second NVSHMEM transport or silently
skip numerical verification. The reference supports at most 512 total experts.

Example per-node test command (existing harness uses `WORLD_SIZE` as the number
of nodes and `RANK` as the node index; this command spawns eight local workers):

```sh
WORLD_SIZE=2 RANK=0 MASTER_ADDR=<node0-address> MASTER_PORT=8361 \
  .venv/bin/python tests/test_mega_moe.py --num-processes 8 \
  --transport nvshmem --num-experts 256 --num-topk 8 \
  --num-max-tokens-per-rank 256 --num-tokens 128 \
  --num-shared-experts 1 --num-correctness-tests 100
```

Run the corresponding `RANK=1` command on node 1. First validate EP1/8, then
EP16, and EP24 with an expert count divisible by 24 (for example 384; 256 is
invalid). Include shared expert off/on, `--empty-rank 0`, masked routing,
capacity/ring wraparound and long replay runs. Still required: compile both
native/RDMA kernels, deliberate delayed-arrival/failure injection, graph stream
ordering under the actual NVSHMEM build, sanitizers, and vLLM model evaluation.
Barrier timeout/error propagation and register pressure require hardware review.
TPOT/throughput measurement is a separate pending stage.

## API references

- [NVSHMEM ordering](https://docs.nvidia.com/nvshmem/api/latest/gen/api/ordering.html)
- [NVSHMEM initialization and module setup](https://docs.nvidia.com/nvshmem/api/latest/gen/api/setup.html)
- [NVSHMEM collective launch source](https://github.com/NVIDIA/nvshmem/blob/eaf955fd938eefbdb17ec607ee11b71493f95959/src/device/launch/collective_launch.cpp)
- [CUDA dynamic kernel-handle interoperability](https://developer.nvidia.com/blog/dynamic-loading-in-the-cuda-runtime/)
