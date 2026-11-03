"""Opt-in NVSHMEM session. Allocations live until worker process exit."""

import os
import types

import torch
import torch.distributed as dist

from .. import _C
from .rdma_contract import RdmaLayout, validate_manifests, validate_peer_mappings

_session = None


def get_mega_moe_rdma_abi_version():
    op = getattr(torch.ops.deep_gemm, 'mega_moe_rdma_abi_version', None)
    return 0 if op is None else op()


def allocate_buffers(group, bootstrap_group, config, native_bytes):
    """Collectively validate, bootstrap, and allocate main + staging buffers.

    This runs before graph capture. All layers sharing a configuration should
    reuse their buffer; individual tensors must never own collective teardown.
    """
    global _session
    if bootstrap_group is None or dist.get_backend(bootstrap_group) != 'gloo':
        raise ValueError('MegaMoE RDMA requires the EP Gloo group as bootstrap_group')
    ranks = tuple(dist.get_process_group_ranks(group))
    bootstrap_ranks = tuple(dist.get_process_group_ranks(bootstrap_group))
    error = ''
    staging_bytes = 0
    try:
        if ranks != bootstrap_ranks:
            raise ValueError('EP and bootstrap rank order must match')
        if torch.cuda.is_current_stream_capturing():
            raise ValueError('Allocate and warm up RDMA buffers before graph capture')
        if get_mega_moe_rdma_abi_version() != 1:
            raise RuntimeError('Rebuild the loaded DeepGEMM with DG_ENABLE_NVSHMEM=1')
        if config['mma_type'] != 'fp8xfp4' or config['activation'] != 'swiglu':
            raise ValueError('RDMA supports only FP8 x MXFP4 SwiGLU')
        if torch.cuda.get_device_capability()[0] != 10:
            raise ValueError('RDMA MegaMoE requires SM100-family GPUs')
        if os.getenv('DG_COMM_KERNEL_DEBUG', '0') != '0':
            raise ValueError('DG_COMM_KERNEL_DEBUG clears live RDMA epochs and is unsupported')
        layout = RdmaLayout(group.size(), config['experts'], config['tokens'], config['hidden'], config['topk'])
        staging_bytes = layout.num_bytes
        if staging_bytes > int(os.getenv('DG_MEGAMOE_RDMA_MAX_STAGING_BYTES', str(4 << 30))):
            raise ValueError(f'RDMA staging needs {staging_bytes} bytes; increase DG_MEGAMOE_RDMA_MAX_STAGING_BYTES explicitly')
        if _session is not None and _session != (ranks, torch.cuda.current_device()):
            raise ValueError('One EP rank ordering and CUDA device per NVSHMEM process are supported')
        cpp_bytes = torch.ops.deep_gemm.mega_moe_rdma_buffer_size(
            layout.ranks, layout.experts, layout.tokens, layout.hidden, layout.topk)
        if cpp_bytes != staging_bytes:
            raise RuntimeError('DeepGEMM Python/C++ RDMA layout ABI mismatch')
    except (ValueError, RuntimeError) as exc:
        error = str(exc)
    manifest = {
        'error': error,
        'config': (ranks, config, native_bytes, staging_bytes, _C.get_num_sms(),
                   os.getenv('NVSHMEM_SYMMETRIC_SIZE'), os.getenv('NVSHMEM_REMOTE_TRANSPORT')),
    }
    manifests = [None] * group.size()
    dist.all_gather_object(manifests, manifest, group=bootstrap_group)
    validate_manifests(manifests)
    ops = torch.ops.deep_gemm
    if _session is None:
        uid = [ops.mega_moe_rdma_unique_id() if group.rank() == 0 else None]
        dist.broadcast_object_list(uid, src=ranks[0], group=bootstrap_group)
        ops.mega_moe_rdma_init(uid[0], group.rank(), group.size())
        _session = (ranks, torch.cuda.current_device())
    buffer, peers = ops.mega_moe_rdma_allocate(native_bytes)
    mappings = [None] * group.size()
    dist.all_gather_object(mappings, [ptr != 0 for ptr in peers], group=bootstrap_group)
    validate_peer_mappings(mappings)
    staging, _ = ops.mega_moe_rdma_allocate(staging_bytes)
    return buffer, types.SimpleNamespace(buffer_ptrs=peers), staging
