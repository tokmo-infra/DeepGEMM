"""CPU tests for allocation agreement; these do not validate CUDA or RDMA."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'rdma_contract', ROOT / 'deep_gemm/mega/rdma_contract.py')
contract = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = contract
spec.loader.exec_module(contract)
RdmaLayout = contract.RdmaLayout


@pytest.fixture(scope='module')
def layout_binary(tmp_path_factory):
    folder = tmp_path_factory.mktemp('rdma-layout')
    source = folder / 'layout.cpp'
    source.write_text('''
#include <deep_gemm/layout/mega_moe_rdma.cuh>
#include <cstdlib>
#include <iostream>
int main(int argc, char** argv) {
    using deep_gemm::layout::MegaMoERdmaBuffer;
    alignas(128) uint8_t header[1024]{};
    MegaMoERdmaBuffer x(header, std::strtoull(argv[1], nullptr, 10),
        std::strtoull(argv[2], nullptr, 10), std::strtoull(argv[3], nullptr, 10),
        std::strtoull(argv[4], nullptr, 10), std::strtoull(argv[5], nullptr, 10));
    std::cout << x.input_bytes() << " " << x.output_bytes() << " "
              << x.input_offset() << " " << x.output_offset() << " " << x.get_num_bytes() << " "
              << reinterpret_cast<uint8_t*>(x.launch_epoch()) - header << " "
              << reinterpret_cast<uint8_t*>(x.peer_tokens(x.num_ranks)) - header;
}
''')
    binary = folder / 'layout'
    subprocess.run(['c++', '-std=c++17', '-Wall', '-Wextra',
                    '-I', str(ROOT / 'deep_gemm/include'), str(source), '-o', str(binary)], check=True)
    return binary


@pytest.mark.parametrize('shape', [
    (1, 256, 4, 128, 1), (8, 256, 256, 7168, 8),
    (16, 256, 2048, 7168, 8), (24, 384, 8192, 7168, 8),
    (72, 144, 65536, 16384, 32),
])
def test_python_and_compiled_layout_agree_without_32bit_overflow(layout_binary, shape):
    layout = RdmaLayout(*shape)
    result = subprocess.check_output([str(layout_binary), *map(str, shape)], text=True)
    values = list(map(int, result.split()))
    assert values == [
        layout.input_bytes, layout.output_bytes, layout.input_offset,
        layout.output_offset, layout.num_bytes, 1016, shape[0] * 8]
    # The replay counter must be aligned and disjoint from all peer token counts.
    epoch_offset, peer_tokens_end = values[-2:]
    assert peer_tokens_end <= epoch_offset and epoch_offset + 8 <= 1024
    assert epoch_offset % 8 == 0
    assert layout.input_offset >= 1024 + layout.experts * layout.tokens * 4
    assert layout.output_offset >= layout.input_offset + layout.ranks * layout.input_bytes
    assert layout.input_offset % 128 == layout.output_offset % 128 == 0


@pytest.mark.parametrize('shape', [
    (24, 256, 256, 7168, 8), (0, 256, 256, 7168, 8),
    (8, 256, 3, 7168, 8), (8, 256, 256, 7169, 8),
    (8, 256, 256, 7168, 33), (1, 1024, 256, 7168, 8),
])
def test_invalid_layout_rejected_before_collective_allocation(shape):
    with pytest.raises(ValueError):
        RdmaLayout(*shape)


def test_mismatched_rank_allocation_rejected():
    with pytest.raises(ValueError, match='ranks \\[1\\] differ'):
        contract.validate_manifests([
            {'error': '', 'config': (256, 148)}, {'error': '', 'config': (512, 148)}])


def test_single_rank_configuration_error_is_visible_to_all_ranks():
    with pytest.raises(ValueError, match='staging budget exceeded'):
        contract.validate_manifests([
            {'error': '', 'config': (256, 148)},
            {'error': 'staging budget exceeded', 'config': (256, 148)}])


def test_identical_allocation_contract_accepted():
    contract.validate_manifests([{'error': '', 'config': (256, 148)}] * 16)


def test_asymmetric_direct_mapping_cannot_double_count_or_miss_a_source():
    with pytest.raises(ValueError, match='Asymmetric'):
        contract.validate_peer_mappings([[True, True], [False, True]])
    contract.validate_peer_mappings([[True, False], [False, True]])
