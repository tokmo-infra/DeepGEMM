"""CPU-only validation of the experimental MegaMoE RDMA allocation contract."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RdmaLayout:
    ranks: int
    experts: int
    tokens: int
    hidden: int
    topk: int

    def __post_init__(self):
        if not 1 <= self.ranks <= 72:
            raise ValueError('MegaMoE RDMA supports 1..72 EP ranks')
        if not 0 < self.experts <= 2048 or self.experts % self.ranks:
            raise ValueError('Physical experts must be divisible by EP size')
        if self.experts // self.ranks > 512:
            raise ValueError('At most 512 experts per rank are supported')
        if not 0 < self.tokens <= 1 << 20 or self.tokens % 4:
            raise ValueError('Token capacity must be positive and aligned to four')
        if not 0 < self.hidden <= 1 << 20 or self.hidden % 128:
            raise ValueError('Hidden size must be positive and aligned to 128')
        if not 0 < self.topk <= min(32, self.experts):
            raise ValueError('Invalid top-k')

    @property
    def input_bytes(self):
        return self.tokens * (self.hidden + self.hidden // 32 + self.topk * 12)

    @property
    def output_bytes(self):
        return self.topk * self.tokens * self.hidden * 2

    @property
    def input_offset(self):
        return (1024 + self.experts * self.tokens * 4 + 127) // 128 * 128

    @property
    def output_offset(self):
        return (self.input_offset + self.ranks * self.input_bytes + 127) // 128 * 128

    @property
    def num_bytes(self):
        return self.output_offset + self.ranks * self.output_bytes


def validate_manifests(manifests):
    """Reject asymmetric allocations on every rank before a collective malloc."""
    if not manifests:
        raise ValueError('Empty EP group')
    errors = [(rank, item['error']) for rank, item in enumerate(manifests) if item['error']]
    if errors:
        raise ValueError(f'MegaMoE RDMA configuration errors: {errors}')
    expected = manifests[0]['config']
    mismatched = [rank for rank, item in enumerate(manifests) if item['config'] != expected]
    if mismatched:
        raise ValueError(f'MegaMoE RDMA requires identical allocation/SM configuration; ranks {mismatched} differ')


def validate_peer_mappings(matrix):
    """Count reduction must classify a source/destination pair identically."""
    size = len(matrix)
    if any(len(row) != size for row in matrix):
        raise ValueError('Invalid peer mapping matrix')
    if any(not matrix[i][i] for i in range(size)):
        raise ValueError('Each PE must map its own symmetric allocation')
    if any(matrix[i][j] != matrix[j][i] for i in range(size) for j in range(size)):
        raise ValueError('Asymmetric NVSHMEM direct mappings are unsupported')
