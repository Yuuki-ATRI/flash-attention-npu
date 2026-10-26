"""Exercise the production D>128 dK/dV write path with a host C++ harness."""

import shutil
import subprocess
from pathlib import Path

import pytest


HEADER = Path(__file__).resolve().parents[1] / "csrc/ascend950/flash_attn_npu_3/fag_mmad_dqkv.hpp"


def _method(source, name):
    start = source.index("    void " + name + "(")
    start = source.rfind("    template <", 0, start)
    opening = source.index("{", start)
    depth = 0
    for pos in range(opening, len(source)):
        if source[pos] == "{":
            depth += 1
        elif source[pos] == "}":
            depth -= 1
            if depth == 0:
                return source[start:pos + 1]
    raise AssertionError(f"unterminated {name}")


@pytest.fixture(scope="module")
def large_dim_harness(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required for the host kernel control-flow test")
    source = HEADER.read_text()
    methods = "\n".join(_method(source, name) for name in (
        "ComputeDqDk", "ComputeDv", "ComputeDqDkLargeHeadDim", "ComputeDvLargeHeadDim"))
    cpp = r'''
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>
#define CATLASS_DEVICE
using uint32_t = std::uint32_t;
namespace AscendC { template <class T> using LocalTensor = int; }
namespace Catlass {
struct GemmCoord {
    uint32_t m_, n_, k_;
    uint32_t m() const { return m_; }
    uint32_t n() const { return n_; }
    uint32_t k() const { return k_; }
};
}
using Catlass::GemmCoord;
namespace tla {
struct Coord { uint32_t row, col; };
Coord MakeCoord(uint32_t row, uint32_t col) { return {row, col}; }
Coord MakeShape(uint32_t row, uint32_t col) { return {row, col}; }
}
struct Tensor { uint32_t offset = 0; };
Tensor GetTile(Tensor tensor, tla::Coord coord, tla::Coord) {
    tensor.offset += coord.col;
    return tensor;
}
template <uint32_t Align> uint32_t RoundUp(uint32_t x) {
    return (x + Align - 1) / Align * Align;
}
uint32_t CeilDiv(uint32_t x, uint32_t y) { return (x + y - 1) / y; }
uint32_t Min(uint32_t x, uint32_t y) { return std::min(x, y); }
struct Ascend950FagL1Layout {
    static uint32_t QSlot(uint32_t ping) { return ping; }
    static uint32_t DySlot(uint32_t ping) { return ping; }
};
struct Harness {
    using ElementA = float;
    struct L1AAlignHelper { static constexpr uint32_t M_ALIGNED = 16, K_ALIGNED = 16; };
    struct L1BAlignHelper { static constexpr uint32_t N_ALIGNED = 16; };
    static constexpr uint32_t BASE = 128, SLOT_DQ = 0, SLOT_DQ_PING = 1,
                              SLOT_DK = 2, SLOT_DV = 3;
    int l1Dy[2]{}, l1Q[2]{}, l0CTensorList[4]{};
    struct Write { uint32_t slot, offset; bool atomic; };
    std::vector<Write> writes;
    template <class... Args> void CopyGmToL1B(Args&&...) {}
    template <class... Args> void GemmRow(Args&&...) {}
    template <class... Args> void GemmColA(Args&&...) {}
    template <class... Args> void ComputeDqDkSmallHeadDim(Args&&...) {}
    template <class... Args> void ComputeDvSmallHeadDim(Args&&...) {}
    void FixpipeTla(Tensor& tensor, int, uint32_t, uint32_t, uint32_t,
                    bool atomic, uint32_t slot) {
        writes.push_back({slot, tensor.offset, atomic});
    }
METHODS
};
int main(int argc, char** argv) {
    if (argc != 4) return 2;
    Harness h;
    Tensor k, dq, dk, dv;
    const uint32_t dim = std::strtoul(argv[2], nullptr, 10);
    const bool atomic = std::strtoul(argv[3], nullptr, 10);
    if (argv[1][0] == 'k') {
        h.ComputeDqDk(0, k, dq, dk, {65, dim, 129}, 0,
                      true, true, true, true, atomic);
    } else {
        h.ComputeDv(0, dv, {65, dim, 129}, 0, true, true, atomic);
    }
    for (auto write : h.writes)
        std::cout << write.slot << ":" << write.offset << ":" << write.atomic << " ";
}
'''.replace("METHODS", methods)
    folder = tmp_path_factory.mktemp("fag_large_dim_atomic")
    source_path = folder / "harness.cpp"
    binary = folder / "harness"
    source_path.write_text(cpp)
    subprocess.run([compiler, "-std=c++17", "-O2", str(source_path), "-o", str(binary)],
                   check=True, capture_output=True, text=True)
    return binary


@pytest.mark.parametrize("kind,slot", [("k", 2), ("v", 3)])
@pytest.mark.parametrize("dim", [129, 192, 256])
@pytest.mark.parametrize("atomic", [False, True])
def test_large_dim_gradient_honors_write_mode(large_dim_harness, kind, slot, dim, atomic):
    result = subprocess.run([str(large_dim_harness), kind, str(dim), str(int(atomic))],
                            check=True, capture_output=True, text=True)
    writes = [tuple(map(int, item.split(":"))) for item in result.stdout.split()]
    gradient_writes = [(offset, flag) for out_slot, offset, flag in writes if out_slot == slot]
    assert gradient_writes == [(offset, int(atomic)) for offset in range(0, dim, 128)]
    if kind == "k":
        assert [flag for out_slot, _, flag in writes if out_slot in (0, 1)] == [1] * len(gradient_writes)
