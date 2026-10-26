"""CPU regression for the actual 950 deterministic cast UB planner."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def planner(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required for the host UB planner test")
    folder = tmp_path_factory.mktemp("fag_cast_ub")
    source = folder / "plan.cpp"
    source.write_text('''
#include <cstdlib>
#include <iostream>
#include "fag_cast_ub_plan.hpp"
int main(int argc, char** argv) {
    if (argc != 6) return 2;
    auto p = FAGTiling950::PlanBn2s2CastUb(
        std::strtoull(argv[1], nullptr, 10), std::strtoull(argv[2], nullptr, 10),
        std::strtoull(argv[3], nullptr, 10), std::strtoull(argv[4], nullptr, 10),
        std::strtoull(argv[5], nullptr, 10));
    std::cout << p.chunkRows << " " << p.pipelineBytes << " "
              << p.inputBytes << " " << p.outputBytes;
}
''')
    binary = folder / "plan"
    subprocess.run([compiler, "-std=c++17", "-O2", "-I",
                    str(ROOT / "csrc/ascend950/flash_attn_npu_3"),
                    str(source), "-o", str(binary)], check=True, capture_output=True, text=True)

    def run(qtile, kvtile, qdim, vdim, ub):
        result = subprocess.run([str(binary), *map(str, (qtile, kvtile, qdim, vdim, ub))],
                                check=True, capture_output=True, text=True)
        return tuple(map(int, result.stdout.split()))
    return run


@pytest.mark.parametrize("qdim,vdim,rows", [
    (32, 32, 32), (64, 64, 32), (80, 80, 32), (128, 128, 32),
    (192, 192, 27), (256, 256, 20), (64, 256, 20), (256, 64, 20),
    (65, 81, 32),
])
def test_cast_plan_fits_device_ub(planner, qdim, vdim, rows):
    chunk, pipeline, inputs, outputs = planner(128, 128, qdim, vdim, 253952)
    assert chunk == rows
    assert pipeline == 222208
    assert inputs % 32 == 0 and outputs % 32 == 0
    assert 253952 - inputs - outputs >= pipeline


def test_d256_failure_regression(planner):
    chunk, pipeline, inputs, outputs = planner(128, 128, 256, 256, 253952)
    assert pipeline + 32 * 256 * 6 > 253952  # Original fixed 32-row allocation fails.
    assert (chunk, inputs, outputs) == (20, 20480, 10240)
    assert pipeline + inputs + outputs == 252928


@pytest.mark.parametrize("ub,rows", [
    (222207, 0), (222208, 0), (222208 + 1535, 0),
    (222208 + 1536, 1), (222208 + 49152, 32),
])
def test_cast_plan_capacity_boundaries(planner, ub, rows):
    chunk, pipeline, inputs, outputs = planner(128, 128, 256, 256, ub)
    assert chunk == rows
    if chunk:
        assert pipeline + inputs + outputs <= ub
    else:
        assert inputs == outputs == 0


def test_cast_plan_respects_tile_rows(planner):
    assert planner(128, 16, 256, 256, 253952)[0] == 16


@pytest.mark.parametrize("qtile,kvtile,qdim,vdim", [(0, 128, 64, 64), (128, 0, 64, 64),
                                                  (128, 128, 0, 64), (128, 128, 64, 0)])
def test_cast_plan_rejects_empty_geometry(planner, qtile, kvtile, qdim, vdim):
    assert planner(qtile, kvtile, qdim, vdim, 253952)[0] == 0
