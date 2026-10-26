"""CPU checks of actual backward detection/wrappers/fakes without loading NPU extensions."""
import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]


def _load(version, suffix, fake=False):
    source = ROOT / f"flash_attn_npu_{version}/flash_attn_npu_interface{suffix}.py"
    tree = ast.parse(source.read_text())
    name = "_flash_attn_backward" if version == 3 else "_flash_attn_backward_op"
    if fake:
        name += "_fake"
    names = {name, "_detect_layout", "_transposed_bnsd_view", "maybe_contiguous", "_maybe_contiguous", "_full_contiguous"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    for node in nodes:
        node.decorator_list = []
    captured = []

    def bwd(*args):
        captured.append(args)
        return args[6], args[7], args[8], torch.empty(2, 4, 17)

    namespace = {"torch": torch, "Optional": Optional, "Tuple": Tuple,
                 "_LAYOUT_TND": 0, "_LAYOUT_BSND": 1, "_LAYOUT_BNSD": 2,
                 f"flash_attn_npu_{version}{suffix}": SimpleNamespace(bwd=bwd),
                 "maybe_contiguous": lambda x: x, "_maybe_contiguous": lambda x: x}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace[name], captured


def _arguments(version, suffix, storage):
    if storage == "tnd":
        q = torch.empty(34, 4, 64)
        cu = torch.tensor([0, 17, 34], dtype=torch.int32)
        lse = torch.empty(4, 34)
    else:
        q = torch.empty(2, 17, 4, 64)
        cu, lse = None, torch.empty(2, 4, 17)
        if storage in ("bnsd", "padded"):
            physical = torch.empty(2, 4, 17 if storage == "bnsd" else 23, 64)
            q = physical[:, :, :17].transpose(1, 2)
    grad = torch.empty(q.shape)
    kwargs = dict(dout=q, q=q, k=q, v=q, out=q, softmax_lse=lse,
                  cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=17, max_seqlen_k=17,
                  dq=grad, dk=grad, dv=grad, softmax_scale=None, window_size_left=-1,
                  window_size_right=-1, softcap=0.0, deterministic=False)
    used = "sequed" if version == 3 and suffix == "" else "seqused"
    kwargs.update({f"{used}_q": None, f"{used}_k": None})
    kwargs["is_causal" if version == 3 else "causal"] = False
    return kwargs


@pytest.mark.parametrize("version", [3, 4])
@pytest.mark.parametrize("suffix", ["", "_950"])
@pytest.mark.parametrize("storage", ["bsnd", "bnsd", "tnd", "padded"])
def test_backward_detects_storage_internally(version, suffix, storage):
    fn, captured = _load(version, suffix)
    kwargs = _arguments(version, suffix, storage)
    fn(**kwargs)
    args = captured[0]
    assert len(args) == 23
    assert args[-1] == {"bsnd": 1, "bnsd": 2, "tnd": 0, "padded": 1}[storage]
    # Detection changes dispatch, never copies/reinterprets the logical input or its strides.
    for index, name in ((0, "dout"), (1, "q"), (2, "k"), (3, "v"), (4, "out")):
        assert args[index] is kwargs[name]
    assert args[6].shape == kwargs["q"].shape
    assert args[6].is_contiguous()
    assert "layout" not in __import__("inspect").signature(fn).parameters


@pytest.mark.parametrize("version", [3, 4])
@pytest.mark.parametrize("suffix", ["", "_950"])
@pytest.mark.parametrize("storage", ["bsnd", "bnsd", "tnd", "padded"])
def test_backward_fake_logical_shape(version, suffix, storage):
    fn, _ = _load(version, suffix, fake=True)
    result = fn(**_arguments(version, suffix, storage))
    expected = (4, 34) if version == 3 and suffix == "_950" and storage == "tnd" else (2, 4, 17)
    assert result.shape == expected
    assert result.dtype == torch.float32
