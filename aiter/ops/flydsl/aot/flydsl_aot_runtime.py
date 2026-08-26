#!/usr/bin/env python3
"""Launch a FlyDSL kernel from an AOT `.co`, with no MLIR and no FlyDSL in the path.

The JIT path parses the cached IR, hands it to an MLIR `ExecutionEngine` to build the
host launcher, and lets MLIR's GPU runtime `hipModuleLoad` the embedded object. This
module keeps only the last part: it `hipModuleLoadData`s the extracted object and
launches it directly, so nothing on the serving path needs LLVM, MLIR, or the kernel
source.

Two details are easy to get wrong and both are silent:

  * **Argument widths.** aiter's existing `hsaco_launcher` packs every Python `int` as
    `c_int32`, which truncates the 64-bit pointers these kernels take. Packing here is
    driven by the recorded `wrapper_arg_types`, not by the Python type.
  * **Which arguments the kernel gets.** The kernel's argument list is a subset of the
    launcher's, in order, with the grid-carrying scalars left out -- for a8w4 stage2 the
    launcher takes 15 arguments and the kernel takes 13. Assuming they match shifts every
    argument.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import os

# ctypes against the HIP runtime rather than hip-python: the serving image does not
# ship hip-python, and going straight at libamdhip64 keeps the runtime dependency of
# an AOT drop down to the HIP runtime that is there anyway.
_LIB = None


def _hip():
    global _LIB
    if _LIB is None:
        path = ctypes.util.find_library("amdhip64") or "libamdhip64.so"
        _LIB = ctypes.CDLL(path)
        _LIB.hipModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        _LIB.hipModuleLoadData.restype = ctypes.c_int
        _LIB.hipModuleGetFunction.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        _LIB.hipModuleGetFunction.restype = ctypes.c_int
        _LIB.hipModuleLaunchKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        _LIB.hipModuleLaunchKernel.restype = ctypes.c_int
        _LIB.hipGetErrorString.argtypes = [ctypes.c_int]
        _LIB.hipGetErrorString.restype = ctypes.c_char_p
    return _LIB


def _hip_check(err: int, what: str):
    if err != 0:
        msg = _hip().hipGetErrorString(err)
        raise RuntimeError(f"{what} failed: {msg.decode() if msg else err}")


def _eval_dim(node, args) -> int:
    """Evaluate one recorded grid/block dimension against the launcher's arguments."""
    if "const" in node:
        return int(node["const"])
    if "arg" in node:
        return int(args[node["arg"]])
    op = node["op"]
    if op == "select":
        return (
            _eval_dim(node["lhs"], args)
            if _eval_dim(node["cond"], args)
            else _eval_dim(node["rhs"], args)
        )
    lhs, rhs = _eval_dim(node["lhs"], args), _eval_dim(node["rhs"], args)
    if op.startswith("icmp."):
        pred = op.split(".", 1)[1]
        cmp = {
            "eq": lhs == rhs,
            "ne": lhs != rhs,
            "slt": lhs < rhs,
            "sle": lhs <= rhs,
            "sgt": lhs > rhs,
            "sge": lhs >= rhs,
            "ult": lhs < rhs,
            "ule": lhs <= rhs,
            "ugt": lhs > rhs,
            "uge": lhs >= rhs,
        }.get(pred)
        if cmp is None:
            raise RuntimeError(f"unsupported icmp predicate {pred!r}")
        return int(cmp)
    if op == "add":
        return lhs + rhs
    if op == "sub":
        return lhs - rhs
    if op == "mul":
        return lhs * rhs
    if op in ("udiv", "sdiv"):
        # C semantics: sdiv truncates toward zero, which differs from Python's floor
        # division for negative operands. The ceil-div idiom these launchers emit relies
        # on truncation, so getting this wrong would shift the grid by one block.
        q = abs(lhs) // abs(rhs)
        return q if (lhs >= 0) == (rhs >= 0) else -q
    if op in ("urem", "srem"):
        r = abs(lhs) % abs(rhs)
        return r if lhs >= 0 else -r
    if op == "and":
        return lhs & rhs
    if op == "or":
        return lhs | rhs
    if op == "xor":
        return lhs ^ rhs
    if op == "shl":
        return lhs << rhs
    if op in ("lshr", "ashr"):
        return lhs >> rhs
    raise RuntimeError(f"unsupported grid op {op!r}")


def _unwrap(value):
    """Reduce a launcher argument to a plain int/float.

    aiter hands FlyDSL its own wrappers rather than raw values: tensors arrive as
    `PointerJitArg` (holding a `ctypes.c_void_p`) and the stream as a `Stream`. Passing
    those straight to ctypes raises; passing their `id()` would not.
    """
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, ctypes.c_void_p):
        return value.value or 0
    for attr in ("pointer", "cuda_stream", "value", "ptr", "handle"):
        if hasattr(value, attr):
            return _unwrap(getattr(value, attr))
    raise RuntimeError(f"cannot unwrap launcher argument of type {type(value).__name__}")


_SCALAR_RE = None


def _innermost_scalar(mlir_type: str) -> str:
    """`!llvm.struct<packed (struct<packed (i32)>)>` -> `i32`.

    The dense GEMM launchers wrap single scalars in nested structs. The code object
    declares such an argument as a 4-byte `by_value`, so only its scalar content matters;
    what must not happen is treating the wrapper as an opaque 8-byte blob, which shifts
    every following argument.
    """
    global _SCALAR_RE
    if _SCALAR_RE is None:
        import re

        _SCALAR_RE = re.compile(r"\b(i1|i8|i16|i32|i64|f16|f32|f64|bf16)\b")
    m = _SCALAR_RE.findall(mlir_type)
    return m[-1] if m else "i32"


def _pack_slot(value, mlir_type: str, slot: dict):
    """One launcher argument -> the exact bytes for the slot the code object declares."""
    value = _unwrap(value)
    sz, kind = slot["size"], slot.get("value_kind")
    if kind == "global_buffer" or mlir_type.startswith("!llvm.ptr"):
        if sz != 8:
            raise RuntimeError(f"指针参数落在 {sz} 字节的槽位上，会被截断")
        return ctypes.c_uint64(int(value))
    scalar = mlir_type.strip() if "struct" not in mlir_type else _innermost_scalar(mlir_type)
    is_float = scalar in ("f16", "f32", "f64", "bf16") or isinstance(value, float)
    if sz == 4:
        return ctypes.c_float(float(value)) if is_float else ctypes.c_int32(int(value))
    if sz == 8:
        return ctypes.c_double(float(value)) if is_float else ctypes.c_uint64(int(value))
    if sz == 2:
        return ctypes.c_int16(int(value))
    if sz == 1:
        return ctypes.c_int8(int(value))
    raise RuntimeError(f"不支持的槽位大小 {sz}")


def _pack(value, mlir_type: str):
    """One launcher argument -> a ctypes cell of the width the kernel expects."""
    value = _unwrap(value)
    t = mlir_type.strip()
    if t.startswith("!llvm.ptr") or t == "i64":
        return ctypes.c_uint64(int(value))
    if t == "i32":
        return ctypes.c_int32(int(value))
    if t == "i16":
        return ctypes.c_int16(int(value))
    if t == "i8" or t == "i1":
        return ctypes.c_int8(int(value))
    if t == "f32":
        return ctypes.c_float(float(value))
    if t == "f64":
        return ctypes.c_double(float(value))
    raise RuntimeError(f"unsupported launcher arg type {mlir_type!r}")


class AotKernel:
    """One `.co` plus its launch facts, loaded on first use."""

    def __init__(self, entry: dict, co_path: str):
        self.entry = entry
        self.co_path = co_path
        # Keyed by device: a HIP module belongs to one context, so a handle cached
        # globally works on whichever GPU happened to load it first and is invalid on
        # the rest. aiter's own ASM loader has this shape (`SynchronizedCache` keyed by
        # symbol name alone) and prefill runs 8-way pipeline parallel, so a global cache
        # here would break on seven of eight ranks.
        self._per_device: dict[int, tuple] = {}

    def _device(self) -> int:
        d = ctypes.c_int()
        _hip_check(_hip().hipGetDevice(ctypes.byref(d)), "hipGetDevice")
        return d.value

    def _ensure(self):
        dev = self._device()
        got = self._per_device.get(dev)
        if got is not None:
            self._func = got[1]
            return
        lib = _hip()
        with open(self.co_path, "rb") as fh:
            data = fh.read()
        self._check_lds(dev)
        blob = ctypes.create_string_buffer(data, len(data))
        mod = ctypes.c_void_p()
        _hip_check(
            lib.hipModuleLoadData(ctypes.byref(mod), ctypes.cast(blob, ctypes.c_void_p)),
            "hipModuleLoadData",
        )
        fn = ctypes.c_void_p()
        _hip_check(
            lib.hipModuleGetFunction(ctypes.byref(fn), mod, self.entry["kernel"].encode()),
            f"hipModuleGetFunction({self.entry['kernel']})",
        )
        # The blob is kept alive alongside the module: HIP does not document copying it.
        self._per_device[dev] = (mod, fn, blob)
        self._func = fn

    def _check_lds(self, dev: int):
        """Refuse a kernel whose static LDS exceeds this device's limit.

        aiter's ASM loader does the same thing (`validate_hsaco_lds`) before handing a
        code object to HIP, because the failure otherwise surfaces at launch as an opaque
        error rather than at load with the two numbers that explain it.
        """
        need = self._abi()["group_segment_fixed_size"]
        HIP_ATTR_MAX_SHARED_PER_BLOCK = 74
        have = ctypes.c_int()
        rc = _hip().hipDeviceGetAttribute(
            ctypes.byref(have), HIP_ATTR_MAX_SHARED_PER_BLOCK, dev
        )
        if rc != 0 or have.value <= 0:
            return  # attribute unavailable: skip rather than block a valid launch
        if need > have.value:
            raise RuntimeError(
                f"{self.entry['kernel']} 需要 {need} 字节 LDS，"
                f"设备 {dev} 上限 {have.value} 字节"
            )

    def _abi(self):
        """The kernel's own ABI declaration, read once from the code object."""
        if getattr(self, "_abi_cache", None) is None:
            try:
                from .co_abi import read_co
            except ImportError:
                from co_abi import read_co

            info = read_co(self.co_path)
            md = next(
                (k for k in info["kernels"] if k["name"] == self.entry["kernel"]), None
            )
            if md is None:
                raise RuntimeError(
                    f"{self.entry['kernel']} 不在 {self.co_path} 的元数据里"
                )
            hidden = [
                a for a in md["args"] if str(a["value_kind"]).startswith("hidden_")
            ]
            if hidden:
                raise RuntimeError(
                    f"{self.entry['kernel']} 声明了隐藏参数 "
                    f"{[a['value_kind'] for a in hidden]}，宿主打包未支持"
                )
            if len(md["args"]) != len(self.entry["kernel_args"]):
                raise RuntimeError(
                    f"{self.entry['kernel']} 参数个数不符：元数据 {len(md['args'])} "
                    f"vs index {len(self.entry['kernel_args'])}"
                )
            self._abi_cache = md
        return self._abi_cache

    def _kernarg_size(self) -> int:
        return int(self._abi()["kernarg_segment_size"])

    def _arg_slots(self) -> list:
        return self._abi()["args"]

    def __call__(self, *launcher_args, stream=None, grid_override=None):
        self._ensure()
        e = self.entry
        plain = [_unwrap(a) if not isinstance(a, (int, float)) else a for a in launcher_args]
        grid = list(grid_override) if grid_override else [
            _eval_dim(d, plain) for d in e["grid"]
        ]
        block = [_eval_dim(d, plain) for d in e["block"]]

        # Build the kernarg segment explicitly and pass it through `extra` rather than
        # letting HIP assemble it from `kernelParams`, which faulted on these kernels.
        #
        # The layout comes from the code object's own declaration, not from re-deriving
        # alignment off the MLIR types. Re-deriving happens to agree for the MoE kernels
        # but disagrees for 502 of the 1330 dense-GEMM artifacts, whose launchers take
        # aggregates -- and disagreeing here does not crash, it reads every later argument
        # from the wrong offset. The MLIR type is still consulted for one thing the
        # metadata does not record: whether a 4-byte by_value is an integer or a float.
        buf = bytearray(self._kernarg_size())
        for slot, i in zip(self._arg_slots(), e["kernel_args"]):
            t = e["wrapper_arg_types"][i].strip()
            off, sz = slot["offset"], slot["size"]
            cell = bytes(_pack_slot(launcher_args[i], t, slot))
            buf[off : off + sz] = cell[:sz]
        kernarg = ctypes.create_string_buffer(bytes(buf), len(buf))
        size = ctypes.c_size_t(len(buf))
        HIP_LAUNCH_PARAM_BUFFER_POINTER = ctypes.c_void_p(1)
        HIP_LAUNCH_PARAM_BUFFER_SIZE = ctypes.c_void_p(2)
        HIP_LAUNCH_PARAM_END = ctypes.c_void_p(3)
        extra = (ctypes.c_void_p * 5)(
            HIP_LAUNCH_PARAM_BUFFER_POINTER,
            ctypes.cast(kernarg, ctypes.c_void_p),
            HIP_LAUNCH_PARAM_BUFFER_SIZE,
            ctypes.cast(ctypes.byref(size), ctypes.c_void_p),
            HIP_LAUNCH_PARAM_END,
        )

        if stream is None and e.get("stream_arg") is not None:
            stream = launcher_args[e["stream_arg"]]
        raw = _unwrap(stream) if stream is not None else 0

        # sharedMemBytes is the *dynamic* LDS added on top of what the kernel already
        # declares. `group_segment_fixed_size` (33280 for a8w4 stage2) is the static
        # amount and is baked into the code object; passing it here asks for it twice.
        # The launcher IR carries no dynamic_shared_memory_size operand, so it is 0.
        _hip_check(
            _hip().hipModuleLaunchKernel(
                self._func,
                grid[0],
                grid[1],
                grid[2],
                block[0],
                block[1],
                block[2],
                e.get("dynamic_shared_bytes", 0),
                ctypes.c_void_p(int(raw)),
                None,
                extra,
            ),
            f"hipModuleLaunchKernel({e['kernel'][-40:]})",
        )

    def __repr__(self):
        e = self.entry
        return f"<AotKernel {e['kernel'][-48:]} {e['co_sha256_12']} lds={e['shared_bytes']}>"


class AotDrop:
    """An AOT directory: `index.json` plus the `.co` files it names."""

    def __init__(self, drop_dir: str):
        self.dir = drop_dir
        with open(os.path.join(drop_dir, "index.json")) as fh:
            self.index = json.load(fh)["kernels"]
        self._loaded: dict[str, AotKernel] = {}

    def names(self) -> list[str]:
        return sorted(self.index)

    def by_source(self, src_dir: str, src_pkl: str):
        """Resolve by the cache entry the artifact was built from -- the only key that
        identifies one compilation exactly."""
        for entries in self.index.values():
            for e in entries:
                if e.get("src_dir") == src_dir and e.get("src_pkl") == src_pkl:
                    k = e["co"]
                    if k not in self._loaded:
                        self._loaded[k] = AotKernel(e, os.path.join(self.dir, e["co"]))
                    return self._loaded[k]
        return None

    def get(self, kernel: str, *, sha12: str | None = None, shared_bytes: int | None = None):
        """Resolve one kernel.

        A symbol name can map to several distinct objects (measured: a8w4 stage1 emits
        the same name at LDS 33792 and 41088), so an ambiguous name has to be narrowed
        rather than resolved by order -- picking the wrong one loads a kernel whose LDS
        request does not match what it was built for.
        """
        cands = self.index[kernel]
        if sha12 is not None:
            cands = [c for c in cands if c["co_sha256_12"] == sha12]
        if shared_bytes is not None:
            cands = [c for c in cands if c["shared_bytes"] == shared_bytes]
        if len(cands) != 1:
            raise RuntimeError(
                f"{kernel!r} resolves to {len(cands)} objects; narrow with sha12= or "
                f"shared_bytes= (available: "
                f"{[(c['co_sha256_12'], c['shared_bytes']) for c in self.index[kernel]]})"
            )
        e = cands[0]
        k = e["co"]
        if k not in self._loaded:
            self._loaded[k] = AotKernel(e, os.path.join(self.dir, e["co"]))
        return self._loaded[k]


if __name__ == "__main__":
    import sys

    drop = AotDrop(sys.argv[1] if len(sys.argv) > 1 else "drop")
    print(f"  drop      {drop.dir}")
    for n in drop.names():
        for e in drop.index[n]:
            print(f"    {n[-56:]:<56} {e['co_sha256_12']}  lds={e['shared_bytes']:<6} {e['co']}")
