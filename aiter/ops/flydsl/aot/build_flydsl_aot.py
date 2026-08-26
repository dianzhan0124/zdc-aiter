#!/usr/bin/env python3
"""Turn a FlyDSL JIT cache into an AOT drop: one `.co` plus one `.json` per kernel.

FlyDSL's ROCm pipeline already ends in `gpu-module-to-binary{format=fatbin}`, so a
cache entry carries a fully linked device object -- the JIT does not rebuild device
code at load, it `hipModuleLoad`s what is already there. What the cache does *not*
give is a form anyone can ship: the entry is a pickle of
`flydsl.compiler.jit_executor.CompiledArtifact`, which

  * only unpickles against the FlyDSL version that wrote it, and
  * carries `source_ir`, the pre-lowering IR complete with
    `loc("…/gemm2.py":443:0)` -- file paths and line numbers of the kernel source.

This script keeps the device object and the launch facts, and drops everything else.

WHAT IS EXTRACTED

The host side of a cache entry is one small function:

    llvm.func @launch_gemm2(%arg0: i64, …, %arg9: i32, %arg10: i64, %arg11: !llvm.ptr) {
      %0 = llvm.mlir.constant(256 : index) : i64
      %1 = llvm.mlir.constant(1 : index) : i64
      %2 = llvm.sext %arg9 : i32 to i64
      gpu.launch_func <%arg11> @kernels::@gemm2_… blocks in (%2, %1, %1)
          threads in (%0, %1, %1) args(%arg0, …, %arg8, %arg10)
    }

Everything a loader needs is in it, and mechanically so: the grid and block come
from SSA values that resolve to constants or to wrapper arguments, the kernel's
argument list is a subset of the wrapper's arguments in order, and the stream is the
async-object operand. Note `%arg9` above: it feeds the grid and is *not* passed to
the kernel, so a loader cannot assume "kernel args == wrapper args".

Grid values are resolved through a small SSA walk rather than a regex, because the
grid is not always a bare argument -- persist_m and split-K kernels compute it with
mul/div/add chains inside the wrapper.

Usage:
    python3 build_flydsl_aot.py <cache_dir> -o <out_dir> [--arch gfx950]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import sys

_HEX = set("0123456789ABCDEFabcdef")


def unescape(text: str) -> bytes:
    """Decode an MLIR string attribute.

    MLIR writes these with `llvm::printEscapedString`, which emits `\\\\` for a
    backslash, `\\"` for a quote, and `\\XX` hex for anything unprintable. All three
    have to be taken before the hex branch, and the quote case is the one that bites:
    a code object containing byte 0x22 comes back one byte longer, every later byte
    shifted. The ELF header and section table still parse -- the damage is inside
    `.text` -- so the only symptom is the kernel faulting on wild addresses at launch.
    """
    out = bytearray()
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "\\" and i + 1 < n:
            nxt = text[i + 1]
            if nxt == "\\":
                out.append(0x5C)
                i += 2
                continue
            if nxt == '"':
                out.append(0x22)
                i += 2
                continue
            if i + 2 < n and nxt in _HEX and text[i + 2] in _HEX:
                out.append(int(text[i + 1 : i + 3], 16))
                i += 3
                continue
        out.append(ord(c))
        i += 1
    return bytes(out)


def elf_expected_size(data: bytes) -> int:
    """Size implied by the ELF header, used to prove the decode was lossless."""
    if len(data) < 64 or data[:4] != b"\x7fELF":
        return -1
    e_shoff = int.from_bytes(data[0x28:0x30], "little")
    e_shentsize = int.from_bytes(data[0x3A:0x3C], "little")
    e_shnum = int.from_bytes(data[0x3C:0x3E], "little")
    return e_shoff + e_shentsize * e_shnum


class WrapperParse:
    """The launch facts of one cache entry, read off the host wrapper."""

    def __init__(self, ir: str):
        self.ir = ir
        self.wrapper_args = self._wrapper_args()
        self._ssa = self._constants()
        launch = self._launch_line()
        self.kernel = re.search(r"@kernels::@([A-Za-z0-9_.]+)", launch).group(1)
        self.grid = self._dims(launch, "blocks in")
        self.block = self._dims(launch, "threads in")
        self.kernel_args = self._kernel_args(launch)
        m = re.search(r"gpu\.launch_func\s+<%arg(\d+)", launch)
        self.stream_arg = int(m.group(1)) if m else None

    # -- pieces -----------------------------------------------------------------

    def _launch_line(self) -> str:
        for line in self.ir.split("\n"):
            if "gpu.launch_func" in line:
                return line.strip()
        raise SystemExit("no gpu.launch_func: this artifact has no host launcher")

    def _wrapper_args(self) -> list[str]:
        m = re.search(r"llvm\.func @(\w+)\(", self.ir)
        if not m:
            raise SystemExit("no llvm.func wrapper in this artifact")
        self.wrapper_name = m.group(1)
        params = self._balanced(self.ir, m.end() - 1)
        types = []
        for part in self._split_top(params):
            part = part.strip()
            if ":" in part:
                types.append(part.split(":", 1)[1].strip())
        return types

    @staticmethod
    def _balanced(text: str, open_idx: int) -> str:
        """Text inside the parenthesis at `open_idx`, respecting nesting.

        `[^)]*` stops at the first `)`, which truncates a signature whose parameters are
        aggregates -- the dense GEMM launchers take `!llvm.struct<packed (struct<packed
        (i32)>)>`. Truncating there yielded two garbage types instead of fourteen, and the
        recorded kernel-argument indices then pointed past the end of the list.
        """
        depth = 0
        for i in range(open_idx, len(text)):
            c = text[i]
            if c in "(<":
                depth += 1
            elif c in ")>":
                depth -= 1
                if depth == 0:
                    return text[open_idx + 1 : i]
        raise SystemExit("unbalanced wrapper signature")

    @staticmethod
    def _split_top(params: str) -> list[str]:
        """Split a parameter list on commas that are not inside brackets."""
        out, depth, start = [], 0, 0
        for i, c in enumerate(params):
            if c in "(<[":
                depth += 1
            elif c in ")>]":
                depth -= 1
            elif c == "," and depth == 0:
                out.append(params[start:i])
                start = i + 1
        tail = params[start:]
        if tail.strip():
            out.append(tail)
        return out

    def _constants(self) -> dict[str, int]:
        """`%N = llvm.mlir.constant(V : ...)` -> {N: V}.

        i1 constants print as bare `true`/`false` with no type suffix, and the signed
        ceil-div idiom compares against one of them, so they have to be recognised too.
        """
        out = {}
        for m in re.finditer(r"%(\d+) = llvm\.mlir\.constant\((-?\d+) : [^)]*\)", self.ir):
            out[m.group(1)] = int(m.group(2))
        for m in re.finditer(r"%(\d+) = llvm\.mlir\.constant\((true|false)\)", self.ir):
            out[m.group(1)] = 1 if m.group(2) == "true" else 0
        return out

    def _resolve(self, ssa: str):
        """Resolve one SSA name to {'const': v} or {'arg': i} or an expression tree.

        Only the forms FlyDSL's launchers actually emit are handled; anything else
        raises rather than being guessed at, because a silently wrong grid produces
        a kernel that runs and returns wrong answers.
        """
        ssa = ssa.strip().lstrip("%")
        if ssa.startswith("arg"):
            return {"arg": int(ssa[3:])}
        if ssa in self._ssa:
            return {"const": self._ssa[ssa]}

        # %N = llvm.sext %argK : i32 to i64   (or zext/trunc)
        m = re.search(rf"%{ssa} = llvm\.(sext|zext|trunc) (%[\w]+)", self.ir)
        if m:
            return self._resolve(m.group(2))

        # %N = llvm.select %c, %a, %b : i1, T
        m = re.search(rf"%{ssa} = llvm\.select (%?[\w]+), (%?[\w]+), (%?[\w]+)", self.ir)
        if m:
            return {
                "op": "select",
                "cond": self._resolve(m.group(1)),
                "lhs": self._resolve(m.group(2)),
                "rhs": self._resolve(m.group(3)),
            }

        # %N = llvm.icmp "pred" %a, %b : T
        m = re.search(rf'%{ssa} = llvm\.icmp "(\w+)" (%?[\w]+), (%?[\w]+)', self.ir)
        if m:
            return {
                "op": f"icmp.{m.group(1)}",
                "lhs": self._resolve(m.group(2)),
                "rhs": self._resolve(m.group(3)),
            }

        # The dense GEMM launchers spell ceil-division out as
        # `sdiv; mul; icmp ne; icmp slt; icmp ne(...,false); and; add 1; select`, so the
        # comparison and boolean ops have to be understood as well -- not just arithmetic.
        m = re.search(
            rf"%{ssa} = llvm\.(add|sub|mul|udiv|sdiv|urem|srem|and|or|xor|shl|lshr|ashr)"
            rf" (%?[\w]+), (%?[\w]+)",
            self.ir,
        )
        if m:
            return {
                "op": m.group(1),
                "lhs": self._resolve(m.group(2)),
                "rhs": self._resolve(m.group(3)),
            }
        raise SystemExit(f"cannot resolve %{ssa} in the launcher; extend _resolve")

    def _dims(self, launch: str, which: str) -> list:
        m = re.search(rf"{which} \(([^)]*)\)", launch)
        if not m:
            raise SystemExit(f"no '{which}' in gpu.launch_func")
        return [self._resolve(p) for p in m.group(1).split(",")]

    def _kernel_args(self, launch: str) -> list[int]:
        m = re.search(r"args\((.*)\)\s*$", launch)
        if not m:
            return []
        idx = []
        for part in m.group(1).split(","):
            a = re.search(r"%arg(\d+)", part)
            if a:
                idx.append(int(a.group(1)))
        return idx

    # -- device object and metadata --------------------------------------------

    def code_object(self) -> bytes:
        start = self.ir.find("#gpu.object<")
        if start < 0:
            raise SystemExit("no #gpu.object: the binary pass did not run")
        end = self.ir.find("\n", start)
        seg = self.ir[start : end if end > 0 else len(self.ir)]
        o = seg.find('bin = "')
        t = seg.rfind('">]')
        if o < 0 or t <= o:
            raise SystemExit('could not locate bin = "..."')
        return unescape(seg[o + len('bin = "') : t])

    def kernel_metadata(self) -> dict:
        out = {}
        for key in (
            "group_segment_fixed_size",
            "private_segment_fixed_size",
            "max_flat_workgroup_size",
            "vgpr_count",
            "sgpr_count",
            "wavefront_size",
        ):
            m = re.search(rf"{key} = (\d+)", self.ir)
            if m:
                out[key] = int(m.group(1))
        m = re.search(r'#rocdl\.target<chip = "([^"]+)"', self.ir)
        if m:
            out["chip"] = m.group(1)
        m = re.search(r"!llvm\.func<void \(([^)]*)\)>", self.ir)
        if m:
            out["kernel_arg_types"] = [t.strip() for t in m.group(1).split(",")]
        return out


def build_one(pkl_path: str, out_dir: str) -> dict | None:
    with open(pkl_path, "rb") as fh:
        art = pickle.load(fh)
    ir = art.ir
    if "gpu.launch_func" not in ir:
        return None
    w = WrapperParse(ir)
    co = w.code_object()
    if co[:4] != b"\x7fELF":
        raise SystemExit(f"{pkl_path}: extracted payload is not an ELF")
    # A mis-decoded escape leaves a loadable ELF whose .text is shifted, so check the
    # size the header implies instead of trusting that it parsed.
    want = elf_expected_size(co)
    if want != len(co):
        raise SystemExit(
            f"{pkl_path}: decoded {len(co)} bytes but the ELF header implies {want}; "
            f"the escape decoding is lossy"
        )

    md = w.kernel_metadata()
    # The symbol name is NOT a unique key: two cache entries can carry the same name
    # and different code (measured on a8w4 stage1 -- same
    # `t32x128x256_pm1_fp8q_sort_async_gui_situv2...` name, LDS 33792 vs 41088, because
    # the LDS budget depends on a parameter the name does not carry). Writing by name
    # alone silently drops one, which is the AOT-layout form of the same-name-different-
    # code hazard the JIT cache has. Disambiguate by content and record it.
    digest = hashlib.sha256(co).hexdigest()[:12]
    co_name = f"{w.kernel}.{digest}.co"
    with open(os.path.join(out_dir, co_name), "wb") as fh:
        fh.write(co)

    entry = {
        "kernel": w.kernel,
        "co": co_name,
        "co_sha256_12": digest,
        # Bind the artifact to the cache entry it came from. FlyDSL's cache path is
        # `{root}/{func_name}_{manager_key}/{sha256(cache_key)[:16]}.pkl`, so this pair
        # identifies one compilation exactly -- which the symbol name does not (§ two
        # stage1 kernels share a name here). Resolving by name is what made an earlier
        # run launch the wrong object with valid-looking arguments.
        "src_dir": os.path.basename(os.path.dirname(pkl_path)),
        "src_pkl": os.path.basename(pkl_path),
        "wrapper": w.wrapper_name,
        "wrapper_arg_types": w.wrapper_args,
        "kernel_args": w.kernel_args,
        "stream_arg": w.stream_arg,
        "grid": w.grid,
        "block": w.block,
        "shared_bytes": md.get("group_segment_fixed_size", 0),
        "chip": md.get("chip"),
        "vgpr_count": md.get("vgpr_count"),
        "sgpr_count": md.get("sgpr_count"),
        "kernel_arg_types": md.get("kernel_arg_types"),
        "co_bytes": len(co),
    }
    with open(os.path.join(out_dir, f"{w.kernel}.{digest}.json"), "w") as fh:
        json.dump(entry, fh, indent=2)
    return entry


def _unsupported_reason(info: dict, entry: dict) -> str | None:
    """Why this artifact cannot be launched by the pure-ctypes loader, if it cannot.

    Emitting it anyway would ship something that loads and then reads its arguments from
    the wrong place. Refusing at build time -- the way Triton's AOT compiler refuses
    kernels needing global scratch -- keeps the drop to artifacts that are actually
    launchable, and makes the unsupported set visible instead of latent.
    """
    md = next((k for k in info["kernels"] if k["name"] == entry["kernel"]), None)
    if md is None:
        return "元数据里没有该符号"
    hidden = [a for a in md["args"] if str(a["value_kind"]).startswith("hidden_")]
    if hidden:
        # Code object v5 places these inside the kernarg segment. Filling them means
        # deriving block counts, group sizes and remainders from the launch geometry;
        # doing that without capturing ground truth from a real launch would be guessing
        # at an ABI, and a wrong guess computes silently wrong answers.
        return f"声明了 {len(hidden)} 个 COV5 隐藏参数，宿主未实现"
    odd = [a["size"] for a in md["args"] if a["size"] not in (1, 2, 4, 8)]
    if odd:
        # Multi-field aggregates passed by value (tensor descriptors: shape + strides).
        # The loader only has the pointers and scalars the op hands it; it cannot
        # materialise a struct it never saw.
        return f"含多字段结构体传值（槽位 {sorted(set(odd))} 字节），宿主未实现"
    return None


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_env() -> dict:
    """What produced this drop. Read defensively -- a missing field should not fail a build."""

    def _read(path):
        try:
            with open(path) as fh:
                return fh.read().strip()
        except OSError:
            return None

    env = {
        "rocm_version": _read("/opt/rocm/.info/version"),
        "hostname": os.uname().nodename,
        "utc": __import__("datetime")
        .datetime.now(__import__("datetime").timezone.utc)
        .isoformat(timespec="seconds"),
    }
    try:
        import flydsl  # noqa: PLC0415

        env["flydsl_version"] = getattr(flydsl, "__version__", None)
    except Exception:
        env["flydsl_version"] = None
    for var in ("AITER_SITUV2_A8W4", "AITER_SITUV2_A4W4", "FLYDSL_RUNTIME_CACHE_DIR"):
        if os.environ.get(var):
            env.setdefault("env", {})[var] = os.environ[var]
    return env


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cache_dir")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument(
        "--arch",
        default=None,
        help="只收这个架构的产物，如 gfx950。gemm 预热会把 gfx1250/gfx942 也编进同一"
        "缓存，混在一起交付出去就会在目标机上加载失败",
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    pkls = []
    for root, _dirs, files in os.walk(args.cache_dir):
        pkls += [os.path.join(root, f) for f in files if f.endswith(".pkl")]
    if not pkls:
        raise SystemExit(f"no .pkl under {args.cache_dir}")

    built, no_launcher, unresolved, wrong_arch = [], 0, [], 0
    unsupported: dict[str, int] = {}
    for p in sorted(pkls):
        try:
            e = build_one(p, args.out)
            if e is not None:
                from co_abi import read_co

                info = read_co(os.path.join(args.out, e["co"]))
                reason = None
                if args.arch and info["arch"] != args.arch:
                    reason = f"arch {info['arch']}"
                else:
                    reason = _unsupported_reason(info, e)
                if reason:
                    os.remove(os.path.join(args.out, e["co"]))
                    meta = os.path.join(args.out, e["co"][:-3] + ".json")
                    if os.path.exists(meta):
                        os.remove(meta)
                    if reason.startswith("arch "):
                        wrong_arch += 1
                    else:
                        unsupported[reason] = unsupported.get(reason, 0) + 1
                    continue
        except SystemExit as exc:
            # An unresolvable grid is skipped rather than guessed at: a wrong grid
            # yields a kernel that launches and returns wrong answers. The dense
            # hgemm launchers need and/icmp/select (signed ceil-div lowering) that
            # the MoE launchers do not, so they land here until that is added.
            unresolved.append((os.path.basename(os.path.dirname(p)), str(exc)))
            continue
        if e is None:
            no_launcher += 1
            continue
        built.append(e)

    # Group by symbol name so a name carrying several distinct objects is visible in the
    # index instead of being resolved arbitrarily by insertion order.
    by_name: dict[str, list] = {}
    for e in built:
        by_name.setdefault(e["kernel"], []).append(e)
    index = {
        "schema": 2,
        # Provenance travels with the drop. Without it a `.co` on a shelf is unidentifiable:
        # you cannot tell which ROCm produced it, which arch it targets, or whether the
        # bytes still match what was tested. Every other AOT delivery records this.
        "build_env": _build_env(),
        "kernels": by_name,
        "checksums": {e["co"]: _sha256_file(os.path.join(args.out, e["co"])) for e in built},
    }
    with open(os.path.join(args.out, "index.json"), "w") as fh:
        json.dump(index, fh, indent=2)
    dupes = {k: len(v) for k, v in by_name.items() if len(v) > 1}

    print(f"  cache entries   {len(pkls)}")
    print(f"  built           {len(built)}")
    print(f"  no launcher     {no_launcher}")
    print(f"  wrong arch      {wrong_arch}")
    print(f"  unresolved grid {len(unresolved)}")
    if unsupported:
        print(f"  unsupported     {sum(unsupported.values())}")
        for why, n in sorted(unsupported.items(), key=lambda x: -x[1]):
            print(f"    {n:>5}  {why}")
    for name, why in unresolved:
        print(f"    {name[:56]:<56} {why}")
    for e in built:
        g = ",".join(
            str(d.get("const", f"arg{d.get('arg')}" if "arg" in d else d.get("op")))
            for d in e["grid"]
        )
        b = ",".join(str(d.get("const", "?")) for d in e["block"])
        print(
            f"    {e['kernel'][-58:]:<58} {e['co_sha256_12']}  {e['co_bytes']:>7}B  grid=({g})  lds={e['shared_bytes']}"
        )
    if dupes:
        print("  same symbol name, different object (must not be keyed by name alone):")
        for k, n in dupes.items():
            print(f"    {k[-58:]:<58} {n} objects")
    print(f"  index           {os.path.join(args.out, 'index.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
