#!/usr/bin/env python3
"""Read a code object's own ABI declaration -- no compiler toolchain required.

An AMDGPU code object states its ABI in a msgpack note (`NT_AMDGPU_METADATA`): the
kernarg segment size and alignment, every argument's offset/size/kind, the LDS budget,
the required workgroup size. That declaration is authoritative. Anything the host packs
must agree with it, and the only way to know is to read it.

`llvm-readelf --notes` prints the same thing, but reaching for it would put the compiler
toolchain back in the loop -- the exact dependency this delivery exists to remove. So the
ELF walk and the msgpack decode are done here, in a few hundred lines, with nothing but
the standard library. That also means the check can run at load time in production.

Two things this catches that nothing else does:

  * **Kernarg layout drift.** The host currently derives offsets from the MLIR argument
    types by re-deriving alignment. If that logic and the code object ever disagree, every
    argument after the first mismatch is read from the wrong offset -- which does not
    crash, it computes wrong numbers.
  * **Hidden arguments.** Code object v5 introduced trailing hidden arguments
    (`hidden_block_count_x` and friends). They are omitted when unused, so their presence
    depends on the kernel. If one appears, it lands *inside* the kernarg segment and every
    assumption about "size == sum of my arguments" breaks.

Usage:
    python3 co_abi.py <file.co> [...]           # dump
    python3 co_abi.py --check <drop_dir>        # cross-check against index.json
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys

# ---------------------------------------------------------------- msgpack


class _Mp:
    """Just enough msgpack to read AMDGPU metadata."""

    def __init__(self, buf: bytes):
        self.b = buf
        self.i = 0

    def _u(self, fmt: str, n: int):
        v = struct.unpack_from(fmt, self.b, self.i)[0]
        self.i += n
        return v

    def read(self):
        c = self.b[self.i]
        self.i += 1
        if c <= 0x7F:
            return c
        if c >= 0xE0:
            return c - 0x100
        if 0x80 <= c <= 0x8F:
            return self._map(c & 0x0F)
        if 0x90 <= c <= 0x9F:
            return [self.read() for _ in range(c & 0x0F)]
        if 0xA0 <= c <= 0xBF:
            return self._str(c & 0x1F)
        if c == 0xC0:
            return None
        if c == 0xC2:
            return False
        if c == 0xC3:
            return True
        if c == 0xC4:
            return self._bin(self._u("<B", 1))
        if c == 0xC5:
            return self._bin(self._u(">H", 2))
        if c == 0xC6:
            return self._bin(self._u(">I", 4))
        if c == 0xCA:
            return self._u("<f", 4)
        if c == 0xCB:
            return self._u("<d", 8)
        if c == 0xCC:
            return self._u("<B", 1)
        if c == 0xCD:
            return self._u(">H", 2)
        if c == 0xCE:
            return self._u(">I", 4)
        if c == 0xCF:
            return self._u(">Q", 8)
        if c == 0xD0:
            return self._u("<b", 1)
        if c == 0xD1:
            return self._u(">h", 2)
        if c == 0xD2:
            return self._u(">i", 4)
        if c == 0xD3:
            return self._u(">q", 8)
        if c == 0xD9:
            return self._str(self._u("<B", 1))
        if c == 0xDA:
            return self._str(self._u(">H", 2))
        if c == 0xDB:
            return self._str(self._u(">I", 4))
        if c == 0xDC:
            return [self.read() for _ in range(self._u(">H", 2))]
        if c == 0xDD:
            return [self.read() for _ in range(self._u(">I", 4))]
        if c == 0xDE:
            return self._map(self._u(">H", 2))
        if c == 0xDF:
            return self._map(self._u(">I", 4))
        raise ValueError(f"msgpack: 未支持的标记 0x{c:02x} @ {self.i - 1}")

    def _map(self, n: int):
        out = {}
        for _ in range(n):
            k = self.read()
            out[k] = self.read()
        return out

    def _str(self, n: int) -> str:
        s = self.b[self.i : self.i + n].decode("utf-8", "replace")
        self.i += n
        return s

    def _bin(self, n: int) -> bytes:
        s = self.b[self.i : self.i + n]
        self.i += n
        return s


# ---------------------------------------------------------------- ELF

NT_AMDGPU_METADATA = 32
SHT_NOTE = 7

# EF_AMDGPU_MACH, low byte of e_flags. Only the ones this delivery can meet are named;
# anything else is reported as a raw value rather than guessed at.
_MACH = {
    0x02C: "gfx900",
    0x02F: "gfx906",
    0x030: "gfx908",
    0x03F: "gfx90a",
    0x040: "gfx940",
    0x041: "gfx941",
    0x049: "gfx1250",
    0x04C: "gfx942",
    0x04F: "gfx950",
}
_TRI = {0: "unsupported", 1: "any", 2: "off", 3: "on"}


def read_co(path: str) -> dict:
    with open(path, "rb") as fh:
        b = fh.read()
    return parse_co(b, path)


def parse_co(b: bytes, path: str = "<bytes>") -> dict:
    if b[:4] != b"\x7fELF":
        raise ValueError(f"{path}: 不是 ELF")
    if b[4] != 2 or b[5] != 1:
        raise ValueError(f"{path}: 只支持 64 位小端")
    osabi, abiversion = b[7], b[8]
    (e_machine,) = struct.unpack_from("<H", b, 0x12)
    (e_flags,) = struct.unpack_from("<I", b, 0x30)
    e_shoff, = struct.unpack_from("<Q", b, 0x28)
    e_shentsize, e_shnum = struct.unpack_from("<HH", b, 0x3A)

    mach = e_flags & 0xFF
    info = {
        "path": path,
        "size": len(b),
        "osabi": osabi,
        "elf_abiversion": abiversion,
        # ELFABIVERSION_AMDGPU_HSA_V4=2, V5=3, V6=4. Reported as both so a shift in
        # either is visible.
        "code_object_version": {2: 4, 3: 5, 4: 6}.get(abiversion),
        "e_machine": e_machine,
        "e_flags": e_flags,
        "arch": _MACH.get(mach, f"mach=0x{mach:03x}"),
        "xnack": _TRI.get((e_flags >> 8) & 0x3, "?"),
        "sramecc": _TRI.get((e_flags >> 10) & 0x3, "?"),
        "target": None,
        "metadata_version": None,
        "kernels": [],
    }

    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        sh_type, = struct.unpack_from("<I", b, off + 4)
        if sh_type != SHT_NOTE:
            continue
        sh_offset, sh_size = struct.unpack_from("<QQ", b, off + 0x18)
        p, end = sh_offset, sh_offset + sh_size
        while p + 12 <= end:
            n_namesz, n_descsz, n_type = struct.unpack_from("<III", b, p)
            p += 12
            name = b[p : p + n_namesz].rstrip(b"\0").decode("ascii", "replace")
            p += (n_namesz + 3) & ~3
            desc = b[p : p + n_descsz]
            p += (n_descsz + 3) & ~3
            if n_type != NT_AMDGPU_METADATA or not name.startswith("AMDGPU"):
                continue
            md = _Mp(desc).read()
            info["target"] = md.get("amdhsa.target")
            ver = md.get("amdhsa.version")
            if isinstance(ver, list):
                info["metadata_version"] = ".".join(str(x) for x in ver)
            for k in md.get("amdhsa.kernels", []) or []:
                info["kernels"].append(
                    {
                        "name": k.get(".name"),
                        "symbol": k.get(".symbol"),
                        "kernarg_segment_size": k.get(".kernarg_segment_size"),
                        "kernarg_segment_align": k.get(".kernarg_segment_align"),
                        "group_segment_fixed_size": k.get(".group_segment_fixed_size"),
                        "private_segment_fixed_size": k.get(
                            ".private_segment_fixed_size"
                        ),
                        "max_flat_workgroup_size": k.get(".max_flat_workgroup_size"),
                        "reqd_workgroup_size": k.get(".reqd_workgroup_size"),
                        "wavefront_size": k.get(".wavefront_size"),
                        "sgpr_count": k.get(".sgpr_count"),
                        "vgpr_count": k.get(".vgpr_count"),
                        "sgpr_spill_count": k.get(".sgpr_spill_count"),
                        "vgpr_spill_count": k.get(".vgpr_spill_count"),
                        "uses_dynamic_stack": k.get(".uses_dynamic_stack"),
                        "args": [
                            {
                                "offset": a.get(".offset"),
                                "size": a.get(".size"),
                                "value_kind": a.get(".value_kind"),
                                "address_space": a.get(".address_space"),
                            }
                            for a in (k.get(".args") or [])
                        ],
                    }
                )
    return info


HIDDEN_PREFIX = "hidden_"


def _sha256_file(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _arch_of_target(target: str | None) -> str | None:
    """`amdgcn-amd-amdhsa--gfx950:sramecc+:xnack-` -> `gfx950`."""
    if not target:
        return None
    tail = target.rsplit("--", 1)[-1]
    return tail.split(":")[0] or None


def try_load(blob: bytes, symbol: str) -> str | None:
    """Load one object and resolve its symbol; returns None on success, else the reason.

    Metadata agreeing with itself is not the same as the runtime accepting the object --
    the prewarm path logs `hipErrorNoBinaryForGpu` while still reporting success, so the
    only way to know a shipped artifact is usable here is to load it.
    """
    import ctypes
    import ctypes.util

    lib = ctypes.CDLL(ctypes.util.find_library("amdhip64") or "libamdhip64.so")
    lib.hipGetErrorString.restype = ctypes.c_char_p
    mod = ctypes.c_void_p()
    buf = ctypes.create_string_buffer(blob, len(blob))
    rc = lib.hipModuleLoadData(ctypes.byref(mod), ctypes.cast(buf, ctypes.c_void_p))
    if rc != 0:
        return f"hipModuleLoadData rc={rc} {(lib.hipGetErrorString(rc) or b'').decode()}"
    fn = ctypes.c_void_p()
    rc = lib.hipModuleGetFunction(ctypes.byref(fn), mod, symbol.encode())
    if rc != 0:
        return f"hipModuleGetFunction rc={rc}"
    return None


def check_drop(
    drop_dir: str, device_arch: str | None = None, do_load: bool = False
) -> int:
    """Cross-check every artifact against its own declaration and against index.json."""
    with open(os.path.join(drop_dir, "index.json")) as fh:
        index = json.load(fh)
    entries = index["kernels"] if isinstance(index, dict) else index
    if isinstance(entries, dict):
        entries = [e for v in entries.values() for e in v]
    checksums = index.get("checksums") if isinstance(index, dict) else None

    bad = 0
    warn_total = 0
    seen_targets, seen_cov = set(), set()
    for e in entries:
        co = os.path.join(drop_dir, e["co"])
        info = read_co(co)
        seen_targets.add(info["target"])
        seen_cov.add((info["code_object_version"], info["metadata_version"]))
        problems: list[str] = []
        warnings: list[str] = []

        # No checksum block is a failure, not a pass. A drop that carries no integrity
        # data cannot be verified, and treating that as "verified" makes every tamper
        # check below pass vacuously -- which is how the pre-schema-2 drop slipped through.
        if not checksums:
            problems.append("index 里没有 checksums，无法核对完整性（需用当前构建器重建）")
        else:
            want = checksums.get(e["co"])
            if want is None:
                problems.append("index 的 checksums 里没有这一项")
            elif want != _sha256_file(co):
                problems.append("sha256 与 index 记录不符（产物被改动或损坏）")

        # HIP enforces the ELF e_flags machine field but *not* the metadata target string:
        # patching `amdhsa.target` from gfx950 to gfx942 still loads with rc=0. So a
        # mislabelled artifact is only caught if the delivery checks this itself.
        md_arch = _arch_of_target(info["target"])
        if md_arch and md_arch != info["arch"]:
            problems.append(
                f"e_flags 架构 {info['arch']} 与元数据 target {md_arch} 不一致"
            )
        if device_arch:
            want = device_arch.split(":")[0]
            if info["arch"] != want:
                problems.append(f"产物架构 {info['arch']} 与本机设备 {want} 不匹配")

        want = e["kernel"]
        ks = [k for k in info["kernels"] if k["name"] == want]
        if not ks:
            problems.append(f"元数据里没有 {want}")
            k = None
        else:
            k = ks[0]

        if k is not None:
            types = [e["wrapper_arg_types"][i].strip() for i in e["kernel_args"]]
            md_args = list(k["args"])
            hidden = [
                a
                for a in md_args
                if str(a["value_kind"]).startswith(HIDDEN_PREFIX)
            ]
            if hidden:
                problems.append(
                    f"存在隐藏参数 {[a['value_kind'] for a in hidden]}，"
                    "宿主打包未考虑"
                )
            if len(md_args) != len(types):
                problems.append(
                    f"参数个数不符：元数据 {len(md_args)} vs index 映射 {len(types)}"
                )
            else:
                # The host packs into the offsets the code object declares, so comparing
                # against a re-derived layout would only test a rule nothing uses. What
                # still has to hold is that each declared slot can hold the value routed
                # into it -- above all that a pointer never lands in a 4-byte slot, which
                # would truncate it without any error.
                for j, (a, t) in enumerate(zip(md_args, types)):
                    is_ptr = t.startswith("!llvm.ptr")
                    if is_ptr and (a["size"] != 8 or a["value_kind"] != "global_buffer"):
                        problems.append(
                            f"参数 {j} 是指针，但槽位是 {a['size']} 字节的 "
                            f"{a['value_kind']}，会被截断"
                        )
                    if not is_ptr and a["value_kind"] == "global_buffer":
                        problems.append(
                            f"参数 {j} 槽位声明为 global_buffer，但 launcher 类型是 {t}"
                        )
                    if a["size"] not in (1, 2, 4, 8):
                        problems.append(f"参数 {j} 槽位大小 {a['size']} 宿主无法打包")
                declared_end = max(
                    (a["offset"] + a["size"] for a in md_args), default=0
                )
                if declared_end > k["kernarg_segment_size"]:
                    problems.append(
                        f"参数越出 kernarg 段：末端 {declared_end} > "
                        f"{k['kernarg_segment_size']}"
                    )
            lds = e.get("lds")
            if lds is not None and k["group_segment_fixed_size"] != lds:
                problems.append(
                    f"LDS 不符：元数据 {k['group_segment_fixed_size']} vs index {lds}"
                )
            rwg = k["reqd_workgroup_size"]
            blk = e.get("block")
            if rwg and blk and all(isinstance(x, int) for x in blk):
                if list(rwg) != list(blk):
                    problems.append(f"block 不符：元数据 {rwg} vs index {blk}")
            # Spills and a dynamic stack are reported but do not fail the drop: HIP
            # provides the scratch the code object asks for, so these are a performance
            # signal, not a launch-correctness one.
            if k["vgpr_spill_count"] or k["sgpr_spill_count"]:
                warnings.append(
                    f"寄存器溢出 vgpr={k['vgpr_spill_count']} "
                    f"sgpr={k['sgpr_spill_count']}"
                )
            if k["uses_dynamic_stack"]:
                warnings.append("uses_dynamic_stack=true，需要 scratch")

        if do_load:
            with open(co, "rb") as fh:
                why = try_load(fh.read(), e["kernel"])
            if why:
                problems.append(f"本机加载失败：{why}")

        tag = "OK  " if not problems else "FAIL"
        print(f"  [{tag}] {os.path.basename(co)[:72]}")
        if k is not None:
            print(
                f"         {info['arch']} cov{info['code_object_version']} "
                f"md{info['metadata_version']} xnack={info['xnack']} "
                f"sramecc={info['sramecc']}  kernarg={k['kernarg_segment_size']}B "
                f"lds={k['group_segment_fixed_size']} vgpr={k['vgpr_count']} "
                f"sgpr={k['sgpr_count']}"
            )
        for p in problems:
            print(f"         - {p}")
        for w in warnings:
            print(f"         ~ {w}")
        bad += 1 if problems else 0
        warn_total += len(warnings)

    print()
    be = index.get("build_env") if isinstance(index, dict) else None
    if be:
        print(
            f"  构建环境      ROCm {be.get('rocm_version')} / flydsl "
            f"{be.get('flydsl_version')} / {be.get('utc')}"
        )
    print(f"  target        {sorted(t for t in seen_targets if t)}")
    print(f"  code object   {sorted(seen_cov)}")
    print(f"  校验和        {'已核对' if checksums else '**index 里没有 checksums**'}")
    print(f"  警告          {warn_total} 条（寄存器溢出等，不阻塞交付）")
    print(f"  结果          {len(entries) - bad}/{len(entries)} 与自身声明一致")
    return 0 if bad == 0 else 1


def device_arch_via_hip() -> str | None:
    """Ask HIP what this machine is, so a drop can be refused before it is launched."""
    import ctypes
    import ctypes.util

    try:
        lib = ctypes.CDLL(ctypes.util.find_library("amdhip64") or "libamdhip64.so")
        buf = ctypes.create_string_buffer(1 << 16)
        if lib.hipGetDeviceProperties(buf, 0) != 0:
            return None
        raw = buf.raw
        i = raw.find(b"gfx")
        if i < 0:
            return None
        j = i
        while j < len(raw) and raw[j] not in (0, 32):
            j += 1
        return raw[i:j].decode("ascii", "replace")
    except OSError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--check", action="store_true", help="把参数当作 drop 目录做交叉校验")
    ap.add_argument(
        "--load", action="store_true", help="逐个真正加载并解析符号"
    )
    ap.add_argument(
        "--device-arch",
        nargs="?",
        const="auto",
        default=None,
        help="同时核对本机设备架构；不给值则向 HIP 查询",
    )
    a = ap.parse_args()
    if a.check:
        arch = a.device_arch
        if arch == "auto":
            arch = device_arch_via_hip()
            print(f"  本机设备      {arch}")
        return max(check_drop(p, arch, a.load) for p in a.paths)
    for p in a.paths:
        info = read_co(p)
        print(json.dumps(info, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
