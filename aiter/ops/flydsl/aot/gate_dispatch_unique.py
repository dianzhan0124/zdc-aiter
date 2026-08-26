#!/usr/bin/env python3
# Dispatch-uniqueness gate for an AOT drop.
#
# Path B resolves a kernel at serve time by (symbol name[, shared_bytes]). If a single name
# carries two DISTINCT code objects that share the same shared_bytes, the runtime cannot pick
# between them and would resolve arbitrarily -- returning wrong numbers with no error. That is
# exactly the failure this gate refuses to ship.
#
# A name carrying several entries that are byte-identical (same co_sha256_12) is fine: it is
# just cache duplication, they all dispatch to the same bytes. A name carrying distinct objects
# is fine ONLY if shared_bytes separates them 1:1.
#
# Usage: python3 gate_dispatch_unique.py <drop_dir>
# Exit 0 = every name resolves to exactly one code object. Nonzero = ambiguous; nothing ships.
import json, os, sys

def main():
    if len(sys.argv) != 2:
        print('用法: gate_dispatch_unique.py <drop 目录>')
        return 2
    drop = sys.argv[1]
    idx_path = os.path.join(drop, 'index.json')
    if not os.path.exists(idx_path):
        print(f'找不到 {idx_path}')
        return 2
    idx = json.load(open(idx_path))
    kernels = idx.get('kernels', {})
    ambiguous = []
    names = 0
    multi = 0
    for name, entries in kernels.items():
        names += 1
        # distinct binaries under this name, keyed by content hash
        by_hash = {}
        for e in entries:
            by_hash.setdefault(e['co_sha256_12'], e)
        if len(by_hash) <= 1:
            continue
        multi += 1
        # distinct objects exist -> shared_bytes must separate them 1:1
        by_sb = {}
        for h, e in by_hash.items():
            by_sb.setdefault(e.get('shared_bytes'), set()).add(h)
        for sb, hashes in by_sb.items():
            if len(hashes) > 1:
                ambiguous.append((name, sb, sorted(hashes)))
    if ambiguous:
        print(f'  名字 {names}，其中带多个不同产物的 {multi}')
        print(f'  无法消歧的名字 {len(ambiguous)}：')
        for name, sb, hashes in ambiguous:
            print(f'    {name}  shared_bytes={sb}  ->  {len(hashes)} 个不同产物 {hashes}')
        print('  同名 + 同 shared_bytes 却是不同 code object：派发只能任选其一，会出错。')
        return 1
    print(f'  名字 {names}，带多个不同产物但可用 shared_bytes 消歧的 {multi}，无歧义')
    return 0

if __name__ == '__main__':
    sys.exit(main())
