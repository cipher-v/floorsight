"""Pack the Floorsight code (tracked + untracked, not ignored) into a .tgz for the instance, minus private data.

    python infra/aws/pack_bundle.py --out build/floorsight-20261023-r1.tgz

Excluded: evaluation evidence, owner-review media, research listings, the private clip map, Terraform
state/plans/vars, scratch folders. Prints the file count, size and SHA-256; writes <out>.files.txt.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXCLUDE_PREFIXES = ("eval/", "owner-review/", "research/", "infra/aws/manual/", "build/", ".next-")
EXCLUDE_SUFFIXES = (".tfstate", ".tfstate.backup", ".tfplan", ".tfvars", ".tgz", ".pyc")
EXCLUDE_NAMES = ("infra/terraform/plan-", "infra/terraform/apply-")


def files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    keep = []
    for f in sorted(set(out.splitlines())):
        if f.startswith(EXCLUDE_PREFIXES) or f.endswith(EXCLUDE_SUFFIXES) or f.startswith(EXCLUDE_NAMES) or "__pycache__" in f:
            continue
        if (ROOT / f).is_file():
            keep.append(f)
    return keep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; bundles are never overwritten (use a new rN)")
    keep = files()
    with tarfile.open(a.out, "w:gz") as tf:
        for f in keep:
            tf.add(ROOT / f, arcname=f)
    digest = hashlib.sha256(a.out.read_bytes()).hexdigest()
    Path(str(a.out) + ".files.txt").write_text("\n".join(keep) + "\n", encoding="utf-8")
    print(f"{a.out} files={len(keep)} bytes={a.out.stat().st_size} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
