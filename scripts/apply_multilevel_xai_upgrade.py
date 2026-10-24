"""Validated legacy overlay installer retained for existing upgrade packages."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import tempfile


def _inside(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if path == root or root not in path.parents:
        raise ValueError("Unsafe upgrade path: " + relative)
    return path


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def apply(package, project, allow_local_changes=False):
    package, project = Path(package).resolve(), Path(project).resolve()
    if not (project / "app.py").is_file() or not (project / "awareml").is_dir():
        raise ValueError("Not an AwareML project root.")
    manifest = json.loads((package / "UPGRADE_MANIFEST.json").read_text(encoding="utf-8"))
    changes, seen = [], set()
    # Validate the entire package before creating a backup or changing a file.
    for item in manifest["files"]:
        rel = item["path"]
        source, target = _inside(package / "payload", rel), _inside(project, rel)
        if target in seen:
            raise ValueError("Duplicate upgrade target: " + rel)
        seen.add(target)
        if not source.is_file() or _sha(source) != item["new_sha256"]:
            raise ValueError("Corrupt package: " + rel)
        current = _sha(target) if target.is_file() else None
        if target.exists() and not target.is_file():
            raise ValueError("Target is not a regular file: " + rel)
        if current == item["new_sha256"]:
            continue
        if current != item.get("base_sha256") and not allow_local_changes:
            raise ValueError("Local changes conflict: " + rel)
        changes.append((rel, source, target, target.is_file()))
    if not changes:
        return {"status": "already_applied"}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = project / ".multilevel_xai_backups" / stamp
    backup.mkdir(parents=True)
    for rel, source, target, existed in changes:
        if existed:
            original = _inside(backup, rel)
            original.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(target), str(original))
    written = []
    try:
        for rel, source, target, existed in changes:
            _write(target, source.read_bytes())
            written.append((rel, target, existed))
    except Exception:
        for rel, target, existed in reversed(written):
            if existed:
                _write(target, _inside(backup, rel).read_bytes())
            else:
                target.unlink()
        raise
    return {"status": "applied", "backup": str(backup), "files": len(changes)}
