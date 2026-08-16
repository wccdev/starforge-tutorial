"""实验插件锁文件（plugins.lock.json）。

与 recipe.lock.json 同一哲学：实验用哪个插件的哪个精确版本、什么内容摘要，
在 `sf plugin install --exp` 时固定下来；提交时原样写进 JobSpec，服务端与
集群侧各自比对 —— 不一致就拒绝，绝不猜。

锁定的是 (id, version, digest) 三元组，不含代码：插件包体由平台在提交时注入
作业包，客户端不上传、也改不了。
"""
from __future__ import annotations

import json
from pathlib import Path

LOCK_FILE = "plugins.lock.json"
LOCK_VERSION = "forge/plugins-lock/v1"


def read_plugin_lock(exp_dir: Path) -> list[dict]:
    """读取实验的插件引用列表；没有锁文件返回空表。"""
    path = Path(exp_dir) / LOCK_FILE
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path} 非法: {exc}") from exc
    if payload.get("apiVersion") != LOCK_VERSION:
        raise ValueError(
            f"{path} 的 apiVersion 不是 {LOCK_VERSION}；"
            "重新执行 `sf plugin install <id> --exp <exp>` 重建锁文件"
        )
    plugins = payload.get("plugins") or []
    for p in plugins:
        if not all(str(p.get(k) or "").strip() for k in ("id", "version", "digest")):
            raise ValueError(f"{path} 存在缺字段的插件引用: {p!r}")
    return [
        {"id": p["id"], "version": p["version"], "digest": p["digest"]}
        for p in plugins
    ]


def upsert_plugin_lock(exp_dir: Path, entry: dict) -> list[dict]:
    """写入/更新一条插件引用（同 id 覆盖），返回更新后的列表。"""
    plugins = [p for p in read_plugin_lock(exp_dir) if p["id"] != entry["id"]]
    plugins.append({"id": entry["id"], "version": entry["version"], "digest": entry["digest"]})
    plugins.sort(key=lambda p: p["id"])
    _write(exp_dir, plugins)
    return plugins


def remove_plugin_lock(exp_dir: Path, plugin_id: str) -> bool:
    """移除一条插件引用；返回是否真的移除了。"""
    plugins = read_plugin_lock(exp_dir)
    kept = [p for p in plugins if p["id"] != plugin_id]
    if len(kept) == len(plugins):
        return False
    _write(exp_dir, kept)
    return True


def _write(exp_dir: Path, plugins: list[dict]) -> None:
    path = Path(exp_dir) / LOCK_FILE
    if not plugins:
        path.unlink(missing_ok=True)
        return
    path.write_text(
        json.dumps(
            {"apiVersion": LOCK_VERSION, "plugins": plugins},
            ensure_ascii=False, indent=2, sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
