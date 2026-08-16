#!/usr/bin/env python
"""教程示例：生成一份极小的合成加法 QA 数据集（jsonl），用于流程冒烟。

data-prep 插件不写 entrypoint：安装到 forge_plugins/<name>/ 后，CLI 按
prepare_<数据集名>.py 的文件名约定发现（本文件 → `sf dataset prepare tiny_qa`），
本行 docstring 的第一行会作为数据集的一句话说明展示在 CLI 列表里。

用法：
    sf dataset prepare tiny_qa            # 写到 <repo>/datasets/tiny_qa/
    sf dataset prepare tiny_qa --out /abs/dir --n 200
"""
import json
import os
import random

import typer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main(
    out: str = typer.Option("", help="输出目录；默认 <repo>/datasets/tiny_qa/"),
    n: int = typer.Option(100, help="样本条数"),
    seed: int = typer.Option(0, help="随机种子"),
) -> None:
    out_dir = out or os.path.join(REPO_ROOT, "datasets", "tiny_qa")
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    path = os.path.join(out_dir, "train.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(n):
            a, b = rng.randint(1, 999), rng.randint(1, 999)
            row = {"question": f"What is {a} + {b}?", "answer": str(a + b)}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    typer.echo(f"已写 {n} 条 → {path}")


if __name__ == "__main__":
    typer.run(main)
