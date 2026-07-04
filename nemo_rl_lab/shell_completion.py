"""显式 shell 补全：不依赖 Typer/shellingham 自动检测当前 shell。

Typer 自带的 --install-completion 在 CI / IDE 终端 / 非 TTY 下常因检测失败而无法使用；
本模块提供 lab completion install|show <shell>，并支持仓库 ./lab shim 的 bash -F 包装。
"""
from __future__ import annotations

import re
from enum import Enum
from pathlib import Path

import typer
from typer._completion_shared import (
    get_completion_script,
    install_bash,
    install_fish,
    install_powershell,
    install_zsh,
)

PROG_NAME = "lab"
COMPLETE_VAR = "_LAB_COMPLETE"

# Typer bash 模板里 prog 名的合法化规则（与 _completion_shared 一致）
_IDENT_RE = re.compile(r"[^a-zA-Z0-9_]")


class ShellChoice(str, Enum):
    bash = "bash"
    zsh = "zsh"
    fish = "fish"
    powershell = "powershell"
    pwsh = "pwsh"


def _install_shell_name(shell: ShellChoice) -> str:
    return "pwsh" if shell == ShellChoice.pwsh else shell.value


def _wrapper_bash_script(repo_root: Path) -> str:
    """bash complete -F：通过 uv run --project <repo> lab 驱动 Typer 补全。"""
    repo = repo_root.resolve()
    lab_shim = (repo / "lab").resolve()
    func = f"_{_IDENT_RE.sub('', PROG_NAME.replace('-', '_'))}_shim_completion"
    repo_q = str(repo).replace("'", "'\\''")
    shim_q = str(lab_shim).replace("'", "'\\''")
    return f"""# lab 仓库 ./lab shim 补全（bash -F）；由 lab completion 生成
{func}() {{
    local IFS=$'\\n'
    COMPREPLY=( $( env COMP_WORDS="${{COMP_WORDS[*]}}" \\
                   COMP_CWORD=$COMP_CWORD \\
                   {COMPLETE_VAR}=complete_bash \\
                   uv run --project '{repo_q}' {PROG_NAME} ) )
    return 0
}}
complete -o default -F {func} '{shim_q}'
complete -o default -F {func} ./lab
"""


def _install_wrapper_bash(repo_root: Path) -> Path:
    completion_path = Path.home() / ".bash_completions" / "lab-shim.sh"
    rc_path = Path.home() / ".bashrc"
    rc_path.parent.mkdir(parents=True, exist_ok=True)
    rc_content = rc_path.read_text() if rc_path.is_file() else ""
    source_line = f"source '{completion_path}'"
    if source_line not in rc_content:
        rc_content = f"{rc_content.rstrip()}\n{source_line}\n"
        rc_path.write_text(rc_content)
    completion_path.parent.mkdir(parents=True, exist_ok=True)
    completion_path.write_text(_wrapper_bash_script(repo_root))
    return completion_path


def show_script(shell: ShellChoice, *, wrapper: bool = False, repo_root: Path | None = None) -> str:
    if wrapper:
        if shell != ShellChoice.bash:
            raise typer.BadParameter("--wrapper 目前仅支持 bash")
        if repo_root is None:
            raise typer.BadParameter("缺少仓库根目录")
        return _wrapper_bash_script(repo_root)
    return get_completion_script(
        prog_name=PROG_NAME,
        complete_var=COMPLETE_VAR,
        shell=_install_shell_name(shell),
    )


def install_script(shell: ShellChoice, *, wrapper: bool = False, repo_root: Path | None = None) -> Path:
    if wrapper:
        if shell != ShellChoice.bash:
            raise typer.BadParameter("--wrapper 目前仅支持 bash")
        if repo_root is None:
            raise typer.BadParameter("缺少仓库根目录")
        return _install_wrapper_bash(repo_root)
    name = _install_shell_name(shell)
    kwargs = dict(prog_name=PROG_NAME, complete_var=COMPLETE_VAR, shell=name)
    if name == "bash":
        return install_bash(**kwargs)
    if name == "zsh":
        return install_zsh(**kwargs)
    if name == "fish":
        return install_fish(**kwargs)
    return install_powershell(**kwargs)
