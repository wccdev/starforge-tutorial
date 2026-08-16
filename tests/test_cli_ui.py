"""CLI 错误格式化单测。"""
from __future__ import annotations

from starforge_cli import cli_ui


def test_parse_hf_preflight_multiline():
    raw = (
        "提交前 HuggingFace 资源预检未通过：\n"
        "- 无权访问 dataset OpenMathInstruct-2（私有/未授权，401）；"
        "当前未绑定 HF 账号，若为私有/gated 资源请先在「集成 / HuggingFace」绑定后重试"
    )
    msg = cli_ui.parse_message(raw)
    assert msg.title == "HuggingFace 资源预检未通过"
    assert len(msg.items) == 1
    assert "OpenMathInstruct-2" in msg.items[0]
    assert msg.hint == "在 Web 控制台绑定 HuggingFace：集成 → HuggingFace"


def test_parse_invalid_id_no_bind_hint():
    raw = (
        "提交前 HuggingFace 资源预检未通过：\n"
        "- dataset OpenMathInstruct-2 不是有效的 HuggingFace repo id（需 org/name，例如 nvidia/OpenMathInstruct-2）"
    )
    msg = cli_ui.parse_message(raw)
    assert msg.hint == ""


def test_shorten_bullet_hf_dataset():
    head, tail = cli_ui._shorten_bullet("无权访问 dataset OpenMathInstruct-2（私有/未授权，401）")
    assert head == "数据集 OpenMathInstruct-2"
    assert "无权访问" in tail


def test_parse_plain_message():
    msg = cli_ui.parse_message("提交失败，请稍后重试。")
    assert msg.title == "提交失败，请稍后重试。"


def test_http_error_detail_json_string():
    import io
    import urllib.error

    body = b'{"detail":"quota exceeded"}'
    err = urllib.error.HTTPError("http://x", 422, "Unprocessable", {}, io.BytesIO(body))
    assert cli_ui.http_error_detail(err, fallback="x") == "quota exceeded"
