"""契约层异常。

只有一个异常类型：调用方（服务端 / CLI / 集群侧 launcher）据此统一把契约违规
翻译成自己的错误表达（HTTP 400、CLI 退出码、作业失败原因），无需关心细节分类。
"""
from __future__ import annotations


class SpecError(ValueError):
    """JobSpec 结构非法、字段缺失或取值越界。

    继承 ValueError 是有意的：服务端提交路径已有 `except ValueError -> HTTP 400`
    的兜底，契约包接入后不必先改错误处理就能得到正确的状态码。
    """

    def __init__(self, message: str, *, field: str | None = None):
        self.field = field
        super().__init__(f"{field}: {message}" if field else message)
