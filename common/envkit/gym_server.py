"""Gym 风格 resources server 薄壳：把 envkit 内核暴露成 HTTP 协议。

对齐 NeMo Gym 的三件套语义：
  POST /seed_session   建会话（注入题面/KB/期望答案等 task payload），返回 session_id
  POST /tools/{name}   类型化工具路由（会话隔离：工具上下文来自本会话）
  POST /verify         终局判分（response → reward），并结束会话

与训练主线的关系：同一内核（envkit.tools / 环境的 verify 逻辑）两条腿 ——
NeMo-RL EnvironmentInterface 走进程内批量调用，本服务走 HTTP 单会话调用。
rollout 框架（NeMo Gym / 自研 harness）只需要说这三句话。

fastapi 按需导入：训练主线不依赖本模块。

注意：本模块**不能**用 `from __future__ import annotations` —— 路由函数的
Pydantic 模型定义在工厂函数局部作用域，字符串化注解会让 FastAPI 解析不到
模型类型，把 body 误判成 query 参数。
"""
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


class GymEnvAdapter(Protocol):
    """一个环境要上 Gym 协议需要实现的三件事。"""

    name: str
    #: name -> callable(arg: str, ctx: dict) -> str；ctx 是本会话的可变状态。
    tools: dict[str, Callable[[str, dict], str]]

    def seed_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        """校验 task payload 并返回会话初始状态（工具与 verify 的 ctx）。"""
        ...

    def verify(self, ctx: dict[str, Any], response: str) -> dict[str, Any]:
        """终局判分。返回至少 {"reward": float}；可附加诊断字段。"""
        ...


@dataclass
class _Session:
    ctx: dict[str, Any]
    adapter_name: str
    tool_calls: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


def create_gym_app(adapter: GymEnvAdapter, *, max_sessions: int = 10000):
    """构建 FastAPI 应用。会话存内存（resources server 与 rollout 同生命周期）。"""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    app = FastAPI(title=f"starforge gym server · {adapter.name}")
    sessions: dict[str, _Session] = {}

    class SeedBody(BaseModel):
        payload: dict[str, Any] = {}

    class ToolBody(BaseModel):
        session_id: str
        arg: str = ""

    class VerifyBody(BaseModel):
        session_id: str
        response: str

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "env": adapter.name, "tools": sorted(adapter.tools)}

    @app.post("/seed_session")
    def seed_session(body: SeedBody) -> dict:
        if len(sessions) >= max_sessions:
            raise HTTPException(429, "会话数已达上限")
        try:
            ctx = adapter.seed_session(body.payload)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        session_id = uuid.uuid4().hex
        sessions[session_id] = _Session(ctx=ctx, adapter_name=adapter.name)
        return {"session_id": session_id, "tools": sorted(adapter.tools)}

    @app.post("/tools/{name}")
    def call_tool(name: str, body: ToolBody) -> dict:
        session = sessions.get(body.session_id)
        if session is None:
            raise HTTPException(404, "session 不存在或已结束")
        tool = adapter.tools.get(name)
        if tool is None:
            raise HTTPException(404, f"未知工具 {name!r}（可用: {', '.join(sorted(adapter.tools))}）")
        with session.lock:  # 会话内串行：工具可能改 ctx
            session.tool_calls += 1
            observation = tool(body.arg, session.ctx)
        return {"observation": observation}

    @app.post("/verify")
    def verify(body: VerifyBody) -> dict:
        session = sessions.pop(body.session_id, None)
        if session is None:
            raise HTTPException(404, "session 不存在或已结束")
        result = adapter.verify(session.ctx, body.response)
        reward = float(result.get("reward", 0.0))
        return {**result, "reward": reward, "tool_calls": session.tool_calls}

    return app
