"""LightRAG llm_func 适配器。

将项目现有的同步 ask_messages（HTTP + 返回带 .content 属性的 response）
包装成 LightRAG 需要的 async llm_func。

LightRAG 调用签名（两种形态都可能出现）：
- async func(messages: list[dict], **kwargs) -> str
- async func(messages: str, **kwargs) -> str

LightRAG 会把这些 kwargs 透传：temperature / max_tokens / response_format / ...
我们的 ask_messages 只认 temperature / max_tokens / model / messages。
"""

from __future__ import annotations

import asyncio
from typing import Any, List, Union


def make_llm_func(llm_model: str):
    """工厂：返回可直接传给 LightRAG(llm_model_func=...) 的 async LLM 函数。

    设计：延迟导入 utils.model —— 只有当工厂真正被调用时才触发 import 链路。
    这样 legacy_adapter 包体导入期间不会因为 utils.model 未安装或配置错误而崩。

    Args:
        llm_model: 模型名称，与 GRAPHRAG_LLM_MODEL 一致。

    Returns:
        async def func(messages, **kwargs) -> str
    """
    if not llm_model:
        raise RuntimeError("llm_model 未配置，无法构造 LightRAG llm_func")

    # 延迟导入：避免模块顶层硬 import 在 legacy 模式下触发失败
    from utils.model import ask_messages as _ask_messages

    async def _llm(messages: Union[str, List[dict]], **kwargs: Any) -> str:
        """LightRAG 调用入口：接收 str 或 list[dict]，返回纯文本 response.content.strip()。"""
        # 1. 归一化 messages 形态
        if isinstance(messages, str):
            normalized_messages: List[dict] = [{"role": "user", "content": messages}]
        else:
            normalized_messages = list(messages)

        # 2. kwargs 透传（ask_messages 只吃 temperature / max_tokens）
        temperature = kwargs.get("temperature", 0.1)
        # BUG-6 修复：LightRAG 内部不同版本可能透传 max_tokens 或 max_output_token，两者都尝试
        max_tokens = kwargs.get("max_tokens") or kwargs.get("max_output_token", 4096)

        # 3. 同步→async 桥接
        response = await asyncio.to_thread(
            _ask_messages,
            model=llm_model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=normalized_messages,
        )
        content = getattr(response, "content", "") or ""
        return content.strip()

    return _llm
