"""问答 Prompt 的确定性契约测试，不依赖真实模型调用。"""

from app.langchain.prompts import RAG_PROMPT, SYSTEM_PROMPT


def test_system_prompt_defines_evidence_based_rule_judgment() -> None:
    """规则类问答必须给出三态结论，不能一律退化为“无法判断”。"""
    assert "只能依据检索资料和用户明确给出的事实回答" in SYSTEM_PROMPT
    assert "结论：是 / 否 / 无法判断" in SYSTEM_PROMPT
    assert "未达到该门槛" in SYSTEM_PROMPT
    assert "先分别给出各分项结论，再给总体结论" in SYSTEM_PROMPT
    assert "仅就时长而言，是或否；是否存在其他违规，无法判断" in SYSTEM_PROMPT
    assert "资料冲突时，明确指出冲突" in SYSTEM_PROMPT
    assert "不替代有权部门的最终认定" in SYSTEM_PROMPT


def test_rag_prompt_includes_rule_judgment_contract_and_context() -> None:
    """普通与流式链共用该模板，问题和证据必须同时传入模型。"""
    messages = RAG_PROMPT.format_messages(
        history=[],
        question="在美创停12楼超过3天算违规吗？",
        context="[1] 公约规定：超长时间停放为5天及以上。",
    )

    assert "结论：是 / 否 / 无法判断" in messages[0].content
    assert "超过3天" in messages[-1].content
    assert "5天及以上" in messages[-1].content
