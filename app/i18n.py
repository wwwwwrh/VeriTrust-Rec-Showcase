"""Small, explicit bilingual UI vocabulary."""

from __future__ import annotations


TEXT = {
    "zh": {
        "title": "VeriTrust-Rec",
        "subtitle": "Evidence-Aware · Verification-Grounded · Selective Recommendation",
        "query_tab": "查询",
        "recommendations_tab": "推荐结果",
        "evidence_tab": "证据与验证",
        "trace_tab": "管线追踪",
        "query_label": "输入您的学习需求",
        "query_placeholder": "我会 Python 和机器学习基础，想学习 Transformer，希望使用 PyTorch，并希望有代码实践。",
        "submit": "获取推荐",
        "clear": "清空",
        "working": "正在运行本地 Query Understanding 与可信推荐管线…",
        "no_result": "提交学习需求后，这里会显示结果。",
        "intent": "解析后的 StructuredIntent",
        "language_notice": "学习资料语言偏好已解析；当前 ranking 尚未启用该偏好。",
        "no_recommendation": "当前暂不推荐资源",
        "query_error": "Query Understanding 失败",
        "recommendation_error": "推荐管线发生错误",
        "generation_fallback": "自然语言生成未通过，下面显示确定性模板解释；推荐决策仍然有效。",
        "generation_failed": "最终解释暂不可用；结构化推荐、Claim 与 Evidence 仍然有效。",
        "confidence_notice": "启发式置信度分数，不是经过校准的正确概率。",
        "why": "为什么推荐？",
        "claims": "Claims",
        "debug": "显示 Debug 信息",
    },
    "en": {
        "title": "VeriTrust-Rec",
        "subtitle": "Evidence-Aware · Verification-Grounded · Selective Recommendation",
        "query_tab": "Query",
        "recommendations_tab": "Recommendations",
        "evidence_tab": "Evidence & Verification",
        "trace_tab": "Pipeline Trace",
        "query_label": "Describe your learning needs",
        "query_placeholder": "I know Python and basic machine learning. I want to learn Transformers with PyTorch and hands-on practice.",
        "submit": "Get recommendations",
        "clear": "Clear",
        "working": "Running local Query Understanding and the trustworthy recommendation pipeline…",
        "no_result": "Submit a learning request to see results.",
        "intent": "Parsed StructuredIntent",
        "language_notice": "The learning-material language preference is parsed; ranking support is not enabled yet.",
        "no_recommendation": "No recommendation is made for this request",
        "query_error": "Query Understanding failed",
        "recommendation_error": "The recommendation pipeline failed",
        "generation_fallback": "Natural-language generation was not accepted. A deterministic template is shown; the recommendation decision remains valid.",
        "generation_failed": "The final explanation is unavailable; structured recommendations, Claims, and Evidence remain valid.",
        "confidence_notice": "Heuristic confidence score; not a calibrated probability.",
        "why": "Why this resource?",
        "claims": "Claims",
        "debug": "Show debug information",
    },
}


def text(language: str, key: str) -> str:
    return TEXT[language][key]


__all__ = ["TEXT", "text"]
