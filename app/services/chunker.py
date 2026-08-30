"""文本切片服务（Day 5）：把解析出的页面文本切成适合 Embedding 的 chunk。

切片是 RAG 检索质量的第一道分水岭，参数直接影响后续所有环节：

- chunk 太大：一个切片混杂多个主题，Embedding 向量的语义被"平均化"而变得模糊，
  检索命中率下降；同时塞进 Prompt 的无关内容变多，挤占 Token 预算、增加成本。
- chunk 太小：语义上下文被切断（一句话的前后半句分家），
  检索到的片段不足以支撑模型组织回答。
- overlap（重叠）：让跨越切片边界的话题在相邻切片中都完整出现，
  避免"答案正好被切在边界上"导致两个切片都语义残缺。

本模块只做确定性文本处理，不调用任何模型，因此测试无需网络与 API Key。
"""

import re
from typing import Dict, List, Tuple

# 目标切片大小：500 token 约对应 300-500 个汉字或 350-500 个英文单词，
# 是检索精度与上下文完整性的常见折中（业界常用范围 256-1024）。
# 调大：检索更"模糊"但单次召回信息更多；调小：检索更精准但容易断章取义。
CHUNK_SIZE = 500
# 重叠量取 chunk 的 10% 左右：太小起不到边界保护作用，太大则存储翻倍率升高、
# 且同一内容被多个切片重复命中，检索结果出现冗余。
CHUNK_OVERLAP = 50

# 中日韩统一表意文字范围：这些字符在主流 tokenizer 中约 1 字 1 token。
_CJK_RANGE = ("\u4e00", "\u9fff")
# 句子边界：中英文常见句末标点。真实项目可用 NLP 分句库处理缩写（如 "U.S."），
# 教学场景用正则足够，且保持零依赖。
_SENTENCE_RE = re.compile(r"[^。！？.!?]+[。！？.!?]*")


def estimate_tokens(text: str) -> int:
    """启发式估算文本的 token 数。

    真实项目应使用目标模型对应的 tokenizer（如 tiktoken），使计数与
    Embedding 模型的上下文窗口精确对齐。这里采用确定性启发式：

    - 中文等 CJK 字符按 1 字 1 token 估算（对主流 tokenizer 偏保守）；
    - 其他字符（英文、数字、符号）按约 4 字符 1 token 统计。

    选择启发式的原因：零依赖、结果稳定可复现，教学与测试场景足够；
    误差通常在 ±20% 内，对切片大小这种"量级敏感、精确度不敏感"的用途无影响。
    """
    if not text:
        return 0
    cjk_count = sum(1 for ch in text if _CJK_RANGE[0] <= ch <= _CJK_RANGE[1])
    other_count = len(text) - cjk_count
    # (n + 3) // 4 是向上取整的整数除法，避免 len % 4 != 0 时低估。
    return cjk_count + (other_count + 3) // 4


def split_pages_into_chunks(pages: List[Tuple[int, str]]) -> List[Dict]:
    """把解析出的 (页码, 文本) 列表切成 chunk 列表。

    设计决策：按页独立切片，不跨页合并。
    - 好处：chunk 的 page 元数据始终精确，Day 10-11 的引用定位可以落到具体页；
    - 代价：页首/页尾可能出现偏小的 chunk，对检索质量影响有限。

    返回 [{"page": int, "content": str, "token_count": int}, ...]
    """
    chunks: List[Dict] = []
    for page, text in pages:
        if not text.strip():
            continue
        # 空白行（\n\n）是天然的语义边界：段落聚合比固定长度硬切更能保持语义完整。
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks.extend(_pack_paragraphs(paragraphs, page))
    return chunks


def _pack_paragraphs(paragraphs: List[str], page: int) -> List[Dict]:
    """按段落贪心聚合：依次装入 chunk，放不下就封口并用尾部内容构造 overlap。

    贪心策略保证除最后一片外每个 chunk 都接近 CHUNK_SIZE，
    同时段落本身不被切断（超长段落已在更早阶段降级处理）。
    """
    units: List[str] = []
    for paragraph in paragraphs:
        if estimate_tokens(paragraph) <= CHUNK_SIZE:
            units.append(paragraph)
        else:
            units.extend(_split_long_paragraph(paragraph))

    chunks: List[Dict] = []
    current: List[str] = []
    current_tokens = 0
    for unit in units:
        unit_tokens = estimate_tokens(unit)
        if current and current_tokens + unit_tokens > CHUNK_SIZE:
            chunks.append(_make_chunk(current, page))
            # overlap：下一片开头携带上一片的尾部，保证边界话题的上下文连续。
            # 预算 = CHUNK_SIZE - 新 unit 的 token 数：overlap 与新 unit 同处一片，
            # 二者合计不得超过 CHUNK_SIZE，否则新 chunk 会突破目标大小。
            overlap_head = _take_tail_for_overlap(current, CHUNK_SIZE - unit_tokens)
            current = overlap_head + [unit]
            current_tokens = sum(estimate_tokens(u) for u in current)
        else:
            current.append(unit)
            current_tokens += unit_tokens
    if current:
        chunks.append(_make_chunk(current, page))
    return chunks


def _split_long_paragraph(paragraph: str) -> List[str]:
    """超长段落先按句子边界聚合；无标点的超长句子按字符硬切兜底。

    两级降级策略：段落 → 句子 → 字符，尽可能保留最完整的语义边界。
    """
    sentences = [s.strip() for s in _SENTENCE_RE.findall(paragraph) if s.strip()]
    units: List[str] = []
    buffer: List[str] = []
    buffer_tokens = 0
    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)
        if sentence_tokens > CHUNK_SIZE:
            # 单句就超限（例如无标点的长代码或长串数字）：先清空缓冲，再硬切。
            if buffer:
                units.append("".join(buffer))
                buffer, buffer_tokens = [], 0
            units.extend(_hard_split(sentence))
            continue
        if buffer and buffer_tokens + sentence_tokens > CHUNK_SIZE:
            units.append("".join(buffer))
            buffer, buffer_tokens = [], 0
        buffer.append(sentence)
        buffer_tokens += sentence_tokens
    if buffer:
        units.append("".join(buffer))
    return units


def _hard_split(text: str) -> List[str]:
    """按 CHUNK_SIZE 字符硬切，作为没有任何语义边界可用的最后手段。

    按"字符数 = CHUNK_SIZE"切是按最坏情况（中文 1 字 1 token）保证不超限；
    对英文会产生偏小的切片，但该路径仅覆盖极端输入，可接受。
    """
    return [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]


def _take_tail_for_overlap(units: List[str], budget: int) -> List[str]:
    """从上一 chunk 尾部取不超过 budget、且约达 CHUNK_OVERLAP token 的内容。

    机制：按段落（unit）为单位回溯累加，达到 overlap 阈值即停。
    以 unit 为最小单位而不是按字符截断，保证重叠部分本身语义完整。
    budget 约束：overlap 会拼进下一个 chunk 的开头，必须给新 unit 留足空间，
    否则"overlap + unit"会超出 CHUNK_SIZE（初版实现正是踩了这个坑）。
    """
    tail: List[str] = []
    tail_tokens = 0
    for unit in reversed(units):
        unit_tokens = estimate_tokens(unit)
        if tail_tokens + unit_tokens > budget:
            break
        tail.insert(0, unit)
        tail_tokens += unit_tokens
        if tail_tokens >= CHUNK_OVERLAP:
            break
    return tail


def _make_chunk(units: List[str], page: int) -> Dict:
    """把一组文本单元组装成 chunk；段落间保留换行，维持原文结构便于引用展示。"""
    content = "\n".join(units)
    return {"page": page, "content": content, "token_count": estimate_tokens(content)}
