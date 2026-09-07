"""集中管理 ChatPromptTemplate，避免普通和流式接口提示词漂移。"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是企业知识库助手。只能依据检索资料回答，资料不足时回答‘根据当前知识库无法确认’，不要编造。请用[1]、[2]标记引用。"),
    # 历史必须是结构化的 HumanMessage / AIMessage，而不是拼接到一个字符串中。
    # 这使 RunnableWithMessageHistory 可以统一加载、注入和写回多轮对话。
    MessagesPlaceholder(variable_name="history"),
    ("human", "问题：{question}\n\n检索资料：\n{context}"),
])
