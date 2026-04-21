from langchain_core.prompts import PromptTemplate


GROUNDED_ANSWER_PROMPT = PromptTemplate.from_template(
    """You are a careful RAG assistant. Answer only with facts supported by the context.

If the context does not contain enough information, say exactly:
"I don't have enough information in the retrieved sources to answer that."

Return concise, useful prose. Do not mention unsupported assumptions.

Question:
{query}

Context:
{context}

Answer:"""
)


STRICT_REGENERATION_PROMPT = PromptTemplate.from_template(
    """You are validating a previous answer that may be unsupported.

Write a new answer using only the context below. If every material sentence is
not directly supported by the context, abstain with:
"I don't have enough information in the retrieved sources to answer that."

Question:
{query}

Context:
{context}

Grounded answer:"""
)

