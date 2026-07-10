from app.core.config import settings
from app.core.logging import logger
from app.models.schemas import StructuredQAResponse, Citation, ConfidenceLevel
from app.services.llm_provider import BaseLLMProvider, OpenAILLMProvider

class LLMService:
    """
    RAG service that handles context formatting, threshold checks, LLM invocation,
    and post-hoc validation of LLM citations against retrieved page metadata.
    """
    def __init__(self, provider: BaseLLMProvider = None):

        self.provider = provider or OpenAILLMProvider()

    def generate_answer(self, query: str, retrieved_chunks: list[dict]) -> StructuredQAResponse:
        """
        Retrieves matching chunks and builds prompt context. If similarity checks pass,
        it calls the LLM, validates its citations, and maps the confidence score.

        Args:
            query: The user question.
            retrieved_chunks: A list of dicts retrieved from Chroma similarity search.

        Returns:
            StructuredQAResponse: A validated response with answers, citations, and status.
        """

        max_similarity = max([c["similarity"] for c in retrieved_chunks]) if retrieved_chunks else 0.0


        if not retrieved_chunks or max_similarity < settings.SIM_THRESHOLD:
            logger.info(
                "Skipping LLM call: similarity did not clear threshold.",
                extra={"extra_fields": {
                    "max_similarity": float(f"{max_similarity:.4f}"),
                    "sim_threshold": settings.SIM_THRESHOLD,
                    "retrieved_chunks_count": len(retrieved_chunks)
                }}
            )
            return StructuredQAResponse(
                answer="Answer not found in the provided document.",
                citations=[],
                found=False,
                confidence=ConfidenceLevel.NONE
            )


        if max_similarity >= 0.8:
            confidence = ConfidenceLevel.HIGH
        elif max_similarity >= 0.6:
            confidence = ConfidenceLevel.MEDIUM
        else:
            confidence = ConfidenceLevel.LOW


        context_parts = []
        retrieved_pages = set()

        for idx, chunk in enumerate(retrieved_chunks):
            for page in chunk["pages"]:
                retrieved_pages.add(page)

            context_parts.append(
                f"<context index=\"{idx}\" pages=\"{chunk['pages']}\">\n"
                f"{chunk['chunk_text']}\n"
                f"</context>"
            )
        context_block = "\n".join(context_parts)


        system_prompt = (
            f"You are a precise document Q&A assistant (System Prompt Version: {settings.PROMPT_VERSION}).\n"
            "Your task is to answer the user's question ONLY using the facts provided in the <context> blocks below.\n"
            "If the context does not contain the answer, you must set found=false and explain that the answer is not present.\n"
            "For every statement you make, you MUST provide at least one citation containing the exact page_number (int) "
            "and the exact excerpt/quote (str) from the context supporting the statement.\n"
            "Strictly follow the tools schema to format your output."
        )

        prompt = (
            f"Context:\n{context_block}\n\n"
            f"Question: {query}"
        )


        try:
            llm_response: StructuredQAResponse = self.provider.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                response_schema=StructuredQAResponse
            )
        except Exception as e:
            logger.error(
                f"LLM Generation failed with error: {str(e)}",
                exc_info=True
            )
            return StructuredQAResponse(
                answer="Failed to generate an answer due to an internal LLM error.",
                citations=[],
                found=False,
                confidence=ConfidenceLevel.NONE
            )

        if not llm_response.found:
            return StructuredQAResponse(
                answer=llm_response.answer,
                citations=[],
                found=False,
                confidence=ConfidenceLevel.NONE
            )


        validated_citations = []
        for citation in llm_response.citations:
            page = citation.page_number
            if page in retrieved_pages:
                validated_citations.append(citation)
            else:

                logger.warning(
                    f"AI mistake caught: hallucinated page number {page} which was not in retrieved chunks.",
                    extra={"extra_fields": {
                        "hallucinated_page": page,
                        "retrieved_pages": list(retrieved_pages),
                        "excerpt": citation.excerpt
                    }}
                )

        llm_response.citations = validated_citations
        llm_response.confidence = confidence

        return llm_response
