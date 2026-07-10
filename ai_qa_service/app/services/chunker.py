import re
import tiktoken

class PageAwareChunker:
    """
    Splits document pages into token-bounded chunks while respecting sentence boundaries,
    tracking spanned pages, and generating character-based excerpts.
    """
    def __init__(self, target_tokens: int = 500, overlap_tokens: int = 50):
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def _split_into_sentences(self, text: str) -> list[str]:
        """Splits text on sentence endings (.!?), keeping punctuation and cleaning spaces."""
        if not text:
            return []

        cleaned_text = re.sub(r'\s+', ' ', text)
        sentences = re.split(r'(?<=[.!?])\s+', cleaned_text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_document(self, pages_data: list[dict]) -> list[dict]:
        """
        Groups sentences into overlapping chunks based on target token count.

        Args:
            pages_data: list of dicts with keys "page" (int) and "text" (str)

        Returns:
            list[dict]: A list of chunks:
                [{"chunk_text": str, "pages": list[int], "excerpt": str}]
        """

        flat_sentences = []
        for page_data in pages_data:
            page_num = page_data["page"]
            text = page_data["text"]
            sentences = self._split_into_sentences(text)
            for s in sentences:
                tokens = len(self.encoding.encode(s))
                flat_sentences.append({
                    "text": s,
                    "page": page_num,
                    "tokens": tokens
                })

        if not flat_sentences:
            return []

        chunks = []
        n = len(flat_sentences)
        i = 0

        while i < n:
            chunk_sentences = []
            chunk_tokens = 0


            j = i
            while j < n and chunk_tokens + flat_sentences[j]["tokens"] <= self.target_tokens:
                chunk_sentences.append(flat_sentences[j])
                chunk_tokens += flat_sentences[j]["tokens"]
                j += 1


            if not chunk_sentences:
                chunk_sentences.append(flat_sentences[i])
                chunk_tokens += flat_sentences[i]["tokens"]
                j = i + 1


            chunk_text = " ".join([s["text"] for s in chunk_sentences])
            pages = sorted(list(set([s["page"] for s in chunk_sentences])))
            excerpt = chunk_text[:100]

            chunks.append({
                "chunk_text": chunk_text,
                "pages": pages,
                "excerpt": excerpt
            })

            if j >= n:
                break


            overlap_sum = 0
            backtrack_idx = j - 1
            while backtrack_idx >= i and overlap_sum + flat_sentences[backtrack_idx]["tokens"] <= self.overlap_tokens:
                overlap_sum += flat_sentences[backtrack_idx]["tokens"]
                backtrack_idx -= 1


            next_i = max(i + 1, backtrack_idx + 1)
            i = next_i

        return chunks
