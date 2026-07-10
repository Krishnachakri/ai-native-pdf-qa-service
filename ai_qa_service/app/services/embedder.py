from sentence_transformers import SentenceTransformer
from app.core.config import settings

class Embedder:
    """
    Singleton wrapper for the sentence-transformers embedding model.
    Generates local vector embeddings using the configured model.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Embedder, cls).__new__(cls, *args, **kwargs)

            cls._instance.model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        return cls._instance

    def encode(self, texts: list[str], batch_size: int = 20) -> list[list[float]]:
        """
        Batch encodes a list of text strings into vector embeddings.

        Args:
            texts: A list of strings to embed.
            batch_size: Number of texts processed at once in inference.

        Returns:
            list[list[float]]: A list of 384-dimensional floating point vectors.
        """
        if not texts:
            return []


        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False
        )
        return embeddings.tolist()
