import os


def _qdrant_search(self, collection_name: str, query_vector=None, limit: int = 5, **kwargs):
    return self.query_points(collection_name=collection_name, query=query_vector, limit=limit, **kwargs)


def get_qdrant():
    from qdrant_client import QdrantClient

    QdrantClient.search = _qdrant_search
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
