"""Utility helpers."""

from typing import List


def format_sources(source_documents) -> List[str]:
    """Extract readable source labels from LangChain Document objects."""
    sources = []
    seen = set()
    for doc in source_documents:
        meta = doc.metadata
        label = None
        if "page" in meta:
            src = meta.get("source", "doc")
            import os
            src = os.path.basename(src)
            label = f"{src} · p.{int(meta['page']) + 1}"
        elif "source" in meta:
            import os
            label = os.path.basename(meta["source"])

        if label and label not in seen:
            sources.append(label)
            seen.add(label)

    return sources[:4]   # max 4 source tags
