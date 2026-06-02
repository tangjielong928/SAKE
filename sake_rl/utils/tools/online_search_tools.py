import json
import urllib.request
from io import BytesIO
from typing import Any, Dict, List, Optional

from PIL import Image

from sake_rl.utils.tools.image_search import _process_image_size
from sake_rl.utils.tools.online_search import OnlineSearchAPI


_ONLINE_SEARCH_API: Optional[OnlineSearchAPI] = None


def _get_online_search_api() -> OnlineSearchAPI:
    """Create one OnlineSearchAPI instance and reuse it across tool calls."""
    global _ONLINE_SEARCH_API
    if _ONLINE_SEARCH_API is None:
        _ONLINE_SEARCH_API = OnlineSearchAPI()
    return _ONLINE_SEARCH_API


def _normalize_queries(queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize queries with the same input tolerance as offline search tools."""
    if not isinstance(queries, list):
        if isinstance(queries, (str, dict)):
            queries = [queries]
        else:
            queries = [str(queries)]

    normalized_queries = []
    for query_item in queries:
        if isinstance(query_item, str):
            normalized_queries.append({"query": query_item})
        elif isinstance(query_item, dict):
            if "query" not in query_item:
                query_str = ""
                for _, value in query_item.items():
                    if isinstance(value, str) and value:
                        query_str = value
                        break
                if not query_str:
                    query_str = str(query_item)
                normalized_queries.append({"query": query_str})
            else:
                normalized_queries.append(query_item.copy())
        else:
            normalized_queries.append({"query": str(query_item)})

    return normalized_queries


def _coerce_top_k(value: Any, default: int, max_value: int) -> int:
    try:
        top_k = int(value) if value is not None else default
    except (ValueError, TypeError):
        top_k = default
    return max(1, min(top_k, max_value))


def _create_text_failure(query: str, message: str) -> Dict[str, Any]:
    return {
        "query": query,
        "status": "failure",
        "result": {
            "summary": message,
            "raw_snippets": [],
            "knowledge_graph": None,
        },
    }


def _create_image_failure(query: str) -> Dict[str, Any]:
    return {
        "query": query,
        "status": "failure",
        "result": {
            "images": [],
        },
        "kg_image_path": None,
    }


def _load_image_from_url(image_url: str) -> Optional[Image.Image]:
    try:
        if image_url.lower().split("?", 1)[0].endswith(".svg"):
            return None

        request = urllib.request.Request(
            image_url,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            image_bytes = response.read()

        if b"<svg" in image_bytes[:1000]:
            return None

        image = Image.open(BytesIO(image_bytes))
        image.verify()

        image = Image.open(BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")
        return _process_image_size(image)
    except Exception as exc:
        print(f"[Online Image Search Warning] Failed to load image URL {image_url}: {exc}")
        return None


def call_online_text_search(
    queries: List[Dict[str, Any]],
    img_id: str = None,
    original_text: str = None,
):
    """
    Online text search with the same output contract as call_text_search.

    Args:
        queries: List of query objects.
        img_id: Image ID for compatibility with the offline interface.
        original_text: Original text content (optional, for LLM summary context).

    Returns:
        tool_returned_str: JSON formatted string for model consumption.
        tool_stat: Dictionary with execution status and metadata.
    """
    normalized_queries = _normalize_queries(queries)
    for query_obj in normalized_queries:
        query_obj["top_k"] = _coerce_top_k(query_obj.get("top_k", 3), default=3, max_value=10)
        if original_text and "original_text" not in query_obj:
            query_obj["original_text"] = original_text
        query_obj.setdefault("llm_as_summary", True)

    try:
        results = _get_online_search_api().text_search(normalized_queries)
    except Exception as exc:
        print(f"[Online Text Search Warning] Online search unavailable: {exc}")
        results = [
            _create_text_failure(
                query_obj.get("query", ""),
                f"[Online Search Error] No search information found for '{query_obj.get('query', '')}'. Please reason with your own knowledge.",
            )
            for query_obj in normalized_queries
        ]

    tool_returned_str = json.dumps(results, ensure_ascii=False, indent=2)
    tool_stat = {
        "success": True,
        "num_queries": len(normalized_queries),
        "num_results": len([r for r in results if r.get("status") == "success"]),
        "from_cache": False,
    }

    return tool_returned_str, tool_stat


def call_online_image_search(
    queries: List[Dict[str, Any]],
    img_id: str = None,
):
    """
    Online image search with the same output contract as call_image_search.

    Args:
        queries: List of query objects.
        img_id: Image ID for compatibility with the offline interface.

    Returns:
        tool_returned_str: JSON formatted string for model consumption.
        tool_returned_images: List of PIL Image objects.
        tool_stat: Dictionary with execution status and metadata.
    """
    normalized_queries = _normalize_queries(queries)
    for query_obj in normalized_queries:
        query_obj["top_k"] = _coerce_top_k(query_obj.get("top_k", 1), default=1, max_value=10)

    try:
        raw_results = _get_online_search_api().image_search(normalized_queries)
    except Exception as exc:
        print(f"[Online Image Search Warning] Online search unavailable: {exc}")
        raw_results = [_create_image_failure(query_obj.get("query", "")) for query_obj in normalized_queries]

    results = []
    tool_returned_images = []

    for result in raw_results:
        loaded_images_data = []
        for image_info in result.get("result", {}).get("images", []):
            image_url = image_info.get("path", "")
            if not image_url:
                continue

            image = _load_image_from_url(image_url)
            if image:
                tool_returned_images.append(image)
                loaded_images_data.append({"path": image_url})

        results.append(
            {
                "query": result.get("query", ""),
                "status": "success" if loaded_images_data else "failure",
                "result": {
                    "images": loaded_images_data,
                },
                "kg_image_path": None,
            }
        )

    tool_returned_str = json.dumps(results, ensure_ascii=False, indent=2)
    tool_stat = {
        "success": True,
        "num_queries": len(normalized_queries),
        "num_images": len(tool_returned_images),
        "from_cache": False,
    }

    return tool_returned_str, tool_returned_images, tool_stat
