import json
import os
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher


def _normalize_query(query: str) -> str:
    """Normalize query string for matching."""
    return query.strip().lower()


def _find_best_match(query: str, cached_queries: List[Dict]) -> Optional[Dict]:
    """
    Find the best matching cached result for a given query.
    First tries exact match, then falls back to fuzzy matching.
    """
    normalized_query = _normalize_query(query)
    
    # First, try exact match (case-insensitive)
    for cached in cached_queries:
        cached_query = cached.get("query", "")
        if _normalize_query(cached_query) == normalized_query:
            return cached
    
    # Fallback to fuzzy matching using SequenceMatcher
    best_match = None
    best_ratio = 0.0
    threshold = 0.6  # Minimum similarity threshold
    
    for cached in cached_queries:
        cached_query = cached.get("query", "")
        ratio = SequenceMatcher(None, normalized_query, _normalize_query(cached_query)).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = cached
    
    return best_match


def _load_cached_results(cache_path: str, img_id: str, search_type: str = "text") -> List[Dict]:
    """
    Load cached search results from JSON file.
    
    Args:
        cache_path: Root directory of the cache
        img_id: Image ID for the sample
        search_type: "text" or "image"
    
    Returns:
        List of cached search results
    """
    cache_file = os.path.join(cache_path, f"{search_type}_search", f"{img_id}_{search_type}_search.json")
    
    if not os.path.exists(cache_file):
        print(f"[Cache Warning] Cache file not found: {cache_file}")
        return []
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[Cache Error] Failed to load cache file {cache_file}: {e}")
        return []


def call_text_search(queries: List[Dict[str, Any]], img_id: str = None, cache_path: str = None, original_text: str = None):
    """
    Text search tool that retrieves results from offline cache.
    
    Args:
        queries: List of query objects, each containing:
            - "query": str, entity name or question to search
            - "top_k": int (optional), number of snippets to retrieve, default 3
        img_id: Image ID for the sample (e.g., "1001683")
        cache_path: Root directory of the cache (e.g., ".../cached_data/twitter_gmner_test")
        original_text: Original text content (optional, for fallback)
    
    Returns:
        tool_returned_str: JSON formatted string for model consumption
        tool_stat: Dictionary with execution status and metadata
    """
    # Handle case where queries is not a list
    if not isinstance(queries, list):
        if isinstance(queries, (str, dict)):
            queries = [queries]
        else:
            queries = [str(queries)]
    
    # Normalize queries: handle cases where LLM generates strings instead of dicts
    normalized_queries = []
    for query_item in queries:
        if isinstance(query_item, str):
            normalized_queries.append({"query": query_item})
        elif isinstance(query_item, dict):
            if "query" not in query_item:
                query_str = ""
                for key, value in query_item.items():
                    if isinstance(value, str) and value:
                        query_str = value
                        break
                if not query_str:
                    query_str = str(query_item)
                normalized_queries.append({"query": query_str})
            else:
                normalized_queries.append(query_item)
        else:
            normalized_queries.append({"query": str(query_item)})
    
    # Load cached results if cache_path and img_id are provided
    cached_results = []
    if cache_path and img_id:
        cached_results = _load_cached_results(cache_path, img_id, "text")
        if cached_results:
            print(f"[Text Search] Loaded {len(cached_results)} cached results for img_id: {img_id}")
    
    results = []
    processed_queries = set()  # Track processed cached queries for deduplication
    
    for query_obj in normalized_queries:
        try:
            query = query_obj.get("query", "")
            if not query:
                continue
            
            # Try to find matching result in cache
            if cached_results:
                matched_result = _find_best_match(query, cached_results)
                if matched_result:
                    # Use cached result with cached query
                    cached_query = matched_result.get("query", query)
                    
                    # Skip if we've already processed this cached query (deduplication)
                    if cached_query in processed_queries:
                        continue
                    
                    processed_queries.add(cached_query)
                    
                    result = {
                        "query": cached_query,  # Use the query from cache
                        "status": matched_result.get("status", "success"),
                        "result": matched_result.get("result", {})
                    }
                    # Ensure result has required fields
                    if "summary" not in result["result"]:
                        result["result"]["summary"] = f"No information found for '{cached_query}'."
                    if "raw_snippets" not in result["result"]:
                        result["result"]["raw_snippets"] = []
                    results.append(result)
                    continue
            
            # Fallback: generate placeholder result if no cache match
            result = {
                "query": query,
                "status": "failure",
                "result": {
                    "summary": f"[No Cache] No search information found for '{query}'. Please reason with your own knowledge.",
                    "raw_snippets": [],
                    "knowledge_graph": None
                }
            }
            results.append(result)
            
        except Exception as e:
            print(f"[Text Search Warning] Error processing query {query_obj}: {e}")
            query_str = str(query_obj.get("query", query_obj) if isinstance(query_obj, dict) else query_obj)
            results.append({
                "query": query_str,
                "status": "failure",
                "result": {
                    "summary": f"Error processing query: {str(e)}",
                    "raw_snippets": [],
                    "knowledge_graph": None
                }
            })
    
    # Format results as JSON string for model
    tool_returned_str = json.dumps(results, ensure_ascii=False, indent=2)
    
    tool_stat = {
        "success": True,
        "num_queries": len(normalized_queries),
        "num_results": len([r for r in results if r.get("status") == "success"]),
        "from_cache": bool(cached_results),
    }

    return tool_returned_str, tool_stat
