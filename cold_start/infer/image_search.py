import json
import math
import os
from io import BytesIO
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher
from PIL import Image


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


def _load_cached_results(cache_path: str, img_id: str, search_type: str = "image") -> List[Dict]:
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


def _process_image_size(image: Image.Image, max_pixels: int = 672 * 672 * 2, min_pixels: int = 512 * 512) -> Image.Image:
    """
    Process image with resize to meet size requirements.
    Reference: mm_rl_dataset.py process_image method.
    
    Args:
        image: PIL Image object
        max_pixels: Maximum allowed pixels (default: 672*672*2)
        min_pixels: Minimum required pixels (default: 512*512)
    
    Returns:
        Processed PIL Image object
    """
    # Resize if too large
    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height), resample=Image.Resampling.NEAREST)
    
    # Resize if too small
    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height), resample=Image.Resampling.NEAREST)
    
    return image


def _load_image_from_path(image_path: str, cache_path: str = None, img_id: str = None) -> Optional[Image.Image]:
    """
    Load image from path. Handles relative and absolute paths.
    All loaded images are resized to meet size requirements.
    
    Args:
        image_path: Path to the image (can be relative to images/{img_id}/)
        cache_path: Cache root directory
        img_id: Image ID for resolving relative paths
    
    Returns:
        PIL Image object (resized) or None if loading fails
    """
    # Helper: try open a single path
    def _open(p: str) -> Optional[Image.Image]:
        if os.path.isfile(p):
            try:
                img = Image.open(p)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                # Apply size processing
                img = _process_image_size(img)
                return img
            except Exception as e:
                print(f"[Image Load Error] Failed to load image {p}: {e}")
        return None

    # Try absolute path first
    if os.path.isabs(image_path):
        img = _open(image_path)
        if img:
            return img
    
    # If cache_path provided, try a few relative resolutions:
    # 1) cache_path + image_path (when image_path already includes images/{img_id}/...)
    # 2) cache_path/images/{img_id}/basename
    # 3) cache_path/images/{img_id}/image_path (preserve subfolders inside the sample)
    if cache_path:
        candidates = []
        if not os.path.isabs(image_path):
            candidates.append(os.path.join(cache_path, image_path))
        if img_id:
            candidates.append(os.path.join(cache_path, "images", img_id, os.path.basename(image_path)))
            candidates.append(os.path.join(cache_path, "images", img_id, image_path))
        for p in candidates:
            img = _open(p)
            if img:
                return img
    
    print(f"[Image Load Warning] Image not found: {image_path}")
    return None


def call_image_search(queries: List[Dict[str, Any]], img_id: str = None, cache_path: str = None):
    """
    Image search tool that retrieves results from offline cache.
    
    Args:
        queries: List of query objects, each containing:
            - "query": str, entity name to search for visual reference
            - "top_k": int (optional), number of reference images to return, default 1
        img_id: Image ID for the sample (e.g., "1001683")
        cache_path: Root directory of the cache (e.g., ".../cached_data/twitter_gmner_test")
    
    Returns:
        tool_returned_str: JSON formatted string for model consumption
        tool_returned_images: List of PIL Image objects representing search results
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
        cached_results = _load_cached_results(cache_path, img_id, "image")
        if cached_results:
            print(f"[Image Search] Loaded {len(cached_results)} cached results for img_id: {img_id}")
    
    results = []
    tool_returned_images = []
    processed_queries = set()  # Track processed cached queries for deduplication
    
    for query_obj in normalized_queries:
        try:
            query = query_obj.get("query", "")
            if not query:
                continue
            
            top_k = query_obj.get("top_k", 1)
            try:
                top_k = int(top_k) if top_k is not None else 1
                top_k = max(1, min(top_k, 10))
            except (ValueError, TypeError):
                top_k = 1
            
            # Try to find matching result in cache
            if cached_results:
                matched_result = _find_best_match(query, cached_results)
                if matched_result:
                    cached_query = matched_result.get("query", query)  # Get query from cache
                    
                    # Skip if we've already processed this cached query (deduplication)
                    if cached_query in processed_queries:
                        continue
                    
                    processed_queries.add(cached_query)
                    
                    status = matched_result.get("status", "failure")
                    cached_images = matched_result.get("result", {}).get("images", [])
                    kg_image_path = matched_result.get("kg_image_path", None)
                    
                    # Load images from cached paths
                    # 重要：kg_image_path排在第一位，然后是其他图片（最多top_k张）
                    loaded_images_data = []
                    images_to_load = []
                    kg_img_loaded = False
                    
                    # 1. kg_image排第一位（如果存在）
                    if kg_image_path and os.path.basename(kg_image_path):
                        kg_img = _load_image_from_path(kg_image_path, cache_path, img_id)
                        if kg_img:
                            tool_returned_images.append(kg_img)
                            kg_img_loaded = True
                    
                    # 2. 加载其他图片（取top_k张）
                    for img_info in cached_images[:top_k]:
                        img_path = img_info.get("path", "")
                        if img_path:
                            img = _load_image_from_path(img_path, cache_path, img_id)
                            if img:
                                tool_returned_images.append(img)
                                loaded_images_data.append({"path": img_path})
                    
                    # 构建返回结果（JSON格式）
                    result = {
                        "query": cached_query,  # Use the query from cache
                        "status": "success" if (loaded_images_data or kg_img_loaded) else status,
                        "result": {
                            "images": loaded_images_data  # 只包含非kg图片，与缓存一致
                        },
                        "kg_image_path": kg_image_path  # kg单独记录
                    }
                    results.append(result)
                    continue
            
            # Fallback: no cache match found
            result = {
                "query": query,
                "status": "failure",
                "result": {
                    "images": []
                },
                "kg_image_path": None
            }
            results.append(result)
            
        except Exception as e:
            print(f"[Image Search Warning] Error processing query {query_obj}: {e}")
            query_str = str(query_obj.get("query", query_obj) if isinstance(query_obj, dict) else query_obj)
            results.append({
                "query": query_str,
                "status": "failure",
                "result": {
                    "images": []
                },
                "kg_image_path": None
            })
    
    # Format results as JSON string for model
    tool_returned_str = json.dumps(results, ensure_ascii=False, indent=2)
    
    tool_stat = {
        "success": True,
        "num_queries": len(normalized_queries),
        "num_images": len(tool_returned_images),
        "from_cache": bool(cached_results),
    }

    return tool_returned_str, tool_returned_images, tool_stat
