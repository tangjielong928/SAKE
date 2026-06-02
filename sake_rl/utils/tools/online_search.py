#!/usr/bin/env python3
"""
Online Search API
"""
import json
import os
import requests
import time
import hashlib
import urllib.request
import io
from typing import List, Dict, Any
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv


def _load_project_env() -> None:
    """Load .env from the project root or the nearest parent directory."""
    current_file = Path(__file__).resolve()
    for directory in [current_file.parent, *current_file.parents]:
        env_path = directory / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            return

    load_dotenv()


_load_project_env()


class OnlineSearchAPI:
    """在线搜索接口 - 不使用缓存，直接在线搜索"""

    def __init__(self, max_retries: int = 3):
        """
        初始化在线搜索API

        Args:
            max_retries: API调用失败重试次数
        """
        # Serper API Keys
        api_keys_str = os.getenv("SERPER_API_KEYS") or os.getenv("SERPER_API_KEY")
        if not api_keys_str:
            raise ValueError("SERPER_API_KEY(S) not found in .env")

        self.api_keys = [k.strip() for k in api_keys_str.split(",") if k.strip()]
        self.current_key_index = 0
        self.all_keys_exhausted = False

        # LLM 配置
        self.llm_api_key = os.getenv("LLM_API_KEY")
        self.llm_base_url = os.getenv("LLM_BASE_URL")
        self.model_name = os.getenv("MODEL_NAME", "qwen3-max")

        self.max_retries = max_retries

        print(f"[Init] Loaded {len(self.api_keys)} Serper API key(s)")
        print(f"[Init] LLM: {self.model_name}")

    def text_search(self, queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        在线文本搜索

        Args:
            queries: [
                {
                    "query": "实体名称或问题",
                    "top_k": 3,  # 可选，默认3
                    "original_text": "原始文本内容",
                    "llm_as_summary": True  # 是否使用LLM做总结
                }
            ]

        Returns:
            [
                {
                    "query": "Nick Jonas",
                    "status": "success",
                    "result": {
                        "summary": "LLM生成的精简摘要",
                        "raw_snippets": ["snippet1", "snippet2", ...],
                        "knowledge_graph": {
                            "title": "Nick Jonas",
                            "type": "American singer-songwriter",
                            "description": "...",
                            "attributes": {...},
                            "image_url": "https://..."
                        }
                    }
                }
            ]
        """
        # 去重查询
        unique_queries = {}
        query_indices = {}  # 记录每个query对应的原始索引

        for idx, q in enumerate(queries):
            query_text = q.get("query", "")
            if query_text and query_text not in unique_queries:
                unique_queries[query_text] = q
                query_indices[query_text] = [idx]
            elif query_text:
                query_indices[query_text].append(idx)

        # 执行搜索
        unique_results = {}
        for query_text, query_obj in unique_queries.items():
            try:
                result = self._perform_text_search(query_obj)
                unique_results[query_text] = result
            except Exception as e:
                print(f"[Error] Text search failed for '{query_text}': {e}")
                unique_results[query_text] = self._create_failure_response(
                    query_text,
                    error_type="text"
                )

        # 按原始顺序重建结果
        results = []
        for q in queries:
            query_text = q.get("query", "")
            if query_text in unique_results:
                results.append(unique_results[query_text])
            else:
                results.append(self._create_failure_response("", error_type="text"))

        return results

    def _get_headers(self):
        """获取当前 API Key 的 headers"""
        return {
            'X-API-KEY': self.api_keys[self.current_key_index],
            'Content-Type': 'application/json'
        }

    def _switch_api_key(self):
        """切换到下一个 API Key"""
        if len(self.api_keys) <= 1:
            return False
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        print(f"\n[API] Switching to key {self.current_key_index + 1}/{len(self.api_keys)}")
        return True

    def _api_call(self, url, payload, debug=False):
        """带重试和 Key 轮换的 API 调用"""
        if self.all_keys_exhausted:
            return None

        tried_keys = set()

        for attempt in range(self.max_retries):
            try:
                if debug:
                    print(f"[DEBUG] Attempt {attempt + 1}/{self.max_retries}")

                response = requests.post(url, headers=self._get_headers(),
                                       data=json.dumps(payload), timeout=15)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.HTTPError as e:
                if debug:
                    print(f"[DEBUG] HTTPError: {response.status_code}")

                # 配额耗尽或频率限制
                is_quota_error = (
                    response.status_code in [400, 402, 429] or
                    "not enough credits" in response.text.lower() or
                    "insufficient credits" in response.text.lower()
                )

                if is_quota_error:
                    tried_keys.add(self.current_key_index)

                    if len(tried_keys) < len(self.api_keys):
                        self._switch_api_key()
                        continue
                    else:
                        self.all_keys_exhausted = True
                        print(f"\n[API] ⚠️  All {len(self.api_keys)} API keys exhausted!")
                        return None

                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
            except Exception as e:
                if debug:
                    print(f"[DEBUG] Exception: {type(e).__name__}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)

        return None

    def _call_llm(self, query: str, snippets: List[str], original_text: str = None,
                  knowledge_graph: Dict = None) -> str:
        """调用 LLM 生成摘要"""
        if not self.llm_api_key or not snippets:
            return snippets[0] if snippets else f"No information found for '{query}'."

        snippets_text = "\n".join([f"- {s}" for s in snippets])

        kg_text = ""
        if knowledge_graph:
            kg_text = f"\n\nKnowledge Graph:\n{json.dumps(knowledge_graph, indent=2, ensure_ascii=False)}"

        if original_text:
            prompt = f"""Based on search results about "{query}", provide a 1-2 sentence summary explaining what it is and how it relates to: {original_text}

Search results:
{snippets_text}{kg_text}

If you think the search results differ significantly from the original text, please provide your background knowledge about the query as the summary.

Summary:"""
        else:
            prompt = f"""Based on search results about "{query}", provide a 1-2 sentence summary.

Search results:
{snippets_text}{kg_text}

Summary:"""

        try:
            response = requests.post(
                f"{self.llm_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.llm_api_key}",
                        "Content-Type": "application/json"},
                json={"model": self.model_name,
                      "messages": [{"role": "user", "content": prompt}],
                      "temperature": 0.3, "max_tokens": 200},
                timeout=30
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()

        except requests.exceptions.HTTPError as e:
            if response.status_code == 400:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", {}).get("message", "")
                    error_type = error_data.get("error", {}).get("type", "")

                    if "inappropriate content" in error_msg.lower() or "data_inspection_failed" in error_type:
                        return snippets[0] if snippets else f"No information found for '{query}'."
                except:
                    pass

            print(f"\n[LLM] ⚠️  API error for '{query[:30]}...'")
            return snippets[0] if snippets else f"No information found for '{query}'."

        except Exception as e:
            print(f"\n[LLM] ⚠️  {type(e).__name__} for '{query[:30]}...'")
            return snippets[0] if snippets else f"No information found for '{query}'."

    def _perform_text_search(self, query_obj: Dict[str, Any]) -> Dict[str, Any]:
        """执行单个文本搜索"""
        query = query_obj.get("query", "")
        top_k = query_obj.get("top_k", 3)
        original_text = query_obj.get("original_text", "")
        llm_as_summary = query_obj.get("llm_as_summary", True)

        # 调用 Serper API
        url = "https://google.serper.dev/search"
        data = self._api_call(url, {"q": query, "num": top_k * 2})

        if not data:
            # API调用失败，直接返回失败
            return {
                "query": query,
                "status": "failure",
                "result": {
                    "summary": f"Search API failed for '{query}'. All API keys may be exhausted.",
                    "raw_snippets": [],
                    "knowledge_graph": None
                }
            }

        # 提取 Knowledge Graph
        knowledge_graph = None
        if "knowledgeGraph" in data:
            kg = data["knowledgeGraph"]
            knowledge_graph = {
                "title": kg.get("title", ""),
                "type": kg.get("type", ""),
                "description": kg.get("description", ""),
                "source": kg.get("source", {}),
                "attributes": kg.get("attributes", {}),
            }
            if "imageUrl" in kg:
                knowledge_graph["image_url"] = kg["imageUrl"]

        # 提取 snippets
        raw_snippets = []

        if knowledge_graph and knowledge_graph.get("description"):
            raw_snippets.append(knowledge_graph["description"])

        for item in data.get("organic", [])[:top_k]:
            snippet = item.get("snippet", "")
            if snippet and snippet not in raw_snippets:
                raw_snippets.append(snippet)

        if not raw_snippets:
            return {
                "query": query,
                "status": "failure",
                "result": {
                    "summary": f"No information found for '{query}'.",
                    "raw_snippets": [],
                    "knowledge_graph": knowledge_graph
                }
            }

        # 生成 summary
        if llm_as_summary:
            summary = self._call_llm(query, raw_snippets, original_text, knowledge_graph)
        else:
            summary = raw_snippets[0]

        return {
            "query": query,
            "status": "success",
            "result": {
                "summary": summary,
                "raw_snippets": raw_snippets[:top_k],
                "knowledge_graph": knowledge_graph
            }
        }

    def image_search(self, queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        在线图片搜索

        Args:
            queries: [
                {
                    "query": "实体名称",
                    "top_k": 1  # 可选，默认1
                }
            ]

        Returns:
            [
                {
                    "query": "Kevin Durant",
                    "status": "success",
                    "result": {
                        "images": [
                            {"path": "/path/to/image1.jpg"},
                            {"path": "/path/to/image2.jpg"}
                        ]
                    },
                    "kg_image_path": "/path/to/kg_image.jpg" or None
                }
            ]
        """
        # 去重查询
        unique_queries = {}
        for idx, q in enumerate(queries):
            query_text = q.get("query", "")
            if query_text and query_text not in unique_queries:
                unique_queries[query_text] = q

        # 执行搜索
        unique_results = {}
        for query_text, query_obj in unique_queries.items():
            try:
                result = self._perform_image_search(query_obj)
                unique_results[query_text] = result
            except Exception as e:
                print(f"[Error] Image search failed for '{query_text}': {e}")
                unique_results[query_text] = self._create_failure_response(
                    query_text,
                    error_type="image"
                )

        # 按原始顺序重建结果
        results = []
        for q in queries:
            query_text = q.get("query", "")
            if query_text in unique_results:
                results.append(unique_results[query_text])
            else:
                results.append(self._create_failure_response("", error_type="image"))

        return results

    def _download_image(self, url: str, save_path: Path) -> bool:
        """下载并验证图片"""
        try:
            if url.lower().endswith('.svg'):
                return False

            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                data = response.read()

                if b'<svg' in data[:1000]:
                    return False

                # 验证
                Image.open(io.BytesIO(data)).verify()

                with open(save_path, 'wb') as f:
                    f.write(data)

                # 再次验证
                img = Image.open(save_path)
                img.verify()
                return True
        except:
            return False

    def _perform_image_search(self, query_obj: Dict[str, Any]) -> Dict[str, Any]:
        """执行单个图片搜索（在线版本，不保存图片到磁盘）"""
        query = query_obj.get("query", "")
        top_k = query_obj.get("top_k", 1)

        # 调用 Serper Image Search API
        url = "https://google.serper.dev/images"
        data = self._api_call(url, {"q": query, "num": top_k * 3})

        if not data:
            return {
                "query": query,
                "status": "failure",
                "result": {"images": []},
                "kg_image_path": None
            }

        images = []
        for item in data.get("images", []):
            if len(images) >= top_k:
                break

            img_url = item.get("imageUrl", "")
            if not img_url:
                continue

            # 在线版本直接返回URL，不下载到本地
            images.append({
                "path": img_url  # 直接返回图片URL
            })

        if not images:
            return {
                "query": query,
                "status": "failure",
                "result": {"images": []},
                "kg_image_path": None
            }

        return {
            "query": query,
            "status": "success",
            "result": {"images": images},
            "kg_image_path": None  # 在线版本不处理KG图片
        }

    def _create_failure_response(self, query: str, error_type: str = "text") -> Dict[str, Any]:
        """
        创建失败响应

        对于文本搜索失败：返回错误信息
        对于图片搜索失败：返回空列表
        """
        if error_type == "text":
            return {
                "query": query,
                "status": "failure",
                "result": {
                    "summary": f"Search API failed for '{query}'. All API keys may be exhausted.",
                    "raw_snippets": [],
                    "knowledge_graph": None
                }
            }
        else:  # image
            return {
                "query": query,
                "status": "failure",
                "result": {
                    "images": []
                },
                "kg_image_path": None
            }


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 初始化API（自动从 .env 加载配置）
    api = OnlineSearchAPI()

    # Test 1: 文本搜索
    print("=" * 60)
    print("Test 1: Online Text Search")
    print("=" * 60)

    text_queries = [
        {
            "query": "Nick Jonas",
            "top_k": 3,
            "original_text": "Nick Jonas feared that his friend would die from drugs.",
            "llm_as_summary": True
        },
        {
            "query": "Kevin Durant",
            "top_k": 2
        }
    ]

    text_results = api.text_search(text_queries)
    print(json.dumps(text_results, indent=2, ensure_ascii=False))

    # Test 2: 图片搜索
    print("\n" + "=" * 60)
    print("Test 2: Online Image Search")
    print("=" * 60)

    image_queries = [
        {"query": "Nick Jonas", "top_k": 2},
        {"query": "Kevin Durant", "top_k": 1}
    ]

    image_results = api.image_search(image_queries)
    print(json.dumps(image_results, indent=2, ensure_ascii=False))

    # Test 3: 去重测试
    print("\n" + "=" * 60)
    print("Test 3: Deduplication Test")
    print("=" * 60)

    duplicate_queries = [
        {"query": "LeBron James", "top_k": 2},
        {"query": "LeBron James", "top_k": 2},  # 重复
        {"query": "Stephen Curry", "top_k": 1}
    ]

    results = api.text_search(duplicate_queries)
    print(f"Input: {len(duplicate_queries)} queries")
    print(f"Output: {len(results)} results")
    for r in results:
        print(f"  - {r['query']}: {r['status']}")
