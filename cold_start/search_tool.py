import requests
import json
from config import Config

class SearchTool:
    @staticmethod
    def text_search(queries):
        if not Config.SERPER_API_KEY:
            return "[Mock Search] Results: Definitions found confirming the entities in context."
        
        results = []
        url = "https://google.serper.dev/search"
        headers = {'X-API-KEY': Config.SERPER_API_KEY, 'Content-Type': 'application/json'}

        for q_obj in queries:
            query = q_obj.get("query")
            if not query: continue
            try:
                payload = json.dumps({"q": query})
                response = requests.post(url, headers=headers, data=payload)
                data = response.json()
                snippets = [item.get('snippet') for item in data.get("organic", [])[:2]]
                results.append(f"Query: '{query}' -> {'; '.join(snippets)}")
            except Exception as e:
                results.append(f"Query: '{query}' -> Error: {e}")
        return "\n".join(results)

    @staticmethod
    def image_search(queries):
        if not Config.SERPER_API_KEY:
            return "[Mock Image Search] Results: Reference images found for visual comparison."

        results = []
        url = "https://google.serper.dev/images"
        headers = {'X-API-KEY': Config.SERPER_API_KEY, 'Content-Type': 'application/json'}

        for q_obj in queries:
            query = q_obj.get("query")
            if not query: continue
            try:
                payload = json.dumps({"q": query})
                response = requests.post(url, headers=headers, data=payload)
                data = response.json()
                images = [f"Title: {item.get('title')}" for item in data.get("images", [])[:2]]
                results.append(f"Query: '{query}' -> Found refs: {'; '.join(images)}")
            except Exception as e:
                results.append(f"Query: '{query}' -> Error: {e}")
        return "\n".join(results)