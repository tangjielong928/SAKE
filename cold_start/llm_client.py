from openai import OpenAI
from config import Config

class LLMClient:
    def __init__(self):
        self.client = OpenAI(
            api_key=Config.API_KEY,
            base_url=Config.BASE_URL
        )

    def call_model(self, system_prompt, user_content, image_base64=None):
        """
        Calls the LLM with text and optional image (base64 encoded).
        
        Args:
            system_prompt: System prompt string
            user_content: User content string
            image_base64: Base64 encoded image string (optional)
        """
        messages = [
            {"role": "system", "content": system_prompt}
        ]

        user_message = {"role": "user", "content": []}
        
        # Add Text
        user_message["content"].append({"type": "text", "text": user_content})
        
        # Add Image if present (base64 format)
        if image_base64:
            user_message["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                }
            })
            
        messages.append(user_message)

        try:
            response = self.client.chat.completions.create(
                model=Config.MODEL_NAME,
                messages=messages,
                temperature=0.7, # Slight creativity for reasoning
                max_tokens=2048
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM Call Error: {e}")
            return ""