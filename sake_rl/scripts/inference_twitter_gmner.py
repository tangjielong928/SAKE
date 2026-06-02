#!/usr/bin/env python3
"""
Twitter GMNER Inference Script
Twitter GMNER multi-turn inference
Supports both Transformers and OpenAI API inference modes.
"""

import argparse
import json
import os
import re
import sys
import base64
import math
from io import BytesIO
from PIL import Image
from typing import Tuple, List, Optional
from sake_rl.utils.tools.image_search import call_image_search
from sake_rl.utils.tools.text_search import call_text_search

# Import evaluation modules
# Add evaluate directory to path for relative imports
evaluate_dir = os.path.join(os.path.dirname(__file__), '..', 'evaluate')
if evaluate_dir not in sys.path:
    sys.path.insert(0, evaluate_dir)
from evaluate import EntityEvaluator
from fg_label_evaluate import fine_to_label_dict
from utils import load_cot_prediction, read_json

# Conditional imports for different inference modes
try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# Image processing constants (same as utils.py)
DEFAULT_MAX_PIXELS = 672 * 672 * 2
DEFAULT_MIN_PIXELS = 512 * 512


def parse_args():
    parser = argparse.ArgumentParser(description="Twitter GMNER Multi-turn Inference")
    
    # Inference mode selection
    parser.add_argument('--inference_mode', type=str, default='transformers',
                        choices=['transformers', 'openai'],
                        help='Inference mode: "transformers" for local model, "openai" for API')
    
    # Transformers mode arguments
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained model checkpoint (for transformers mode)')
    
    # OpenAI mode arguments
    parser.add_argument('--api_key', type=str, default=None,
                        help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--api_base', type=str, default=None,
                        help='OpenAI API base URL (for compatible APIs like vLLM)')
    parser.add_argument('--model_name', type=str, default='gpt-4o',
                        help='Model name for OpenAI API')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Temperature for OpenAI API sampling')
    
    # Common arguments
    parser.add_argument('--test_data', type=str, required=True,
                        help='Path to test JSON file (twitter_gmner format)')
    parser.add_argument('--image_root', type=str, 
                        default="/data/xx/SAKE-Dev/sake_rl/data/Twitter-GMNER/whole_image",
                        help='Root directory for images')
    parser.add_argument('--search_cache_path', type=str,
                        default="xxx/data/twitter_cached_search/twitter_gmner_test",
                        help='Root directory for search cache')
    parser.add_argument('--output_file', type=str, default='predictions.json',
                        help='Output file for predictions')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Start index in the dataset')
    parser.add_argument('--end_idx', type=int, default=None,
                        help='End index in the dataset (None means process all)')
    parser.add_argument('--max_new_tokens', type=int, default=2048,
                        help='Maximum new tokens to generate per turn')
    parser.add_argument('--round_1_prompt', type=str,
                        default="/data/xx/SAKE-Dev/sake_rl/prompts/Twitter-GMNER-Prompt/round_1_user_prompt_1.txt",
                        help='Path to round 1 user prompt file')
    parser.add_argument('--after_text_search_prompt', type=str,
                        default="/data/xx/SAKE-Dev/sake_rl/prompts/Twitter-GMNER-Prompt/after_text_search_prompt_1.txt",
                        help='Path to after text search prompt file')
    parser.add_argument('--after_image_search_prompt', type=str,
                        default="/data/xx/SAKE-Dev/sake_rl/prompts/Twitter-GMNER-Prompt/after_image_search_prompt_1.txt",
                        help='Path to after image search prompt file')
    parser.add_argument('--gt_path', type=str, default=None,
                        help='Path to ground truth JSON file for evaluation (optional)')
    parser.add_argument('--evaluate', action='store_true',
                        help='Run evaluation after inference if gt_path is provided')

    parser.add_argument('--task_type', type=str, default='gmner',
                        help='Task(gmner/fmnerg) for evaluation (optional)')

    args = parser.parse_args()
    
    # Validate arguments based on inference mode
    if args.inference_mode == 'transformers':
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers mode requires 'transformers' and 'qwen_vl_utils' packages")
        if args.model_path is None:
            raise ValueError("--model_path is required for transformers mode")
    elif args.inference_mode == 'openai':
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI mode requires 'openai' package. Install with: pip install openai")
        if args.api_key is None:
            args.api_key = os.environ.get('OPENAI_API_KEY')
        if args.api_key is None:
            raise ValueError("--api_key is required for openai mode (or set OPENAI_API_KEY env var)")
    
    return args


def load_model_and_processor(model_path: str):
    """Load model and processor for Transformers mode"""
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def create_openai_client(api_key: str, api_base: Optional[str] = None):
    """Create OpenAI client"""
    if api_base:
        client = OpenAI(api_key=api_key, base_url=api_base)
    else:
        client = OpenAI(api_key=api_key)
    return client


def generate_response_openai(client, model_name: str, messages: list, 
                              max_tokens: int = 2048, temperature: float = 0.0) -> str:
    """Generate response using OpenAI API"""
    # Convert messages to OpenAI format
    openai_messages = []
    
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        
        if isinstance(content, str):
            # Simple text content
            openai_messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            # Multimodal content with images
            openai_content = []
            for item in content:
                if item["type"] == "text":
                    openai_content.append({
                        "type": "text",
                        "text": item["text"]
                    })
                elif item["type"] == "image":
                    # Handle image - already in base64 format
                    image_url = item.get("image", "")
                    if not image_url.startswith("data:"):
                        image_url = f"data:image/jpeg;base64,{image_url}"
                    openai_content.append({
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    })
            openai_messages.append({"role": role, "content": openai_content})
        else:
            openai_messages.append({"role": role, "content": str(content)})
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=openai_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"  >>> OpenAI API error: {e}")
        raise


def process_image(image_path: str, max_pixels: int = DEFAULT_MAX_PIXELS, min_pixels: int = DEFAULT_MIN_PIXELS) -> Tuple[Image.Image, Tuple[int, int], Tuple[int, int]]:
    """
    处理图像并进行缩放，返回处理后的图像和尺寸信息。
    
    Returns:
        processed_image: PIL Image 对象
        original_size: (width, height) 原始尺寸
        processed_size: (width, height) 处理后的尺寸
    """
    image = Image.open(image_path)
    
    # 存储原始尺寸
    original_width, original_height = image.width, image.height
    original_size = (original_width, original_height)
    
    # 如果太大则缩放
    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height), resample=Image.Resampling.NEAREST)
    
    # 如果太小则放大
    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height), resample=Image.Resampling.NEAREST)
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    processed_size = (image.width, image.height)
    
    return image, original_size, processed_size


def pil_image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")


def unscale_box2d(box: List[int], original_size: Tuple[int, int], processed_size: Tuple[int, int]) -> List[int]:
    """
    将模型输出的box2d从处理后尺寸还原到原始图像尺寸。
    这是scale_box2d的逆操作。
    
    Args:
        box: [x1, y1, x2, y2] 处理后图像上的坐标
        original_size: (width, height) 原始图像尺寸
        processed_size: (width, height) 处理后的图像尺寸
    
    Returns:
        还原到原始图像的 box2d 坐标
    """
    if not box or len(box) != 4:
        return box
    
    orig_w, orig_h = original_size
    proc_w, proc_h = processed_size
    
    # 计算还原比例（与scale_box2d相反）
    scale_x = orig_w / proc_w
    scale_y = orig_h / proc_h
    
    x1, y1, x2, y2 = box
    unscaled_box = [
        int(x1 * scale_x),
        int(y1 * scale_y),
        int(x2 * scale_x),
        int(y2 * scale_y)
    ]
    
    return unscaled_box


def load_image_from_path(image_path: str) -> str:
    """Load image and convert to base64 (deprecated, use process_image instead)"""
    try:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        return base64.b64encode(image_bytes).decode('utf-8')
    except Exception as e:
        raise RuntimeError(f"Failed to load image from {image_path}: {e}")


def pil_images_to_base64_list(images):
    """Convert PIL images to base64 list"""
    base64_list = []
    for img in images:
        buffered = BytesIO()
        img.save(buffered, format=img.format if img.format else "JPEG")
        img_bytes = buffered.getvalue()
        base64_str = base64.b64encode(img_bytes).decode("utf-8")
        base64_list.append(base64_str)
    return base64_list


def generate_response(model, processor, messages, max_new_tokens=2048):
    """Generate model response"""
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]


def extract_tag_content(text: str, tag: str) -> str:
    """Extract content from XML-like tags"""
    pattern = f'<{tag}>(.*?)</{tag}>'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


def build_interleaved_content(text_with_placeholders: str, images_base64: list) -> list:
    """
    Build interleaved content by replacing <image> placeholders with actual images.
    Similar to llm_client.py's _build_interleaved_content method.
    
    Args:
        text_with_placeholders: Text containing <image> placeholders
        images_base64: List of base64 encoded images in order
    
    Returns:
        List of content items with text and images interleaved
    """
    # Count <image> placeholders
    num_placeholders = text_with_placeholders.count('<image>')
    num_images = len(images_base64)
    
    # Assert that the number of placeholders matches the number of images
    assert num_placeholders == num_images, (
        f"Mismatch: Found {num_placeholders} <image> placeholder(s) in text, "
        f"but {num_images} image(s) were provided. "
        f"Please ensure the number of <image> tags matches the number of images."
    )
    
    content_list = []
    image_idx = 0
    
    # Split text by <image> placeholder
    parts = re.split(r'(<image>)', text_with_placeholders)
    
    for part in parts:
        if part == '<image>':
            # Insert image at this position
            if image_idx < len(images_base64):
                content_list.append({
                    "type": "image",
                    "image": f"data:image/jpeg;base64,{images_base64[image_idx]}"
                })
                image_idx += 1
            else:
                print(f"[Warning] Not enough images provided. Expected at least {image_idx + 1} images.")
        elif part:  # Non-empty text part
            content_list.append({
                "type": "text",
                "text": part
            })
    
    # Check if all images were used
    if image_idx < len(images_base64):
        print(f"[Warning] Only {image_idx} out of {len(images_base64)} images were used. Check <image> placeholders.")
    
    return content_list


def parse_queries_json(queries_str: str) -> list:
    """Parse queries JSON string and normalize"""
    try:
        queries = json.loads(queries_str)
        
        # Normalize queries
        if not isinstance(queries, list):
            queries = [queries]
        
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
        
        return normalized_queries
    except json.JSONDecodeError:
        return []


def format_text_search_results(results_json: list) -> str:
    """Format text search results for model"""
    formatted = "[Text Search Results] Below are the search results for each query:\n"
    for result in results_json:
        query = result.get("query", "")
        status = result.get("status", "unknown")
        if status == "success":
            summary = result.get("result", {}).get("summary", "")
            formatted += f"\nQuery: {query}\nSummary: {summary}\n"
        else:
            error_msg = result.get("result", {}).get("summary", "No information found.")
            formatted += f"\nQuery: {query}\nSummary: {error_msg}\n"
    return formatted


def format_image_search_results(results_json: list) -> str:
    """Format image search results for model with <image> placeholders"""
    formatted = "[Image Search Results] Below are the visual references for each query:\n"
    total_images = 0
    
    for result in results_json:
        query = result.get("query", "")
        status = result.get("status", "unknown")
        kg_image_path = result.get("kg_image_path", None)
        
        if status == "success":
            images = result.get("result", {}).get("images", [])
            num_images = len(images)
            if kg_image_path and os.path.basename(kg_image_path):
                num_images += 1
            
            total_images += num_images
            formatted += f"\nQuery: {query}\nFound {num_images} reference image(s):\n"
            
            # Add <image> placeholders for each image
            for index in range(num_images):
                formatted += f"  Image {index+1}: <image>\n"
            
            if num_images == 0:
                formatted += "  No images found.\n"
        else:
            formatted += f"\nQuery: {query}\nNo images found.\n"
    
    return formatted, total_images


def run_inference(sample, image_root, search_cache_path, 
                  round_1_prompt, after_text_search_prompt, after_image_search_prompt,
                  max_new_tokens=2048,
                  # Transformers mode params
                  model=None, processor=None,
                  # OpenAI mode params
                  openai_client=None, model_name=None, temperature=0.0,
                  inference_mode='transformers'):
    """Run multi-turn inference for a single sample
    
    Args:
        sample: Data sample dict
        image_root: Root directory for images
        search_cache_path: Path to search cache
        round_1_prompt: Initial prompt
        after_text_search_prompt: Prompt after text search
        after_image_search_prompt: Prompt after image search
        max_new_tokens: Maximum tokens to generate
        model: Transformers model (for transformers mode)
        processor: Transformers processor (for transformers mode)
        openai_client: OpenAI client (for openai mode)
        model_name: Model name for OpenAI API
        temperature: Sampling temperature for OpenAI
        inference_mode: 'transformers' or 'openai'
    
    Returns:
        response: Model's final response
        original_size: Original image size (width, height)
        processed_size: Processed image size (width, height)
    """
    
    # Define generate function based on mode
    def generate(messages):
        if inference_mode == 'transformers':
            return generate_response(model, processor, messages, max_new_tokens)
        else:  # openai mode
            return generate_response_openai(openai_client, model_name, messages, 
                                           max_new_tokens, temperature)
    
    # Extract sample info
    user_text = " ".join(sample.get("tokens", []))
    img_id = sample.get("img_id", "")
    
    # Load and process original image
    # Check if img_id already has an extension
    if img_id.lower().endswith(('.jpg', '.jpeg', '.png')):
        # img_id already has extension, use it directly
        image_path = os.path.join(image_root, img_id)
    else:
        # Try adding extensions
        image_path = None
        for ext in ['.jpg', '.png', '.jpeg']:
            test_path = os.path.join(image_root, f"{img_id}{ext}")
            if os.path.exists(test_path):
                image_path = test_path
                break
        
        # If no file found, default to .jpg
        if image_path is None:
            image_path = os.path.join(image_root, f"{img_id}.jpg")
    
    # Process image with resize and get size info
    processed_image, original_size, processed_size = process_image(image_path)
    image_base64 = pil_image_to_base64(processed_image)
    
    print(f"  >>> Image: original={original_size}, processed={processed_size}")
    
    # Initialize messages
    messages = []
    
    # Round 1: Initial analysis
    print(f"  >>> Round 1: Initial analysis...")
    # Keep <image> placeholder and use interleaved format
    user_content_1 = round_1_prompt + f"**Original social media post:**\n{user_text}"
    
    # Build interleaved content with image at placeholder position
    content_list_1 = build_interleaved_content(user_content_1, [image_base64])
    
    messages.append({
        "role": "user",
        "content": content_list_1
    })
    
    response_1 = generate(messages)
    print(f"  >>> Response 1: {response_1}")
    
    messages.append({
        "role": "assistant",
        "content": response_1
    })
    
    # Check for answer
    if extract_tag_content(response_1, "answer"):
        return response_1, original_size, processed_size
    
    # Round 2: Handle text search
    text_search_content = extract_tag_content(response_1, "text_search")
    if text_search_content:
        print(f"  >>> Round 2: Text search...")
        queries = parse_queries_json(text_search_content)
        
        if queries:
            # Call text search
            tool_returned_str, tool_stat = call_text_search(
                queries=queries,
                img_id=img_id.split('.')[0],
                cache_path=search_cache_path,
                original_text=user_text
            )
            
            # Parse and format results
            results_json = json.loads(tool_returned_str)
            formatted_results = format_text_search_results(results_json)
            
            # Build next user message (keep <image> placeholder)
            user_content_2 = (
                f"<information>{formatted_results}</information>\n"
                f"Here are the image and original social media post for your task:\n<image>\n{user_text}\n"
                f"{after_text_search_prompt}"
            )
            
            # Build interleaved content with image at placeholder position
            content_list_2 = build_interleaved_content(user_content_2, [image_base64])
            
            messages.append({
                "role": "user",
                "content": content_list_2
            })
            
            response_2 = generate(messages)
            print(f"  >>> Response 2: {response_2}")
            
            messages.append({
                "role": "assistant",
                "content": response_2
            })
            
            # Check for answer
            if extract_tag_content(response_2, "answer"):
                return response_2, original_size, processed_size
            
            # Check for image search
            image_search_content = extract_tag_content(response_2, "image_search")
            if image_search_content:
                queries = parse_queries_json(image_search_content)
            else:
                return response_2, original_size, processed_size
        else:
            return response_1, original_size, processed_size
    else:
        # Check for image search directly from round 1
        image_search_content = extract_tag_content(response_1, "image_search")
        if not image_search_content:
            return response_1, original_size, processed_size
        
        queries = parse_queries_json(image_search_content)
    
    # Round 3: Handle image search
    if queries:
        print(f"  >>> Round 3: Image search...")
        
        # Call image search
        tool_returned_str, tool_returned_images, tool_stat = call_image_search(
            queries=queries,
            img_id=img_id.split('.')[0],
            cache_path=search_cache_path
        )
        
        # Parse and format results
        results_json = json.loads(tool_returned_str)
        formatted_results, num_images = format_image_search_results(results_json)
        
        # Convert search images to base64
        search_images_base64 = pil_images_to_base64_list(tool_returned_images)
        
        # Build next user message with multiple images (keep <image> placeholders)
        user_content_3 = (
            f"<information>{formatted_results}</information>\n"
            f"Here are the image and original social media post for your task:\n<image>\n{user_text}\n"
            f"{after_image_search_prompt}"
        )
        
        # Build interleaved content with all images in order (search images + original image)
        # The order matches the <image> placeholders: search results first, then original image
        all_images_base64 = search_images_base64 + [image_base64]
        content_list_3 = build_interleaved_content(user_content_3, all_images_base64)
        
        messages.append({
            "role": "user",
            "content": content_list_3
        })
        
        response_3 = generate(messages)
        print(f"  >>> Response 3: {response_3}")
        
        return response_3, original_size, processed_size
    
    return response_1, original_size, processed_size


def extract_answer(response: str) -> list:
    """Extract answer from response"""
    answer_content = extract_tag_content(response, "answer")
    if answer_content:
        try:
            return json.loads(answer_content)
        except json.JSONDecodeError:
            print(f"  Warning: Failed to parse answer JSON")
            return []
    return []


def restore_box2d_to_original_size(entities: list, original_size: Tuple[int, int], processed_size: Tuple[int, int]) -> list:
    """
    将模型输出的实体中的box2d从处理后尺寸还原到原始图像尺寸。
    
    Args:
        entities: 模型输出的实体列表
        original_size: (width, height) 原始图像尺寸
        processed_size: (width, height) 处理后的图像尺寸
    
    Returns:
        还原后的实体列表
    """
    if original_size == processed_size:
        # 如果尺寸相同，无需还原
        return entities
    
    restored_entities = []
    for entity in entities:
        restored_entity = entity.copy()
        
        # 还原 box2d
        if "box2d" in entity and entity["box2d"]:
            restored_entity["box2d"] = unscale_box2d(entity["box2d"], original_size, processed_size)
        
        # 兼容 region_box 字段
        if "region_box" in entity and entity["region_box"]:
            restored_entity["region_box"] = unscale_box2d(entity["region_box"], original_size, processed_size)
        
        restored_entities.append(restored_entity)
    
    return restored_entities


def main():
    args = parse_args()
    
    # Load test data
    print(f">>> Loading test data from: {args.test_data}")
    with open(args.test_data, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # Determine processing range
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx is not None else len(test_data)
    print(f">>> Processing samples from index {start_idx} to {end_idx}")
    
    # Initialize model/client based on inference mode
    model, processor = None, None
    openai_client = None
    
    print(f">>> Inference mode: {args.inference_mode}")
    
    if args.inference_mode == 'transformers':
        print(f">>> Loading model from: {args.model_path}")
        model, processor = load_model_and_processor(args.model_path)
    else:  # openai mode
        print(f">>> Creating OpenAI client for model: {args.model_name}")
        if args.api_base:
            print(f">>> Using custom API base: {args.api_base}")
        openai_client = create_openai_client(args.api_key, args.api_base)
    
    # Load prompts
    print(f">>> Loading prompts...")
    with open(args.round_1_prompt, 'r', encoding='utf-8') as f:
        round_1_prompt = f.read()
        # Remove the trailing "**Input Data:**\n<image>\n" part
        if "**Input Data:**" in round_1_prompt:
            round_1_prompt = round_1_prompt.split("**Input Data:**")[0] + "---\n\n**Input Data:** Here are the image and original social media post for your task:\n<image>\n"
    
    with open(args.after_text_search_prompt, 'r', encoding='utf-8') as f:
        after_text_search_prompt = f.read()
    
    with open(args.after_image_search_prompt, 'r', encoding='utf-8') as f:
        after_image_search_prompt = f.read()
    
    # Process samples
    predictions = []
    
    for idx in range(start_idx, end_idx):
        sample = test_data[idx]
        img_id = sample.get("img_id", f"sample_{idx}")
        
        print(f"\n{'='*60}")
        print(f">>> [{idx+1}/{end_idx}] Processing: {img_id}")
        
        try:
            final_response, original_size, processed_size = run_inference(
                sample=sample,
                image_root=args.image_root,
                search_cache_path=args.search_cache_path,
                round_1_prompt=round_1_prompt,
                after_text_search_prompt=after_text_search_prompt,
                after_image_search_prompt=after_image_search_prompt,
                max_new_tokens=args.max_new_tokens,
                # Transformers mode params
                model=model,
                processor=processor,
                # OpenAI mode params
                openai_client=openai_client,
                model_name=args.model_name,
                temperature=args.temperature,
                inference_mode=args.inference_mode
            )
            
            # Extract answer
            answer = extract_answer(final_response)
            
            # Restore box2d to original image size
            if original_size != processed_size and answer:
                print(f"  >>> Restoring box2d: {processed_size} -> {original_size}")
                answer = restore_box2d_to_original_size(answer, original_size, processed_size)
            
            prediction = {
                "img_id": img_id,
                "tokens": sample.get("tokens", []),
                "prediction": answer,
                "raw_response": final_response,
                "image_size": {
                    "original": list(original_size),
                    "processed": list(processed_size)
                }
            }
            
            predictions.append(prediction)
            print(f"  >>> Extracted {len(answer)} entities")
            
        except Exception as e:
            print(f"  >>> Error processing {img_id}: {e}")
            import traceback
            traceback.print_exc()
            
            predictions.append({
                "img_id": img_id,
                "tokens": sample.get("tokens", []),
                "prediction": [],
                "error": str(e)
            })
    
    # Save predictions
    print(f"\n{'='*60}")
    print(f">>> Saving predictions to: {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    
    print(f">>> Completed! Processed {len(predictions)} samples")
    
    # Print statistics
    success_count = sum(1 for p in predictions if "error" not in p and len(p["prediction"]) > 0)
    error_count = sum(1 for p in predictions if "error" in p)
    print(f"\n📊 Statistics:")
    print(f"  Total: {len(predictions)}")
    print(f"  ✅ Success: {success_count}")
    print(f"  ⚠️  Error: {error_count}")
    
    # Run evaluation if requested
    if args.evaluate and args.gt_path:
        print(f"\n{'='*60}")
        print(f">>> Running evaluation...")
        print(f">>> GT file: {args.gt_path}")
        
        try:
            # Convert predictions to evaluation format
            # The predictions are already in the correct format (with 'prediction' field)
            # load_cot_prediction will convert 'prediction' -> 'pre_entities' and 
            # 'entity'/'type'/'box2d' -> 'phrase'/'entity_type'/'region_box'
            
            # Save predictions to a temporary file for load_cot_prediction
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                json.dump(predictions, tmp_file, indent=2, ensure_ascii=False)
                tmp_pred_path = tmp_file.name
            
            # Load and convert predictions
            prediction_data = load_cot_prediction(tmp_pred_path)
            
            if args.task_type=='gmner':
                label_dict = {"LOC": 1, "PER": 2, "ORG": 3, "MISC": 4, "None-type": 5}
            elif args.task_type=='fmnerg':
                label_dict = fine_to_label_dict()
            # Create evaluator for gmner
            evaluator = EntityEvaluator(
                label_dict=label_dict,
                gt_path=args.gt_path
            )
            
            # Run evaluation
            evaluator.evaluate(prediction_data)
            
            # Clean up temporary file
            os.unlink(tmp_pred_path)
            
        except Exception as e:
            print(f"  >>> Evaluation error: {e}")
            import traceback
            traceback.print_exc()
    elif args.evaluate:
        print(f"\n⚠️  Evaluation requested but --gt_path not provided. Skipping evaluation.")


if __name__ == "__main__":
    main()

