import re
import json
import os
import base64
import math
from io import BytesIO
from PIL import Image
from typing import Tuple, Optional, List

def extract_tag(text, tag):
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None

def parse_json(text):
    try:
        # 清理 Markdown 代码块
        text = re.sub(r"```json|```", "", text).strip()
        return json.loads(text)
    except:
        return []

def load_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def resolve_image_path(root_dir: str, img_id: str) -> str:
    """解析图像路径，兼容有无扩展名的情况"""
    base = os.path.join(root_dir, str(img_id))
    if os.path.isfile(base):
        return base
    # 尝试常见扩展名
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        cand = base + ext
        if os.path.isfile(cand):
            return cand
    return base

def load_image_as_base64(image_path: str) -> Optional[str]:
    """从文件路径加载图像并转换为 base64 字符串"""
    if not image_path or not os.path.isfile(image_path):
        return None
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def process_image(image_path: str, max_pixels: int = 672 * 672 * 2, min_pixels: int = 512 * 512) -> Tuple[Image.Image, Tuple[int, int], Tuple[int, int]]:
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

def scale_box2d(box: List[int], original_size: Tuple[int, int], processed_size: Tuple[int, int]) -> List[int]:
    """
    根据图像缩放比例缩放 box2d 坐标。
    
    Args:
        box: [x1, y1, x2, y2] 或 []
        original_size: (width, height) 原始图像尺寸
        processed_size: (width, height) 处理后的图像尺寸
    
    Returns:
        缩放后的 box2d 坐标
    """
    if not box or len(box) != 4:
        return box
    
    orig_w, orig_h = original_size
    proc_w, proc_h = processed_size
    
    scale_x = proc_w / orig_w
    scale_y = proc_h / orig_h
    
    x1, y1, x2, y2 = box
    scaled_box = [
        int(x1 * scale_x),
        int(y1 * scale_y),
        int(x2 * scale_x),
        int(y2 * scale_y)
    ]
    
    return scaled_box