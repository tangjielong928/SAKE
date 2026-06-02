import json
import copy
import os
from llm_client import LLMClient
from search_tool import SearchTool
from prompts import (
    SYSTEM_PROMPT_TEACHER, STEP_1_TEACHER_PROMPT, 
    STEP_2_TEACHER_PROMPT, STEP_3_TEACHER_PROMPT
)
from utils import (
    extract_tag, parse_json, load_file, 
    resolve_image_path, load_image_as_base64, 
    process_image, scale_box2d
)
from config import Config

class TrajectoryGenerator:
    def __init__(self):
        self.llm = LLMClient()
        self.search = SearchTool()
        self.user_prompts = {
            "step1": load_file("xxx/prompts/Twitter-GMNER-Prompt/round_1_user_prompt_1.txt"),
            "step2": load_file("xxx/prompts/Twitter-GMNER-Prompt/after_text_search_prompt_1.txt"),
            "step3": load_file("xxx/prompts/Twitter-GMNER-Prompt/after_image_search_prompt_1.txt")
        }

    def _generate_behavior_instructions(self, ground_truth):
        """
        根据 GT 中的 is_search 字段生成自然语言指令给 Teacher。
        """
        instructions = []
        for item in ground_truth:
            entity = item['entity']
            mode = item.get('is_search', None)
            
            if mode == 'only_text':
                instructions.append(f"- Entity '{entity}': [FORCE TEXT SEARCH]. Pretend you don't know what this is.")
            elif mode == 'only_image':
                instructions.append(f"- Entity '{entity}': [FORCE IMAGE SEARCH]. Pretend you know the concept, but don't know what it looks like visually.")
            elif mode == 'both':
                instructions.append(f"- Entity '{entity}': [FORCE TEXT SEARCH] AND [FORCE IMAGE SEARCH]. First pretend you don't know the concept.")
            else:
                instructions.append(f"- Entity '{entity}': [NO SEARCH]. You have full knowledge.")
        
        return "\n".join(instructions)

    def _clean_gt_for_output(self, ground_truth):
        """
        移除 is_search 字段，用于最终输出和 Step 3 的 prompt。
        """
        clean_gt = copy.deepcopy(ground_truth)
        for item in clean_gt:
            if 'is_search' in item:
                del item['is_search']
        return json.dumps(clean_gt, indent=2)
    
    def _load_and_process_image(self, img_id: str, image_root: str):
        """
        加载图像，进行缩放处理，并返回 base64 编码和尺寸信息。
        
        Returns:
            image_base64: base64 编码的图像字符串
            original_size: (width, height) 原始尺寸
            processed_size: (width, height) 处理后的尺寸
        """
        image_path = resolve_image_path(image_root, img_id)
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # 处理图像缩放
        processed_img, original_size, processed_size = process_image(image_path)
        
        # 将处理后的图像转换为 base64
        from io import BytesIO
        import base64
        buffered = BytesIO()
        processed_img.save(buffered, format="JPEG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return image_base64, original_size, processed_size
    
    def _scale_ground_truth_boxes(self, ground_truth, original_size, processed_size):
        """
        缩放 ground_truth 中的 box2d 坐标（region_box）。
        """
        scaled_gt = copy.deepcopy(ground_truth)
        for item in scaled_gt:
            # 处理 region_box（对应 box2d）
            if 'region_box' in item and item['region_box']:
                item['region_box'] = scale_box2d(
                    item['region_box'], 
                    original_size, 
                    processed_size
                )
            # 也处理 box2d（如果存在）
            if 'box2d' in item and item['box2d']:
                item['box2d'] = scale_box2d(
                    item['box2d'], 
                    original_size, 
                    processed_size
                )
        return scaled_gt
    
    def _check_resume(self, img_id: str) -> dict:
        """
        检查是否存在已完成的轨迹文件，如果存在则返回。
        
        Returns:
            如果存在已完成轨迹则返回轨迹字典，否则返回 None
        """
        # 使用与 main.py 相同的路径生成逻辑（替换点号为下划线）
        out_path = os.path.join(Config.OUTPUT_DIR, f"{img_id.replace('.jpg', '_cot.json')}")
        if os.path.exists(out_path):
            try:
                with open(out_path, 'r', encoding='utf-8') as f:
                    trajectory = json.load(f)
                # 检查是否已完成（有 steps 且最后一步有 answer）
                if trajectory.get('steps') and len(trajectory['steps']) > 0:
                    last_step = trajectory['steps'][-1]
                    last_response = last_step.get('response', '')
                    if extract_tag(last_response, 'answer'):
                        print(f">>> Found existing trajectory for {img_id}, skipping...")
                        return trajectory
            except Exception as e:
                print(f">>> Error reading existing trajectory: {e}")
        return None

    def run(self, img_id: str, user_text: str, ground_truth: list, image_root: str = None):
        """
        生成轨迹数据。
        
        Args:
            img_id: 图像 ID
            user_text: 用户文本
            ground_truth: ground truth 列表，每个元素包含 entity, type, region_box (或 box2d), is_search
            image_root: 图像根目录，默认为 "./Twitter-GMNER/whole_image"
        
        Returns:
            轨迹字典
        """
        # 检查断点重启
        existing_trajectory = self._check_resume(img_id)
        if existing_trajectory:
            return existing_trajectory
        
        # 设置默认图像根目录
        if image_root is None:
            image_root = "./Twitter-GMNER/whole_image"
        
        # 加载和处理图像
        print(f">>> Loading and processing image: {img_id}")
        image_base64, original_size, processed_size = self._load_and_process_image(img_id, image_root)
        
        # 缩放 ground_truth 中的 box2d/region_box
        scaled_ground_truth = self._scale_ground_truth_boxes(ground_truth, original_size, processed_size)
        
        # 准备输出用的 ground_truth（移除 is_search，转换格式）
        output_gt = []
        for item in scaled_ground_truth:
            gt_item = {
                "entity": item.get("entity") or item.get("phrase", ""),
                "type": item.get("type") or item.get("entity_type", ""),
                "box2d": item.get("box2d") or item.get("region_box", [])
            }
            output_gt.append(gt_item)
        
        trajectory = {
            "input": {
                "text": user_text, 
                "image": img_id
            }, 
            "gt": output_gt, 
            "steps": [],
            "image_size_info": {
                "original_size": original_size,
                "processed_size": processed_size
            }
        }
        
        # 1. 准备指令
        behavior_instr = self._generate_behavior_instructions(scaled_ground_truth)
        clean_gt_str = json.dumps(output_gt, indent=2)

        # --- STEP 1 ---
        print(">>> Generating Step 1 (Analysis)...")
        prompt_1 = STEP_1_TEACHER_PROMPT.format(
            user_text=user_text, 
            behavior_instructions=behavior_instr, # 注入控制指令
            clean_ground_truth_json=clean_gt_str, # 注入干净的 GT
            user_prompt_content=self.user_prompts["step1"]
        )
        resp_1 = self.llm.call_model(SYSTEM_PROMPT_TEACHER, prompt_1, image_base64)
        trajectory["steps"].append({"step": 1, "response": resp_1})

        # 检查是否直接结束
        if extract_tag(resp_1, "answer"):
            print("Finished at Step 1.")
            # 保存轨迹
            os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
            safe_img_id = img_id.replace('.', '_')
            out_path = os.path.join(Config.OUTPUT_DIR, f"{safe_img_id}_cot.json")
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(trajectory, f, indent=2, ensure_ascii=False)
            return trajectory
        
        # 执行 Text Search (如果有)
        text_queries = parse_json(extract_tag(resp_1, "text_search") or "[]")
        search_res = self.search.text_search(text_queries) if text_queries else "No text search performed."

        # --- STEP 2 ---
        print(">>> Generating Step 2 (Post-Text Search)...")
        reasoning_1 = extract_tag(resp_1, "reason")
        prompt_2 = STEP_2_TEACHER_PROMPT.format(
            step_1_reasoning=reasoning_1,
            search_results=search_res,
            behavior_instructions=behavior_instr, # 再次注入指令，确保 Image Search 被触发
            clean_ground_truth_json=clean_gt_str, # 注入干净的 GT
            user_prompt_content=self.user_prompts["step2"]
        )
        resp_2 = self.llm.call_model(SYSTEM_PROMPT_TEACHER, prompt_2, image_base64)
        trajectory["steps"].append({"step": 2, "context": search_res, "response": resp_2})

        # 检查是否直接结束
        if extract_tag(resp_2, "answer"):
            print("Finished at Step 2.")
            return trajectory

        # 执行 Image Search (如果有)
        img_queries = parse_json(extract_tag(resp_2, "image_search") or "[]")
        img_res = self.search.image_search(img_queries) if img_queries else "No image search performed."

        # --- STEP 3 ---
        print(">>> Generating Step 3 (Post-Image Search)...")
        reasoning_2 = extract_tag(resp_2, "reason")
        prompt_3 = STEP_3_TEACHER_PROMPT.format(
            step_2_reasoning=reasoning_2,
            image_results=img_res,
            clean_ground_truth_json=clean_gt_str, # 这里只给干净的 GT
            user_prompt_content=self.user_prompts["step3"]
        )
        resp_3 = self.llm.call_model(SYSTEM_PROMPT_TEACHER, prompt_3, image_base64)
        trajectory["steps"].append({"step": 3, "context": img_res, "response": resp_3})

        return trajectory