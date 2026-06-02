import os
import json
from config import Config
from generator import TrajectoryGenerator

def convert_gt_entities_to_ground_truth(gt_entities, default_is_search="both"):
    """
    将 twitter_gmner_gt.json 格式的 gt_entities 转换为 ground_truth 格式。
    
    Args:
        gt_entities: gt_entities 列表，每个元素包含 phrase, entity_type, region_box
        default_is_search: 默认的 is_search 值，默认为 "both"
    
    Returns:
        ground_truth 列表，每个元素包含 entity, type, box2d (或 region_box), is_search
    """
    ground_truth = []
    for ent in gt_entities:
        item = {
            "entity": ent.get("phrase", ""),
            "type": ent.get("entity_type", ""),
            "box2d": ent.get("region_box", []),  # 使用 region_box 作为 box2d
            "is_search": ent.get("is_search", default_is_search)  # 默认为 both, text_only, image_pnly, None
        }
        ground_truth.append(item)
    return ground_truth

def join_tokens(tokens):
    """将 tokens 列表拼接为文本"""
    if not tokens:
        return ""
    text = " ".join(tokens)
    text = " ".join(text.split())  # 清理多余空格
    return text

def main():
    Config.validate()
    
    # 示例：从 twitter_gmner_gt.json 格式读取数据
    # 这里展示如何转换数据格式
    sample_data = {
        "tokens": [
            "Join",
            "the",
            "Centennial",
            "Celebration",
            "of",
            "#",
            "ThePeoplesPalace",
            "San",
            "Francisco",
            "City",
            "Hall",
            "!",
            "http://t.",
            "co/1DzVeOJVyM",
            "#",
            "SFCityHall100"
        ],
        "img_id": "475049.jpg",
        "gt_entities": [
            {
                "start": 7,
                "end": 10,
                "entity_type": "LOC",
                "phrase": "San Francisco City Hall",
                "region_box": [1, 339, 597, 720], #bounding
                "is_search": "both" #both, only_text, only_image, None
            }
        ]
    }
    
    # 转换为所需格式
    user_text = join_tokens(sample_data["tokens"])
    img_id = sample_data["img_id"]
    ground_truth = convert_gt_entities_to_ground_truth(
        sample_data["gt_entities"], 
        default_is_search="both"
    )
    
    print(f">>> Processing image: {img_id}")
    print(f">>> Text: {user_text}")
    print(f">>> Ground truth entities: {len(ground_truth)}")
    
    gen = TrajectoryGenerator()
    result = gen.run(
        img_id=img_id,
        user_text=user_text,
        ground_truth=ground_truth,
        image_root="xxx/data/Twitter-GMNER/whole_image"  # 默认图像根目录
    )

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(Config.OUTPUT_DIR, f"{img_id.replace('.jpg', '_cot.json')}")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"Done! Saved to {out_path}")

if __name__ == "__main__":
    main()