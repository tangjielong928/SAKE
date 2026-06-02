import json
import base64
import torch
import os
from torchvision.ops import box_iou
import re
import pdb

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
# 字符串转 JSON
def string_to_json(json_string):
    try:
        json_obj = json.loads(json_string)
        return json_obj
    except json.JSONDecodeError:
        print("错误: 输入的字符串不是有效的 JSON 格式!")
        return None

# JSON 转字符串
def json_to_string(json_obj):
    try:
        json_string = json.dumps(json_obj)
        return json_string
    except TypeError:
        print("错误: 输入的对象无法转换为 JSON 字符串!")
        return None

def str_to_list(input_str):
    if isinstance(input_str, list):
        return input_str
    try:
        input_str = input_str.strip('[]')
        elements = input_str.split(',')
        if len(elements) == 4:
            return [int(i) for i in elements]
        else:
            return []
    except (AttributeError, ValueError, IndexError):
        return []

def find_sublist(main_list, sub_list):
    """
    在主列表中查找连续的子列表，返回子列表的开始和结束索引。
    如果子列表不存在于主列表中，则返回None。
    注意：假设连续子列表在主列表中最多只出现一次。
    Args:
        main_list (list): 主列表
        sub_list (list): 要查找的子列表
    Returns:
        tuple or None: 如果找到子列表，返回(start_index, end_index)；否则返回None
    """
    if not sub_list: 
        return None
    if not main_list: 
        return None
        
    sub_len = len(sub_list)
    main_len = len(main_list)
    
    if sub_len > main_len:
        return None
    
    for i in range(main_len - sub_len + 1):
        match = True
        
        for j in range(sub_len):
            if main_list[i + j] != sub_list[j]:
                match = False
                break
        if match:
            return (i, i + sub_len - 1)
    return None

def calculate_iou(box1, box2):
    """
    使用torchvision.ops.box_iou计算两个边界框的IoU
    :param box1: 第一个边界框，格式为[x1, y1, x2, y2]
    :param box2: 第二个边界框，格式为[x1, y1, x2, y2]
    :return: IoU值
    """
    if len(box1) == len(box2) == 0:
        return 100.0
    elif len(box1) == 0 or len(box2) == 0:
        return 0.0
    elif len(box1) != 4 or len(box2) != 4:
        return 0.0
    # 转换为PyTorch张量
    box1_tensor = torch.tensor([box1], dtype=torch.float)
    box2_tensor = torch.tensor([box2], dtype=torch.float)
    iou_matrix = box_iou(box1_tensor, box2_tensor)
    return iou_matrix[0, 0].item()

def collect(union, sample_pred, sample_gt):
    fn = [x for x in union if x not in sample_gt]
    fp = []
    for i in range(len(sample_pred)):
        if sample_pred[i] in fn:
            for tp in sample_gt:
                if sample_pred[i][:3] == tp[:3] and sample_pred[i][3] != tp[3] and (sample_pred[i][0], sample_pred[i][1], sample_pred[i][2], 0):
                    fp.append((sample_pred[i][0], sample_pred[i][1], sample_pred[i][2], 0))
    union.update(fp)
    sample_pred.extend(fp)
    return sample_pred, union

def read_json_to_list(file_path):
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                obj = json.loads(line)
                # 检查解析结果是否为字典
                if isinstance(obj, dict):
                    # 检查是否存在 pre_entities 键且其值为列表
                    if 'pre_entities' in obj and isinstance(obj['pre_entities'], list):
                        data_list.append(obj)
            except json.JSONDecodeError:
                # 若解析失败，跳过当前行
                continue
    return data_list

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file.readlines()]
    return data

def transform_swift_output(output_jsonl_path):
    dataset = read_jsonl(output_jsonl_path)
    transformed_dataset = []

    for d in dataset:
        try:
            item = {"img_id": d["images"][0]["path"]}
            entities = d["response"]
            entities = re.sub(r",\s*(?=[\]\}])", "", entities)
            match = re.search(r"```json\s*(.*?)```", entities, flags=re.S)
            json_text = match.group(1).strip()
            json_data = json.loads(json_text)
            pre_entities = [{'phrase': jd["entity_name"], 'entity_type': jd["entity_type"], 'region_box': jd["entity_region"]} for jd in json_data["pre_entities"]]
            item.update({"pre_entities": pre_entities})
            transformed_dataset.append(item)
        except:
            print("a data comes error, using empty line instead")
            item.update({"pre_entities": []})
            transformed_dataset.append(item)
            continue
    return transformed_dataset


def transform_swift_GT(GT_jsonl_path):
    dataset = read_jsonl(GT_jsonl_path)
    transformed_dataset = []
    for d in dataset:
        item = {"img_id": d["image"][0]}
        entities = d["messages"][-1]["content"]
        entities = re.sub(r",\s*(?=[\]\}])", "", entities)
        match = re.search(r"```json\s*(.*?)```", entities, flags=re.S)
        json_text = match.group(1).strip()
        json_data = json.loads(json_text)
        pre_entities = [{'phrase': jd["entity_name"], 'entity_type': jd["entity_type"], 'region_box': jd["entity_region"]} for jd in json_data["pre_entities"]]
        item.update({"gt_entities": pre_entities})
        if "unseen_tag" in d:
            item.update({"unseen_tag": d["unseen_tag"]})
        transformed_dataset.append(item)
    return transformed_dataset


def split_unseen_tag_dataset(gt_path, output_path):
    GT = transform_swift_GT(gt_path)
    output = transform_swift_output(output_path)
    result = [{"pred": [], "gt": []} for _ in range(4)]
    for gt, out in zip(GT, output):
        for idx, tag in enumerate(gt["unseen_tag"]):
            if tag == "0,0":
                result[0]["pred"].append(out)
                result[0]["gt"].append({"gt_entities": [gt["gt_entities"][idx]], "img_id": gt["img_id"]})
            elif tag == "0,1":
                result[1]["pred"].append(out)
                result[1]["gt"].append({"gt_entities": [gt["gt_entities"][idx]], "img_id": gt["img_id"]})
            elif tag == "1,0":
                result[2]["pred"].append(out)
                result[2]["gt"].append({"gt_entities": [gt["gt_entities"][idx]], "img_id": gt["img_id"]})            
            elif tag == "1,1":
                result[3]["pred"].append(out)
                result[3]["gt"].append({"gt_entities": [gt["gt_entities"][idx]], "img_id": gt["img_id"]})
    return result

def load_cot_prediction(path):
    with open(path, "r") as fr:
        prediction = json.load(fr)
    for p in prediction:
        p["pre_entities"] = p.pop("prediction")
        for idx, e in enumerate(p["pre_entities"]):
            try:
                e["phrase"] = e.pop("entity")
                e["entity_type"] = e.pop("type")
                e["region_box"] = e.pop("box2d")
            except:
                p["pre_entities"][idx] = {"phrase": "", "entity_type": "", "region_box": []}
    return prediction
        
