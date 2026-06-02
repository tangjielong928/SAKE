import json
import os
import re
from typing import List
from collections import Counter

CHECK_FILE_PATH = "xxxcold_start/get_is_search_tag/result_fmnerg/Qwen2.5-VL-7B-Instruct/qwen_7B_fmnerg_train.jsonl"

# TASK = "fmnerg"
# MODEL = "7B"
# SPLIT = "train"

# GROUND_TRUE_PATH = f"xxxbaseline_experiment/data/twitter_{TASK}_{SPLIT}.json"
# OUTPUT_PATH = f"xxxcold_start/get_is_search_tag/data/4times_result/twitter_{TASK}_{SPLIT}_search_tag_{MODEL}.json"
# CHECK_FOLDER_PATH = f"xxxcold_start/get_is_search_tag/result/4times/Qwen2.5-VL-{MODEL}-Instruct/{TASK}/{SPLIT}"

def check_box(boxA: List, boxB: List, fmt='xyxy', threshold=0.5) -> bool:
    """
    计算两个边界框的 IoU（Intersection over Union）。

    参数
    - boxA, boxB: 长度为 4 的可迭代对象（list/tuple/np.array）
      如果 fmt == 'xyxy'，则格式为 [x1, y1, x2, y2]（左上和右下）
      如果 fmt == 'xywh'，则格式为 [x, y, w, h]（左上 + 宽高）
    - fmt: 'xyxy' 或 'xywh'，默认 'xyxy'

    """
    if len(boxA) == len(boxB) == 0:
        return True
    elif len(boxA) != len(boxB):
        return False

    # 转成 float
    try:
        xa = [float(x) for x in boxA]
        xb = [float(x) for x in boxB]
    except:
        return False

    # 将 fmt='xywh' 转为 xyxy
    if fmt == 'xywh':
        ax1, ay1, aw, ah = xa
        bx1, by1, bw, bh = xb
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh
    else:  # assume 'xyxy'
        ax1, ay1, ax2, ay2 = xa
        bx1, by1, bx2, by2 = xb

    # 规范化（确保 x1<=x2, y1<=y2）
    ax1, ax2 = min(ax1, ax2), max(ax1, ax2)
    ay1, ay2 = min(ay1, ay2), max(ay1, ay2)
    bx1, bx2 = min(bx1, bx2), max(bx1, bx2)
    by1, by2 = min(by1, by2), max(by1, by2)

    # 交集坐标
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    # 交集宽高（非负）
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter_area = iw * ih

    # 各自面积
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter_area
    if union <= 0:
        res =  0.0
    else:
        res = inter_area / union
    return True if res >= threshold else False


def main2(num_file: int):
    file_lis = os.listdir(CHECK_FOLDER_PATH)
    responses = []
    for file in file_lis[:num_file]:
        with open(os.path.join(CHECK_FOLDER_PATH, file), "r") as fr:
            responses.append([json.loads(line) for line in fr.readlines()])
    
    prediction = []
    for line in zip(*responses):
        anchor = line[0]
        anchor["response"] = [anchor["response"]]
        input_text = anchor["messages"][0]["content"]
        for element in line[1:]:
            assert element["messages"][0]["content"]== input_text, "the order between check files are not aligned"
            anchor["response"].append(element["response"])
        prediction.append(anchor)

    with open(GROUND_TRUE_PATH, "r") as fr:
        ground_true = json.load(fr)
    

    failure_count = 0
    dataset_with_tag = []
    for pre, gt in zip(prediction, ground_true):
        gt_text = "<image>" + " ".join(gt["tokens"])
        assert pre["messages"][0]["content"]== gt_text, "the order of check file and ground true is not aligned"
        
        pre_entities = []
        for res in pre["response"]:
            try:
                res = re.sub(r",\s*(?=[\]\}])", "", res)
                match = re.search(r"```json\s*(.*?)```", res, flags=re.S)
                json_text = match.group(1).strip()
                json_data = json.loads(json_text)
                pre_entities.extend([{'phrase': jd["entity_name"], 'entity_type': jd["entity_type"], 'region_box': jd["entity_region"]} for jd in json_data["pre_entities"]])
            except:
                pre_entities.extend([{'phrase': " ", 'entity_type': " ", 'region_box': []}])
                failure_count += 1
        
        for gt_en in gt["gt_entities"]:
            hit_text = 0
            hit_image = 0
            for en in pre_entities:
                if en["phrase"] == gt_en["phrase"]:
                    if gt_en["entity_type"] == en["entity_type"]:
                        hit_text += 1
                    if check_box(en["region_box"], gt_en["region_box"]):
                        hit_image += 1
            
            if hit_image > 0 and hit_text > 0:
                search_tag = "None"
            elif hit_image == 0 and hit_text > 0:
                search_tag = "only_image"
            elif hit_text == 0 and hit_image > 0:
                search_tag = "only_text"
            else:
                search_tag = "both"

            gt_en.update({"is_search": search_tag})
            gt_en.update({"hit_image": min(hit_image, num_file)})
            gt_en.update({"hit_text": min(hit_text, num_file)})
            dataset_with_tag.append(search_tag)
    
    total = len(dataset_with_tag)
    counts = Counter(dataset_with_tag)

    result = {}
    for key, value in counts.items():
        ratio = value / total * 100   
        result[key] = (value, ratio)
    print(result)
    print(f"{failure_count} prediction data cannot been converted")
    with open(OUTPUT_PATH, "w") as fw:
        json.dump(ground_true, fw, indent=4, ensure_ascii=False)


def main():
    with open(CHECK_FILE_PATH, "r") as fr:
        prediction = [json.loads(line) for line in fr.readlines()]
    with open(GROUND_TRUE_PATH, "r") as fr:
        ground_true = json.load(fr)
    

    is_search_count = []
    dataset_with_tag = []
    for pre, gt in zip(prediction, ground_true):
        gt_text = "<image>" + " ".join(gt["tokens"])
        assert pre["messages"][0]["content"]== gt_text, "the order of check file and ground true is not aligned"
        pre_entities = pre["messages"][-1]["content"]
        pre_entities = re.sub(r",\s*(?=[\]\}])", "", pre_entities)
        match = re.search(r"```json\s*(.*?)```", pre_entities, flags=re.S)
        try:
            json_text = match.group(1).strip()
            json_data = json.loads(json_text)
            pre_entities = [{'phrase': jd["entity_name"], 'entity_type': jd["entity_type"], 'region_box': jd["entity_region"]} for jd in json_data["pre_entities"]]
        except:
            pre_entities = [{'phrase': " ", 'entity_type': " ", 'region_box': []}]
            print("a prediction data cannot been converted")
        
        for gt_en in gt["gt_entities"]:
            is_search_list = []
            for en in pre_entities:
                is_search = "both"
                is_box_match = check_box(en["region_box"], gt_en["region_box"])
                is_text_match = gt_en["phrase"] == en["phrase"] and gt_en["entity_type"] == en["entity_type"]
                if is_box_match and is_text_match:
                    is_search = "None"
                elif is_text_match:
                    is_search = "only_image"
                elif is_box_match:
                    is_search = "only_text"
                is_search_list.append(is_search)
            
            if "None" in is_search_list:
                search_tag = "None"
            elif "only_image" in is_search_list:
                search_tag = "only_image"
            elif "only_text" in is_search_list:
                search_tag = "only_text"
            else:
                search_tag = "both"

            gt_en.update({"is_search": search_tag})
            dataset_with_tag.append(search_tag)
    
    total = len(dataset_with_tag)
    counts = Counter(dataset_with_tag)

    result = {}
    for key, value in counts.items():
        ratio = value / total * 100   
        result[key] = (value, ratio)
    print(result)

    with open(OUTPUT_PATH, "w") as fw:
        json.dump(ground_true, fw, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    # main()
    TASK_LIST = ["gmner", "fmnerg"]
    MODEL_LIST = ["3B", "7B"]
    SPLIT_LIST = ["train", "dev", "test"]
    PASSK = 6

    for TASK in TASK_LIST:
        for SPLIT in SPLIT_LIST:
            for MODEL in MODEL_LIST:
                GROUND_TRUE_PATH = f"xxxbaseline_experiment/data/twitter_{TASK}_{SPLIT}.json"
                OUTPUT_PATH = f"xxxcold_start/get_is_search_tag/data/{PASSK}times_result/twitter_{TASK}_{SPLIT}_search_tag_{MODEL}.json"
                CHECK_FOLDER_PATH = f"xxxcold_start/get_is_search_tag/result/4times/Qwen2.5-VL-{MODEL}-Instruct/{TASK}/{SPLIT}"
                main2(PASSK)