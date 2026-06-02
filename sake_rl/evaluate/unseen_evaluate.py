import json
import os
from utils import transform_swift_output


def check_box(boxA, boxB, fmt='xyxy', threshold=0.5) -> bool:
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
    xa = [float(x) for x in boxA]
    xb = [float(x) for x in boxB]

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

def main(g, r):
    gt_path = g
    result_path = r
    with open(gt_path, "r") as fr:
        ground_true = json.load(fr)
    prediction = transform_swift_output(result_path)

    tag_dict = {"0,0": [], "0,1": [], "1,0": [], "1,1": []}
    for gt in ground_true:
        for en in gt["gt_entities"]:
            tag = en["visual_and_textual_unseen"]
            tag_dict[tag].append((en["phrase"], en["entity_type"], en["region_box"]))

    pred_set = []
    for pre in prediction:
        for en in pre["pre_entities"]:
            pred_set.append((en["phrase"], en["entity_type"], en["region_box"]))

    for k, v in tag_dict.items():
        res = {"gmner": 0, "eeg": 0, "mner": 0}
        for gten in v:
            gmner = False
            eeg = False
            mner = False
            for pred in pred_set:
                if pred[0] != gten[0]:
                    continue
                match_box = check_box(gten[2], pred[2])
                match_type = gten[1] == pred[1]
                if match_box and match_type:
                    gmner = True
                elif match_box:
                    eeg = True
                elif match_type:
                    mner = True
            if gmner:
                res["gmner"] += 1
                res["eeg"] += 1
                res["mner"] += 1
            elif eeg:
                res["eeg"] += 1
            elif mner:
                res["mner"] += 1
        rate = list(map(lambda x: str(round(100*x / len(v), 2)), res.values()))
        print(f"| {k} | " + " | ".join(rate) + " |")



if __name__ == "__main__":
    gt_path = [
        "xxx/baseline_experiment/process_data/unseen_result/twitter_fmnerg_test_unseen.json",
        "xxx/baseline_experiment/process_data/unseen_result/twitter_fmnerg_test_unseen.json",
        "xxx/baseline_experiment/process_data/unseen_result/twitter_gmner_test_unseen.json",
        "xxx/baseline_experiment/process_data/unseen_result/twitter_gmner_test_unseen.json",
    ]
    result_path = [
        "xxx/baseline_experiment/output_fmnerg/v1-20251211-105709/checkpoint-6570/infer_result/20251213-075607.jsonl",
        "xxx/baseline_experiment/output_fmnerg/v1-20251211-105709/checkpoint-6570/infer_result/Qwen-7B.jsonl",
        "xxx/baseline_experiment/output/Qwen2.5-VL-3B-Instruct/v9-20251120-024308/checkpoint-6570/infer_result/20251124-060317.jsonl",
        "xxx/baseline_experiment/output/Qwen2.5-VL-3B-Instruct/v9-20251120-024308/checkpoint-6570/infer_result/Qwen-7B.jsonl"
    ]
    for g, r in zip(gt_path, result_path):
        main(g, r)