import os
import json
import re


def change_path(path_head, sample):
    '''
        correct the image path and move it
    '''
    new_path = []
    try:
        for item in sample["messages"]:
            if "images" in item.keys():
                for p in item["images"]:
                    split_idx = p.rfind("/")
                    new_path.append(path_head + p[split_idx:])
                del item["images"]
    except:
        import pdb
        pdb.set_trace()
    sample.update({"images": new_path})
    return sample



def test():
    sam1_path = "./cot_train_3k/16_05_01_6_cot.json"
    sam2_path = "./cot_train_3k/16_05_01_29_cot.json"
    path_head = "./cot_data/cot_train_3k_image"

    with open(sam1_path, "r") as fr:
        sam1 = json.load(fr)
    with open(sam2_path, "r") as fr:
        sam2 = json.load(fr)
    
    data = []
    data.append(change_path(path_head, sam1))
    data.append(change_path(path_head, sam2))

    with open("../train/test_sample.jsonl", "w") as fw:
        for d in data:
            fw.write(json.dumps(d))
            fw.write("\n")


def main():
    messages_path = "xxx/cold_start/cot_trajectories_test_1500"
    output_path = "./cot_test_3k.jsonl"
    path_head = "./cot_data/cot_train_3k_image"
    data = []
    path_lis = os.listdir(messages_path)
    for path in path_lis:
        if "progress" in path or os.path.isdir(os.path.join(messages_path, path)):
            continue
        with open(os.path.join(messages_path, path), "r") as fr:
            d = json.load(fr)
            # data.append(change_path(path_head, d))
            data.append(d)
    
    with open(output_path, "w") as fw:
        for d in data:
            fw.write(json.dumps(d))
            fw.write("\n")



if __name__ == "__main__":
    main()