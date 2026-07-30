import argparse
import os
import json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert raw SeCoT SFT samples into the jsonl format required by swift sft"
    )
    parser.add_argument('--messages_path', type=str, required=True,
                        help='Directory of raw SFT data (one json sample per file)')
    parser.add_argument('--output_path', type=str, default='./sake_secot.jsonl',
                        help='Output jsonl file used as the swift sft dataset')
    parser.add_argument('--image_root', type=str, default=None,
                        help='If set, rewrite every image path to this directory and '
                             'collect them into a top-level "images" field')
    return parser.parse_args()


def change_path(path_head, sample):
    '''
        correct the image path and move it
    '''
    new_path = []
    for item in sample["messages"]:
        if "images" in item.keys():
            for p in item["images"]:
                split_idx = p.rfind("/")
                new_path.append(path_head + p[split_idx:])
            del item["images"]
    sample.update({"images": new_path})
    return sample


def main():
    args = parse_args()

    data = []
    path_lis = sorted(os.listdir(args.messages_path))
    for path in path_lis:
        if "progress" in path or os.path.isdir(os.path.join(args.messages_path, path)):
            continue
        with open(os.path.join(args.messages_path, path), "r") as fr:
            d = json.load(fr)
            if args.image_root is not None:
                d = change_path(args.image_root, d)
            data.append(d)

    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    os.makedirs(output_dir, exist_ok=True)

    with open(args.output_path, "w") as fw:
        for d in data:
            fw.write(json.dumps(d, ensure_ascii=False))
            fw.write("\n")

    print(f"Converted {len(data)} samples -> {args.output_path}")


if __name__ == "__main__":
    main()
