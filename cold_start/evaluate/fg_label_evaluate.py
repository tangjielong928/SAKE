from evaluate import EntityEvaluator
from utils import read_json, transform_swift_output, split_unseen_tag_dataset
coarse_fine_tree = {
    'location': ['city',
                 'country',
                 'state',
                 'continent',
                 'location_other',
                 'park',
                 'road'],
    'building': ['building_other',
                 'cultural_place',
                 'entertainment_place',
                 'sports_facility'],
    'organization': ['company',
                     'educational_institution',
                     'band',
                     'government_agency',
                     'news_agency',
                     'organization_other',
                     'political_party',
                     'social_organization',
                     'sports_league',
                     'sports_team'],
    'person': ['politician',
               'musician',
               'actor',
               'artist',
               'athlete',
               'author',
               'businessman',
               'character',
               'coach',
               'director',
               'intellectual',
               'journalist',
               'person_other'],
    'other': ['animal',
              'award',
              'medical_thing',
              'website',
              'ordinance'],
    'art': ['art_other',
            'film_and_television_works',
            'magazine',
            'music',
            'written_work'],
    'event': ['event_other',
              'festival',
              'sports_event'],
    'product': ['brand_name_products',
                'game',
                'product_other',
                'software']}

def fine_to_label_dict(coarse_fine_tree = coarse_fine_tree):
    label_dict = {}
    index = 1
    for coarse in coarse_fine_tree:
        for fine_label in coarse_fine_tree[coarse]:
            label_dict[fine_label] = index
            index+=1
    label_dict["None-type"] = index
    return label_dict

def coarse_to_label_dict(coarse_fine_tree = coarse_fine_tree):
    label_dict = {}
    index = 1
    for keys in coarse_fine_tree:
        label_dict[keys] = index
        index+=1
    label_dict["None-type"] = index
    return label_dict

def print_fg_type_prompt(coarse_fine_tree = coarse_fine_tree):
    str = r""
    for coarse in coarse_fine_tree:
        for fine_label in coarse_fine_tree[coarse]:
            str += "- " + fine_label + "\n"
    print(repr(str))

def main1(gt_path, result_path):
    evaluator = EntityEvaluator(label_dict= fine_to_label_dict(), gt_path=gt_path)
    # local model输出结果
    prediction_data = transform_swift_output(result_path)
    # 评估原始结果
    res = evaluator.evaluate(prediction_data)
    # print_fg_type_prompt()

def main2(gt_path, result_path):
    split_dataset = split_unseen_tag_dataset(gt_path, result_path)
    # import pdb;pdb.set_trace()
    result = []
    for pairs in split_dataset:
        evaluator = EntityEvaluator(label_dict= fine_to_label_dict(), gt_path=gt_path)
        evaluator.gt_entities = pairs["gt"]
        res = evaluator.evaluate(pairs["pred"])
        result.append(list(map(lambda x: round(x, 2), [i[2] for i in res])))
    print(result)



if __name__ == '__main__':
    gt_path = "xxx/baseline_experiment/training_data/FMNERG/swift_input_test_fmnerg_GT_scale.jsonl"
    result_path = "xxx/baseline_experiment/output_fmnerg/v1-20251211-105709/checkpoint-6570/infer_result/Qwen-7B.jsonl"
    # main1(gt_path, result_path)
    main2(gt_path, result_path)