from typing import List, Tuple, Dict
from sklearn.metrics import precision_recall_fscore_support as prfs
from utils import collect, read_jsonl, save_json, calculate_iou, str_to_list, find_sublist, transform_swift_GT, transform_swift_output
from utils import split_unseen_tag_dataset, load_cot_prediction, read_json
# from fg_label import coarse_to_label_dict, fine_to_label_dict
import re
import sys



class EntityEvaluator:
    def __init__(self, label_dict, gt_path):
        '''
        @ param label_dict: 标签字典
        @ param gt_path: 标注文件路径
        '''
        self.label_dict = label_dict
        self.gt_path = gt_path
        self.gt_entities = read_json(gt_path)

    def evaluate(self, pred_json):
        '''
        @ param pred_json: 待评估的预测结果，格式为：
        [{'img_id': '1', 'pre_entities': [{'phrase': 'Curry', 'entity_type': 'PER', 'region_box': [1, 2, 3, 4]}]}
        其中 region_box 为 [x1, y1, x2, y2]为边界框的坐标. 
        1. 提供region_box; 或者2.提供mapping_regions与region (index), 0表示没有region.
        @ return: 评估指标
        '''
        assert len(pred_json) == len(self.gt_entities)
        pred_entities, gt_entities = [], []
        for pred_item, gt_item in zip(pred_json, self.gt_entities):
            assert pred_item['img_id'] == gt_item['img_id']
            pred_items, gt_items = [], []
            for pred_entity in pred_item['pre_entities']:
                if pred_entity["entity_type"] not in self.label_dict:
                    continue
                if 'region_box' in pred_entity:
                    pred_items.append((pred_entity['phrase'], pred_entity['phrase'], pred_entity['entity_type'], pred_entity['region_box']))
                elif 'region' and 'mapping_regions' in pred_entity: 
                    pred_items.append((pred_entity['phrase'], pred_entity['phrase'], pred_entity['entity_type'], [] if pred_entity['region'] == 0 else pred_entity['mapping_regions'][pred_entity['region']-1]))
                else:
                    pred_items.append((pred_entity['phrase'], pred_entity['phrase'], pred_entity['entity_type'], []))
            for gt_entity in gt_item['gt_entities']:
                gt_items.append((gt_entity['phrase'], gt_entity['phrase'], gt_entity['entity_type'], gt_entity['region_box']))
            pred_entities.append(pred_items)
            gt_entities.append(gt_items)

        return self.compute_scores(gt_entities, pred_entities)

        
    def _convert_by_setting(self, gt: List[List[Tuple]], pred: List[List[Tuple]],
                          include_entity_types: bool = True, include_score: bool = False, include_region: bool = False, only_eeg: bool = False):
        assert len(gt) == len(pred)

        # pred: [(start, end, entity_type, match_region, cls_score)]
        # either include or remove entity types based on setting
        def convert(t):
            if only_eeg and len(t) >3:
                c = [t[0], t[1], 'None-type', t[3]]
                return tuple(c)
            if not include_entity_types:
                # remove entity type and score for evaluation
                c = [t[0], t[1], 'None-type']
            else:
                c = list(t[:3])

            if include_region and len(t) > 3:
                # include prediction scores
                c.append(t[3])

            return tuple(c)
        
        def convert_for_bbox(tupe1, tupe2, IoU_threshold = 0.5):
            if not include_region and not only_eeg:
                return tupe1, tupe2
            if len(tupe1) == 0 or len(tupe2) == 0:
                return [], []
            if not (len(tupe1[-1]) > 3 and len(tupe2[-1]) > 3):
                return tupe1, tupe2
            if not (isinstance(tupe1[-1][3], list) and isinstance(tupe2[-1][3], list)):
                return tupe1, tupe2

            tupe1_list = [list(t) for t in tupe1]
            tupe2_list = [list(t) for t in tupe2]
            # 创建边界框到索引的映射
            box_to_index = [([],0)]
            for i, t1 in enumerate(tupe1_list):
                if len(t1) > 3 and isinstance(t1[3], list) and len(t1[3]) > 0:
                    box_to_index.append((t1[3], i+1))
                    t1[3] = i+1  
                elif len(t1) > 3 and isinstance(t1[3], list) and len(t1[3]) == 0:
                    t1[3] = 0
            # 为tupe2中的每个元素分配匹配的索引
            for i, t2 in enumerate(tupe2_list):
                if len(t2) <= 3 or not isinstance(t2[3], list):
                    continue
                match_found = False
                for box, index in box_to_index:
                    if isinstance(t2[3], list) and len(t2[3]) == 0:
                        t2[3] = 0
                    # IoU大于阈值为其找到gt索引    
                    if isinstance(t2[3], list) and len(t2[3]) > 0 and calculate_iou(t2[3], box) >= IoU_threshold \
                    and tupe1_list[index-1][0] == t2[0] and tupe1_list[index-1][1] == t2[1]: 
                        t2[3] = index  
                        match_found = True
                        break
                if not match_found and isinstance(t2[3], list):
                    t2[3] = -100  # 未找到匹配时的默认值
            new_tupe1 = [tuple(t) for t in tupe1_list]
            new_tupe2 = [tuple(t) for t in tupe2_list]
            return new_tupe1, new_tupe2
                
        converted_gt, converted_pred = [], []

        for sample_gt, sample_pred in zip(gt, pred):
            gt, pred = convert_for_bbox([convert(t) for t in sample_gt], [convert(t) for t in sample_pred])
            converted_gt.append(gt)
            converted_pred.append(pred)

        return converted_gt, converted_pred


    def _score(self, gt: List[List[Tuple]], pred: List[List[Tuple]], print_results: bool = False, cls_metric = False, only_text = True):
        assert len(gt) == len(pred)
        # import pdb;pdb.set_trace()
        gt_flat = []
        pred_flat = []
        types = set()

        for (sample_gt, sample_pred) in zip(gt, pred):
            union = set()
            if cls_metric:
                union.update(sample_gt)
                loc_gt = list(map(lambda x:(x[0],x[1]), sample_gt))
                sample_loc_true_pred =  list(filter(lambda x:(x[0], x[1]) in  loc_gt, sample_pred))
                union.update(sample_loc_true_pred)
            else:
                union.update(sample_gt)
                union.update(sample_pred)
            if not only_text:
                sample_pred, union = collect(union, sample_pred, sample_gt)
            for s in union:
                if s in sample_gt:
                    t = s[2]
                    gt_flat.append(self.label_dict[t])
                    types.add(t)
                else:
                    gt_flat.append(0)

                if s in sample_pred:
                    t = s[2]
                    pred_flat.append(self.label_dict[t])
                    types.add(t)
                else:
                    pred_flat.append(0)
        metrics = self._compute_metrics(gt_flat, pred_flat, types, print_results, only_text = only_text)
        return metrics

    def _compute_metrics(self, gt_all, pred_all, types, print_results: bool = False, only_text = True):
        labels = [self.label_dict[t] for t in types]
        per_type = prfs(gt_all, pred_all, labels=labels, average=None)
        micro = prfs(gt_all, pred_all, labels=labels, average='micro')[:-1]
        macro = prfs(gt_all, pred_all, labels=labels, average='macro')[:-1]
        total_support = sum(per_type[-1])

        if print_results:
            self._print_results(per_type, list(micro) + [total_support], list(macro) + [total_support], types, only_text = only_text)

        return [m * 100 for m in micro + macro]


    # def _print_results(self, per_type: List, micro: List, macro: List, types: List, only_text = True):
    #     columns = ('type', 'precision', 'recall', 'f1-score', 'support')

    #     row_fmt = "%20s" + (" %12s" * (len(columns) - 1))
    #     print(row_fmt % columns)

    #     metrics_per_type = []
    #     for i, t in enumerate(types):
    #         metrics = []
    #         for j in range(len(per_type)):
    #             metrics.append(per_type[j][i])
    #         metrics_per_type.append(metrics)

    #     for m, t in zip(metrics_per_type, types):
    #         print(row_fmt % self._get_row(m, t))

    #     print('')
    #     print(row_fmt % self._get_row(micro, 'micro'))
    #     print(row_fmt % self._get_row(macro, 'macro'))


    # def _get_row(self, data, label):
    #     row = [label]
    #     for i in range(len(data) - 1):
    #         row.append("%.2f" % (data[i] * 100))
    #     row.append(data[3])
    #     return tuple(row)


    def _print_results(self, per_type: List, micro: List, macro: List, types: List, only_text=True):
        columns = ('type', 'precision', 'recall', 'f1-score', 'support')

        # 每列宽度（可以按需微调）
        col_widths = [25, 10, 10, 10, 10]

        def format_row(items):
            return "| " + " | ".join(
                f"{item:<{w}}" if i == 0 else f"{item:>{w}}"
                for i, (item, w) in enumerate(zip(items, col_widths))
            ) + " |"

        # 表头
        print(format_row(columns))
        print("| " + " | ".join("-" * w for w in col_widths) + " |")

        # 组织 per-type 数据
        metrics_per_type = []
        for i in range(len(types)):
            metrics = [per_type[j][i] for j in range(len(per_type))]
            metrics_per_type.append(metrics)

        # 各类别
        for m, t in zip(metrics_per_type, types):
            print(format_row(self._get_row(m, t)))

        # micro / macro
        print(format_row(self._get_row(micro, 'micro')))
        print(format_row(self._get_row(macro, 'macro')))


    def _get_row(self, data, label):
        row = [label]
        for i in range(len(data) - 1):
            row.append(f"{data[i] * 100:.2f}")
        row.append(str(data[3]))
        return row




    def compute_scores(self, _gt_entities, _pred_entities):
        print("Evaluation")
        print("")
        print("--- GMNER ---")
        print("")
        gt, pred = self._convert_by_setting(_gt_entities, _pred_entities, include_entity_types=True, include_region=True)
        gmner_eval = self._score(gt, pred, print_results=True, only_text=False)

        print("")
        print("--- EEG ---")
        print("")
        gt, pred = self._convert_by_setting(_gt_entities, _pred_entities, only_eeg=True)
        eeg_eval = self._score(gt, pred, print_results=True, only_text=False)

        print("")
        print("--- MNER ---")
        # print("An entity is considered correct if the entity type and span is predicted correctly")
        print("")
        gt, pred = self._convert_by_setting(_gt_entities, _pred_entities, include_entity_types=True)
        ner_eval = self._score(gt, pred, print_results=True)

        # print("")
        # print("--- MNER on Localization ---")
        # print("")
        # gt_wo_type, pred_wo_type = self._convert_by_setting(_gt_entities, _pred_entities, include_entity_types=False)
        # ner_loc_eval = self._score(gt_wo_type, pred_wo_type, print_results=True)

        # print("")
        # print("--- MNER on Classification ---")
        # # print("An entity is considered correct if the entity type and span is predicted correctly")
        # print("")
        # # gt, pred = _convert_by_setting(_gt_entities, _pred_entities, include_entity_types=True)
        # ner_cls_eval = self._score(gt, pred, print_results=True, cls_metric=True)

        return gmner_eval, eeg_eval, ner_eval


def main1(gt_path, result_path):
    evaluator = EntityEvaluator(label_dict= {"LOC": 1, "PER": 2, "ORG": 3, "MISC": 4, "None-type":5}, gt_path=gt_path)
    # 模型输出结果样例
    prediction_data = load_cot_prediction(result_path)

    # 评估原始结果
    evaluator.evaluate(prediction_data)



def main2(gt_path, result_path):
    split_dataset = split_unseen_tag_dataset(gt_path, result_path)
    # import pdb;pdb.set_trace()
    result = []
    for pairs in split_dataset:
        evaluator = EntityEvaluator(label_dict={"LOC": 1, "PER": 2, "ORG": 3, "MISC": 4, "None-type":5}, gt_path=gt_path)
        evaluator.gt_entities = pairs["gt"]
        res = evaluator.evaluate(pairs["pred"])
        result.append(list(map(lambda x: round(x, 2), [i[2] for i in res])))
    print(result)

if __name__ == '__main__':
    gt_path = "xxx/cold_start/evaluate/top_10_gt.json"
    result_path = "xxx/cold_start/infer/predictions.json"
    main1(gt_path, result_path)