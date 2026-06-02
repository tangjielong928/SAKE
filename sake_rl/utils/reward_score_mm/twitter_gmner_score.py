import re
import json


def calculate_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes.
    box format: [x1, y1, x2, y2]
    """
    if not box1 or not box2 or len(box1) != 4 or len(box2) != 4:
        return 0.0
    
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def normalize_entity(entity_str):
    """Normalize entity string for comparison."""
    return entity_str.lower().strip()


def scale_bbox_to_original(bbox, processed_size, original_size):
    """
    Scale bbox from processed image size to original image size.
    
    Args:
        bbox: [x1, y1, x2, y2] in processed image coordinates
        processed_size: (width, height) of processed image
        original_size: (width, height) of original image
    
    Returns:
        Scaled bbox [x1, y1, x2, y2] in original image coordinates
    """
    if not bbox or len(bbox) != 4 or not isinstance(bbox, list):
        return []
    
    proc_w, proc_h = processed_size
    orig_w, orig_h = original_size
    
    # Calculate scale factors
    scale_x = orig_w / proc_w if proc_w > 0 else 1.0
    scale_y = orig_h / proc_h if proc_h > 0 else 1.0
    
    # Scale coordinates
    try:
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        scaled_bbox = [
            int(x1 * scale_x),
            int(y1 * scale_y),
            int(x2 * scale_x),
            int(y2 * scale_y)
        ]
    except Exception as e:
        print(f"Error scaling bbox: {e}, predicted_bbox: {bbox}, scale_x: {scale_x}, scale_y: {scale_y}")
        scaled_bbox = []
    return scaled_bbox


def calculate_mner_f1_score(predictions, ground_truths):
    """
    Calculate F1 score for MNER task (entity + type only).
    Entity and type match is considered correct.
    
    Args:
        predictions: list of dicts with keys: entity, type, box2d
        ground_truths: list of dicts with keys: entity, type, bbox
    
    Returns:
        f1_score: float
        precision: float  
        recall: float
        tp_count: int
        fp_count: int
        fn_count: int
    """
    if not predictions and not ground_truths:
        return 1.0, 1.0, 1.0, 0, 0, 0
    
    if not predictions:
        return 0.0, 0.0, 0.0, 0, 0, len(ground_truths)
    
    if not ground_truths:
        return 0.0, 0.0, 0.0, 0, len(predictions), 0
    
    tp = 0
    fp = 0
    fn = 0
    
    matched_gt_indices = set()
    for pred in predictions:
        pred_entity = normalize_entity(pred.get('entity', ''))
        pred_type = pred.get('type', '').strip()
        
        matched = False
        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx in matched_gt_indices:
                continue
            
            gt_entity = normalize_entity(gt.get('entity', ''))
            gt_type = gt.get('type', '').strip()
            
            # Check entity and type match
            if pred_entity == gt_entity and pred_type == gt_type:
                tp += 1
                matched_gt_indices.add(gt_idx)
                matched = True
                break
        
        if not matched:
            fp += 1
    
    fn = len(ground_truths) - len(matched_gt_indices)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return f1, precision, recall, tp, fp, fn


def calculate_eeg_f1_score(predictions, ground_truths, iou_threshold=0.5, image_size_info=None):
    """
    Calculate F1 score for EEG task (entity + box2d only).
    Entity match and IoU > threshold is considered correct.
    
    Args:
        predictions: list of dicts with keys: entity, type, box2d
        ground_truths: list of dicts with keys: entity, type, bbox
        iou_threshold: IoU threshold for bbox matching (default: 0.5)
        image_size_info: list of dicts with 'original_size' and 'processed_size' for each image
    
    Returns:
        f1_score: float
        precision: float  
        recall: float
        tp_count: int
        fp_count: int
        fn_count: int
    """
    if not predictions and not ground_truths:
        return 1.0, 1.0, 1.0, 0, 0, 0
    
    if not predictions:
        return 0.0, 0.0, 0.0, 0, 0, len(ground_truths)
    
    if not ground_truths:
        return 0.0, 0.0, 0.0, 0, len(predictions), 0
    
    # Get image size info for bbox scaling (use first image if available)
    original_size = None
    processed_size = None
    if image_size_info and len(image_size_info) > 0:
        size_info = image_size_info[0]  # Use first image's size info
        original_size = size_info.get('original_size')
        processed_size = size_info.get('processed_size')
    
    tp = 0
    fp = 0
    fn = 0
    
    matched_gt_indices = set()
    for pred in predictions:
        pred_entity = normalize_entity(pred.get('entity', ''))
        pred_box = pred.get('box2d', [])
        
        # Scale predicted bbox from processed size to original size
        if len(pred_box)==4 and original_size is not None and processed_size is not None:
            pred_box = scale_bbox_to_original(pred_box, processed_size, original_size)
        
        matched = False
        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx in matched_gt_indices:
                continue
            
            gt_entity = normalize_entity(gt.get('entity', ''))
            gt_box = gt.get('bbox', [])
            
            # Check entity match
            if pred_entity == gt_entity:
                # Check bbox match
                if len(pred_box)!=4 and len(gt_box)!=4:
                    # Both empty boxes - match
                    tp += 1
                    matched_gt_indices.add(gt_idx)
                    matched = True
                    break
                elif len(pred_box)==4 and len(gt_box)==4:
                    # Calculate IoU (both boxes are now in original image coordinates)
                    iou = calculate_iou(pred_box, gt_box)
                    if iou >= iou_threshold:
                        tp += 1
                        matched_gt_indices.add(gt_idx)
                        matched = True
                        break
        
        if not matched:
            fp += 1
    
    fn = len(ground_truths) - len(matched_gt_indices)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return f1, precision, recall, tp, fp, fn


def calculate_f1_score(predictions, ground_truths, iou_threshold=0.5, image_size_info=None):
    """
    Calculate F1 score for GMNER task (entity + type + box2d).
    
    Args:
        predictions: list of dicts with keys: entity, type, box2d
        ground_truths: list of dicts with keys: entity, type, bbox
        iou_threshold: IoU threshold for bbox matching (default: 0.5)
        image_size_info: list of dicts with 'original_size' and 'processed_size' for each image
    
    Returns:
        f1_score: float
        precision: float  
        recall: float
        tp_count: int
        fp_count: int
        fn_count: int
    """
    if not predictions and not ground_truths:
        return 1.0, 1.0, 1.0, 0, 0, 0
    
    if not predictions:
        return 0.0, 0.0, 0.0, 0, 0, len(ground_truths)
    
    if not ground_truths:
        return 0.0, 0.0, 0.0, 0, len(predictions), 0
    
    # Get image size info for bbox scaling (use first image if available)
    original_size = None
    processed_size = None
    if image_size_info and len(image_size_info) > 0:
        size_info = image_size_info[0]  # Use first image's size info
        original_size = size_info.get('original_size')
        processed_size = size_info.get('processed_size')
    
    tp = 0
    fp = 0
    fn = 0
    
    matched_gt_indices = set()
    for pred in predictions:
        pred_entity = normalize_entity(pred.get('entity', ''))
        pred_type = pred.get('type', '').strip()
        pred_box = pred.get('box2d', [])
        
        # Scale predicted bbox from processed size to original size
        if len(pred_box)==4 and original_size is not None and processed_size is not None:
            pred_box = scale_bbox_to_original(pred_box, processed_size, original_size)
        
        matched = False
        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx in matched_gt_indices:
                continue
            
            gt_entity = normalize_entity(gt.get('entity', ''))
            gt_type = gt.get('type', '').strip()
            gt_box = gt.get('bbox', [])
            
            # Check entity and type match
            if pred_entity == gt_entity and pred_type == gt_type:
                # Check bbox match
                if len(pred_box)!=4 and len(gt_box)!=4:
                    # Both empty boxes - match
                    tp += 1
                    matched_gt_indices.add(gt_idx)
                    matched = True
                    break
                elif len(pred_box)==4 and len(gt_box)==4:
                    # Calculate IoU (both boxes are now in original image coordinates)
                    iou = calculate_iou(pred_box, gt_box)
                    if iou >= iou_threshold:
                        tp += 1
                        matched_gt_indices.add(gt_idx)
                        matched = True
                        break
        
        if not matched:
            fp += 1
    
    fn = len(ground_truths) - len(matched_gt_indices)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return f1, precision, recall, tp, fp, fn


def extract_answer_json(prediction):
    """Extract JSON from <answer>...</answer> tag."""
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, prediction, re.DOTALL))
    
    if not matches:
        return None
    
    answer_str = matches[-1].group(1).strip()
    
    try:
        answer_json = json.loads(answer_str)
        return answer_json
    except json.JSONDecodeError as e:
        print(f"Error parsing answer JSON: {answer_str}")
        print("错误信息:", e)
        return None


def extract_search_json(prediction, search_type='text'):
    """
    Extract search queries from tags.
    search_type: 'text' for <text_search>...</text_search>
                 'image' for <image_search>...</image_search>
    """
    if search_type == 'text':
        pattern = r'<text_search>(.*?)</text_search>'
    elif search_type == 'image':
        pattern = r'<image_search>(.*?)</image_search>'
    else:
        return None
    
    matches = list(re.finditer(pattern, prediction, re.DOTALL))
    
    if not matches:
        return None
    
    search_str = matches[-1].group(1).strip()
    
    try:
        search_json = json.loads(search_str)
        return search_json
    except json.JSONDecodeError:
        return None


def is_valid_answer_json(answer_json):
    """
    Validate answer JSON format.
    Expected format: list of dicts with keys: entity, type, box2d
    """
    if not isinstance(answer_json, list):
        return False
    
    for item in answer_json:
        if not isinstance(item, dict):
            return False
        if 'entity' not in item or 'type' not in item or 'box2d' not in item:
            return False
        if not isinstance(item['entity'], str):
            return False
        if not isinstance(item['type'], str):
            return False
        if not isinstance(item['box2d'], list):
            return False
        # box2d should be either empty or have 4 numbers
        if item['box2d'] and len(item['box2d']) != 4:
            return False
    
    return True


def is_valid_search_json(search_json):
    """
    Validate search JSON format.
    Expected format: list of dicts with key: query
    """
    if not isinstance(search_json, list):
        return False
    
    for item in search_json:
        if not isinstance(item, dict):
            return False
        if 'query' not in item:
            return False
        if not isinstance(item['query'], str):
            return False
    
    return True


def is_valid_direct_answer(response, direct_answer_format) -> bool:
    """
    Check Direct Answer: <reason>...</reason><answer>...</answer>
      1) Structure Matching
      2) Pattern Count: <reason>...</reason>, <answer>...</answer>
      3) No any search actions included
      4) Valid JSON format in <answer>
    """
    pattern = direct_answer_format
    # 1). Structure Matching
    if not re.match(pattern, response, re.DOTALL):
        return False
    # 2). Pattern Count
    if response.count('<reason>') != 1 or response.count('</reason>') != 1:
        return False
    if response.count('<answer>') != 1 or response.count('</answer>') != 1:
        return False
    # 3). Search actions not allowed
    if '<text_search>' in response or '</text_search>' in response:
        return False
    if '<image_search>' in response or '</image_search>' in response:
        return False
    # 4). Validate JSON format
    answer_json = extract_answer_json(response)
    if answer_json is None or not is_valid_answer_json(answer_json):
        return False
    return True


def is_valid_image_search(response, call_image_search_format) -> bool:
    """
    Check Image Search: <reason>...</reason>...<image_search>[{"query": "..."}]</image_search>
      1) Structure Matching
      2) Pattern Count
      3) Valid JSON format in <image_search>
    """
    pattern = call_image_search_format
    # 1). Structure Matching
    if not re.match(pattern, response, re.DOTALL):
        return False
    # 2). Pattern Count
    if response.count('<reason>') != 1 or response.count('</reason>') != 1:
        return False
    if response.count('<image_search>') != 1 or response.count('</image_search>') != 1:
        return False
    # 3). Answer or text_search not allowed
    if '<answer>' in response or '</answer>' in response:
        return False
    if '<text_search>' in response or '</text_search>' in response:
        return False
    # 4). Validate JSON format
    search_json = extract_search_json(response, 'image')
    if search_json is None or not is_valid_search_json(search_json):
        return False
    return True


def is_valid_text_search(response, call_text_search_format) -> bool:
    """
    Check Text Search: <reason>...</reason>...<text_search>[{"query": "..."}]</text_search>
      1) Structure Matching
      2) Pattern Count
      3) Valid JSON format in <text_search>
    """
    pattern = call_text_search_format
    # 1). Structure Matching
    if not re.match(pattern, response, re.DOTALL):
        return False
    # 2). Pattern Count
    if response.count('<reason>') != 1 or response.count('</reason>') != 1:
        return False
    if response.count('<text_search>') != 1 or response.count('</text_search>') != 1:
        return False
    # 3). Answer or image_search not allowed
    if '<answer>' in response or '</answer>' in response:
        return False
    if '<image_search>' in response or '</image_search>' in response:
        return False
    # 4). Validate JSON format
    search_json = extract_search_json(response, 'text')
    if search_json is None or not is_valid_search_json(search_json):
        return False
    return True


def format_reward(input_string: list):
    """
    Check if the model's response follows the required formats and return a reward.
    [1-turn]:
        - Direct Answer
    [2-turn]:
        - Call Image Search + Answer
        - Call Text Search + Answer
    [3-turn]:
        - Call Image Search + Call Text Search + Answer
    Args:
    - input_string (list): A list of responses
    Returns:
    - format_score: float, 1.0 for right format, 0.0 for wrong
    - search_count: int, times of search tools called
    """
    conv_rounds = len(input_string)
    format_score, search_count = 0, 0
    
    # All allowed formats
    direct_answer_format = r'^<reason>.*</reason>.*<answer>.*</answer>$'
    call_image_search_format = r'^<reason>.*</reason>.*<image_search>.*</image_search>$'
    call_text_search_format = r'^<reason>.*</reason>.*<text_search>.*</text_search>$'
    
    # 1-turn
    if conv_rounds == 1:
        response_1 = input_string[0].strip()
        if ('<image_search>' in response_1) or ('<text_search>' in response_1):
            search_count += 1
        # Direct Answer
        if is_valid_direct_answer(response_1, direct_answer_format):
            format_score = 1
    # 2-turn
    elif conv_rounds == 2:
        response_1, response_2 = input_string[0].strip(), input_string[1].strip()
        if ('<image_search>' in response_1) or ('<text_search>' in response_1):
            search_count += 1
        # Call Image Search + Answer
        if is_valid_image_search(response_1, call_image_search_format) and is_valid_direct_answer(response_2, direct_answer_format):
            format_score = 1
        # Call Text Search + Answer
        elif is_valid_text_search(response_1, call_text_search_format) and is_valid_direct_answer(response_2, direct_answer_format):
            format_score = 1
    # 3-turn
    elif conv_rounds == 3:
        response_1, response_2, response_3 = input_string[0].strip(), input_string[1].strip(), input_string[2].strip()
        if ('<image_search>' in response_1) or ('<text_search>' in response_1):
            search_count += 1
        if ('<image_search>' in response_2) or ('<text_search>' in response_2):
            search_count += 1
        # Call Image Search + Call Text Search + Answer
        if (
            is_valid_image_search(response_1, call_image_search_format)
            and is_valid_text_search(response_2, call_text_search_format)
            and is_valid_direct_answer(response_3, direct_answer_format)
        ):
            format_score = 1
    else:
        raise ValueError(f"[Error Occured] Number of responses is {conv_rounds}, which is not supported currently!")
    
    return format_score, search_count


def compute_score(prediction: list, ground_truth: list, extra_info=None):
    """
    Compute score for Twitter-GMNER task.
    
    Args:
        prediction: list of response strings
        ground_truth: list of dicts with keys: entity, type, bbox
        extra_info: dict with optional parameters
            - reward_mode: 'EM' or 'MultiEM'
            - lambda_1, lambda_2, lambda_3: weights for MultiEM (default: 0.25, 0.25, 0.5)
    
    Returns:
        score: float, weighted combination of F1 and format scores
    """
    search_penalty, format_penalty = 0.01, 0.1
    iou_threshold = 0.5
    reward_mode = 'EM'  # default to EM
    lambda_1, lambda_2, lambda_3 = 0.25, 0.25, 0.5  # default MultiEM weights
    
    if extra_info is not None:
        # search_penalty = extra_info.get('search_penalty', 0)
        format_penalty = extra_info.get('format_penalty', 0.1)
        iou_threshold = extra_info.get('iou_threshold', 0.5)
        reward_mode = extra_info.get('reward_mode', 'EM')
        lambda_1 = extra_info.get('lambda_1', 0.25)
        lambda_2 = extra_info.get('lambda_2', 0.25)
        lambda_3 = extra_info.get('lambda_3', 0.5)
    
    # Extract Answer
    # print("调试prediction：", prediction)
    # print("调试ground_truth：", ground_truth)
    assert len(prediction) > 0, "[Error Occured] Model Responses are empty!"
    answer_json = extract_answer_json(prediction[-1])
    
    # Parse ground truth
    try:
        if isinstance(ground_truth, list):
            gt_entities = json.loads(ground_truth[-1])
        else:
            gt_entities = json.loads(ground_truth)
    except json.JSONDecodeError:
        print(f"Error parsing ground truth JSON: {ground_truth}")
        gt_entities = []
    
    score = 0
    # Get image size info for bbox scaling
    image_size_info = None
    if extra_info is not None:
        image_size_info = extra_info.get('image_size_info')
    
    # Correctness Check: F1 Score
    if answer_json is not None and is_valid_answer_json(answer_json):
        if reward_mode == 'EM':
            # EM mode: use GMNER F1 score
            f1_score, precision, recall, tp, fp, fn = calculate_f1_score(
                answer_json, gt_entities, iou_threshold, image_size_info=image_size_info
            )
            score = f1_score
            
            # Store metrics in extra_info for logging
            if extra_info is not None:
                extra_info['f1_score'] = f1_score
                extra_info['precision'] = precision
                extra_info['recall'] = recall
                extra_info['tp'] = tp
                extra_info['fp'] = fp
                extra_info['fn'] = fn
                extra_info['gmner_f1_score'] = f1_score  # alias for consistency
        
        elif reward_mode == 'MultiEM':
            # MultiEM mode: weighted combination of MNER, EEG, and GMNER scores
            mner_f1, mner_precision, mner_recall, mner_tp, mner_fp, mner_fn = calculate_mner_f1_score(
                answer_json, gt_entities
            )
            eeg_f1, eeg_precision, eeg_recall, eeg_tp, eeg_fp, eeg_fn = calculate_eeg_f1_score(
                answer_json, gt_entities, iou_threshold, image_size_info=image_size_info
            )
            gmner_f1, gmner_precision, gmner_recall, gmner_tp, gmner_fp, gmner_fn = calculate_f1_score(
                answer_json, gt_entities, iou_threshold, image_size_info=image_size_info
            )
            
            # Weighted combination
            score = lambda_1 * mner_f1 + lambda_2 * eeg_f1 + lambda_3 * gmner_f1
            
            # Store metrics in extra_info for logging
            if extra_info is not None:
                # Store individual scores
                extra_info['mner_f1_score'] = mner_f1
                extra_info['mner_precision'] = mner_precision
                extra_info['mner_recall'] = mner_recall
                extra_info['mner_tp'] = mner_tp
                extra_info['mner_fp'] = mner_fp
                extra_info['mner_fn'] = mner_fn
                
                extra_info['eeg_f1_score'] = eeg_f1
                extra_info['eeg_precision'] = eeg_precision
                extra_info['eeg_recall'] = eeg_recall
                extra_info['eeg_tp'] = eeg_tp
                extra_info['eeg_fp'] = eeg_fp
                extra_info['eeg_fn'] = eeg_fn
                
                extra_info['gmner_f1_score'] = gmner_f1
                extra_info['gmner_precision'] = gmner_precision
                extra_info['gmner_recall'] = gmner_recall
                extra_info['gmner_tp'] = gmner_tp
                extra_info['gmner_fp'] = gmner_fp
                extra_info['gmner_fn'] = gmner_fn
                
                # Store combined score (for backward compatibility)
                extra_info['f1_score'] = score
                extra_info['precision'] = gmner_precision  # use GMNER precision as default
                extra_info['recall'] = gmner_recall  # use GMNER recall as default
                extra_info['tp'] = gmner_tp
                extra_info['fp'] = gmner_fp
                extra_info['fn'] = gmner_fn
        else:
            raise ValueError(f"Unknown reward_mode: {reward_mode}. Must be 'EM' or 'MultiEM'")
    
    # Format Check
    format_score, search_count = format_reward(prediction)
    
    # Search Penalty (only apply to correct answers)
    if search_count > 0 and score > 0.8:
        use_search_count_penalty = extra_info.get('use_search_count_penalty', False) if extra_info else False
        if use_search_count_penalty:
            # penalty w/ search count
            for _ in range(search_count):
                score = score - search_penalty
    

    final_score = (1 - format_penalty) * score + format_penalty * format_score
    
    return final_score
