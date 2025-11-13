from typing import Any, Dict, List, Optional
import json
import math

DECISION_THRESHOLD = 0.85
RANK_LAMBDA = 0.36
POSITION_SCORES = [1.00, 0.60, 0.35]
OVERLAP_BASE = 0.475
SCORE_CONF_SIGMA = 0.175  # Constant sigma for Gaussian confidence scoring

def clamp_to_unit_interval(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value

def gaussian_distance_score(value: float, center: float, spread: float) -> float:
    spread = max(1e-6, spread)
    normalized_distance = (value - center) / spread
    return math.exp(-0.5 * normalized_distance * normalized_distance)


def parse_json_response(response_object: Any) -> Optional[Dict[str, Any]]:
    if isinstance(response_object, dict):
        return response_object
    if not isinstance(response_object, str):
        return None
    parsed_data = json.loads(response_object)
    return parsed_data if isinstance(parsed_data, dict) else None

def evaluate_select_chapters(user_data: Dict, assistant_data: Dict, targets: Dict, threshold: float) -> Dict:
    target_chapters = targets.get("chapters", [])
    if not isinstance(target_chapters, list) or len(target_chapters) < 1:
        return {"score": 0.0, "is_score_valid": False, "reason": "targets.chapters must be list with at least one item"}
    ground_truth_entry = target_chapters[0]
    if not isinstance(ground_truth_entry, dict) or "chapter" not in ground_truth_entry:
        return {"score": 0.0, "is_score_valid": False, "reason": "first target chapter must be dict with chapter key"}
    ground_truth_chapter = str(ground_truth_entry["chapter"]).strip()
    predicted_chapters = assistant_data.get("chapters")
    if not isinstance(predicted_chapters, list) or len(predicted_chapters) != 3:
        return {"score": 0.0, "is_score_valid": False, "reason": "assistant must return exactly 3 chapters"}
    ground_truth_position = None
    for index, chapter_entry in enumerate(predicted_chapters):
        if not isinstance(chapter_entry, dict):
            return {"score": 0.0, "is_score_valid": False, "reason": f"chapter at index {index} must be dict"}
        chapter_name = str(chapter_entry.get("chapter", "")).strip()
        if not chapter_name:
            return {"score": 0.0, "is_score_valid": False, "reason": f"chapter at index {index} has empty name"}
        if chapter_name == ground_truth_chapter:
            ground_truth_position = index
    if ground_truth_position is None:
        return {"score": 0.0, "is_score_valid": True, "reason": f"ground truth '{ground_truth_chapter}' not found in predictions"}
    rank_factor_value = math.exp(-RANK_LAMBDA * ground_truth_position)
    clamped_score = clamp_to_unit_interval(rank_factor_value)
    return {
        "score": clamped_score,
        "is_score_valid": True,
        "reason": f"position={ground_truth_position}, rank_factor={rank_factor_value:.3f}"
    }

def evaluate_select_candidates(user_data: Dict, assistant_data: Dict, targets: Dict, threshold: float) -> Dict:
    correct_indices = targets.get("selected_indices", [])
    if not isinstance(correct_indices, list) or len(correct_indices) != 3:
        return {"score": 0.0, "is_score_valid": False, "reason": "targets.selected_indices must have exactly 3 elements"}
    if not all(isinstance(x, int) for x in correct_indices):
        return {"score": 0.0, "is_score_valid": False, "reason": "all correct indices must be integers"}
    predicted_indices = assistant_data.get("selected_indices")
    if not isinstance(predicted_indices, list) or len(predicted_indices) != 3:
        return {"score": 0.0, "is_score_valid": False, "reason": "assistant must return exactly 3 indices"}
    if not all(isinstance(x, int) for x in predicted_indices):
        return {"score": 0.0, "is_score_valid": False, "reason": "all predicted indices must be integers"}
    primary_correct = correct_indices[0]
    if primary_correct not in predicted_indices:
        return {"score": 0.0, "is_score_valid": True, "reason": f"primary correct {primary_correct} missing from predictions"}
    primary_position = predicted_indices.index(primary_correct)
    position_score = POSITION_SCORES[primary_position]
    overlap_fraction = len(set(correct_indices) & set(predicted_indices)) / 3.0
    overlap_dampener = OVERLAP_BASE + (1.0 - OVERLAP_BASE) * overlap_fraction
    adjusted_score = clamp_to_unit_interval(position_score * overlap_dampener)
    return {
        "score": adjusted_score,
        "is_score_valid": True,
        "reason": f"position={primary_position} ({position_score:.3f}); overlap={overlap_fraction:.2f}, dampener={overlap_dampener:.3f}"
    }

def evaluate_score_candidate(user_data: Dict, assistant_data: Dict, targets: Dict, threshold: float) -> Dict:
    correct_confidence = targets.get("confidence")
    if not isinstance(correct_confidence, (int, float)):
        return {"score": 0.0, "is_score_valid": False, "reason": "targets.confidence must be numeric"}
    predicted_confidence = assistant_data.get("confidence")
    if not isinstance(predicted_confidence, (int, float)):
        return {"score": 0.0, "is_score_valid": False, "reason": "assistant.confidence must be numeric"}
    gaussian_score = gaussian_distance_score(predicted_confidence, correct_confidence, SCORE_CONF_SIGMA)
    clamped_gaussian_score = clamp_to_unit_interval(gaussian_score)
    return {
        "score": clamped_gaussian_score,
        "is_score_valid": True,
        "reason": f"pred={predicted_confidence:.2f}, gold={correct_confidence:.2f}, σ={SCORE_CONF_SIGMA:.2f}, G={gaussian_score:.3f}"
    }

def evaluate(user_data: Dict, answer: str, targets: Dict) -> Dict:
    if not isinstance(user_data, dict):
        return {"score": 0.0, "is_score_valid": False, "reason": "user_data must be dict"}
    assistant_response = parse_json_response(answer)
    if assistant_response is None:
        return {"score": 0.0, "is_score_valid": False, "reason": "assistant response must be dict"}
    task_type = user_data.get("task")
    if not isinstance(task_type, str):
        return {"score": 0.0, "is_score_valid": False, "reason": "task must be string"}
    if task_type == "select_chapters":
        evaluation_result = evaluate_select_chapters(user_data, assistant_response, targets, DECISION_THRESHOLD)
    elif task_type == "select_candidates":
        evaluation_result = evaluate_select_candidates(user_data, assistant_response, targets, DECISION_THRESHOLD)
    elif task_type == "score_candidate":
        evaluation_result = evaluate_score_candidate(user_data, assistant_response, targets, DECISION_THRESHOLD)
    else:
        return {"score": 0.0, "is_score_valid": False, "reason": f"unknown task '{task_type}'"}
    return {
        "score": evaluation_result["score"],
        "is_score_valid": evaluation_result["is_score_valid"],
        "reason": evaluation_result["reason"]
    }
