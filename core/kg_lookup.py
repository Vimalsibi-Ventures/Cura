import json
import os
from typing import List, Dict, Any
from utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# --- Load the Knowledge Graph ---
KG_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'kg.json')

try:
    with open(KG_FILE_PATH, 'r') as f:
        knowledge_graph = json.load(f)
    # UPDATED: We now expect a list of conditions, not a dictionary.
    CONDITIONS = knowledge_graph.get("conditions", [])
    logger.info(f"Knowledge graph with {len(CONDITIONS)} conditions and alias structure loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load or parse knowledge graph conditions: {e}")
    CONDITIONS = []

def find_conditions(normalized_symptoms: List[str]) -> List[Dict[str, Any]]:
    """
    Finds the most relevant medical conditions using a weighted scoring model.
    It reads the new KG structure containing symptom objects with names and aliases.

    Args:
        normalized_symptoms (List[str]): A list of clean, normalized symptoms from the symptom_normalizer.

    Returns:
        List[Dict[str, Any]]: A sorted list of the top 3 matching condition objects.
    """
    if not normalized_symptoms or not CONDITIONS:
        return []

    symptom_set = set(normalized_symptoms)
    scored_conditions = []

    # Weights for scoring
    COMMON_SYMPTOM_WEIGHT = 1
    SPECIFIC_SYMPTOM_WEIGHT = 3

    for condition_data in CONDITIONS:
        # --- THIS IS THE UPGRADED LOGIC ---
        # Extract the 'name' from each symptom object to create a set of strings for matching.
        common_symptoms_names = {s_obj['name'] for s_obj in condition_data.get("common_symptoms", [])}
        specific_symptoms_names = {s_obj['name'] for s_obj in condition_data.get("specific_symptoms", [])}
        # ------------------------------------

        # The rest of the logic remains the same, but uses the new sets of names.
        common_matches = len(symptom_set.intersection(common_symptoms_names))
        specific_matches = len(symptom_set.intersection(specific_symptoms_names))
        
        score = (common_matches * COMMON_SYMPTOM_WEIGHT) + (specific_matches * SPECIFIC_SYMPTOM_WEIGHT)

        if score > 0:
            scored_conditions.append({
                "score": score,
                "data": condition_data
            })
    
    if not scored_conditions:
        logger.info("No matching conditions found for the given symptoms.")
        return []

    # Sort the list of dictionaries by score in descending order
    sorted_conditions = sorted(scored_conditions, key=lambda x: x['score'], reverse=True)
    
    # Get just the data part of the top 3 results
    top_matches = [item['data'] for item in sorted_conditions[:3]]
    
    logger.info(f"Found {len(top_matches)} relevant matches. Best score: {sorted_conditions[0]['score']}")
    return top_matches
