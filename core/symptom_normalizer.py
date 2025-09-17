import json
import os
from typing import List, Set
from core.llm_wrapper import get_llm_response
from utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# --- Build a Symptom Alias Map from the Knowledge Graph ---
KG_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'kg.json')
ALIAS_MAP = {}

try:
    with open(KG_FILE_PATH, 'r') as f:
        knowledge_graph = json.load(f)
        
        for condition_data in knowledge_graph.get("conditions", []):
            symptom_lists = condition_data.get("common_symptoms", []) + condition_data.get("specific_symptoms", [])
            for symptom_obj in symptom_lists:
                official_name = symptom_obj["name"].lower()
                # Map the official name to itself
                ALIAS_MAP[official_name] = official_name
                # Map all aliases to the official name
                for alias in symptom_obj.get("aliases", []):
                    ALIAS_MAP[alias.lower()] = official_name

    logger.info(f"Built symptom alias map with {len(ALIAS_MAP)} entries.")
except Exception as e:
    logger.error(f"Failed to build symptom alias map from knowledge graph: {e}")

def normalize_symptoms(user_input: str) -> List[str]:
    """
    Uses an LLM to extract potential symptom phrases and then uses a local
    alias map with flexible matching to normalize them to official symptom names.
    """
    system_message = (
        "You are an expert medical recognition tool. Your task is to identify and extract "
        "any potential medical symptoms from the user's text. List the phrases that look like symptoms, "
        "separated by commas."
    )
    
    prompt = f"""
    From the user input below, extract a comma-separated list of phrases that describe medical symptoms.
    Do not add any explanation or any other text.
    User Input: "{user_input}"
    """

    llm_response = get_llm_response(prompt, system_message=system_message)

    if not llm_response:
        logger.warning("LLM could not extract any symptom phrases.")
        return []

    raw_phrases = [phrase.strip().lower() for phrase in llm_response.split(',')]
    
    # --- NEW ROBUST MATCHING LOGIC ---
    normalized_symptoms: Set[str] = set()
    # Check each phrase extracted by the LLM
    for phrase in raw_phrases:
        # See if any of our known aliases are contained within that phrase
        for alias, official_name in ALIAS_MAP.items():
            if alias in phrase:
                normalized_symptoms.add(official_name)
                # Once a match is found for a phrase, we can move to the next phrase
                break 
    # --- END OF NEW LOGIC ---
    
    logger.info(f"LLM extracted: {raw_phrases}. Normalized to: {list(normalized_symptoms)}")
    return list(normalized_symptoms)