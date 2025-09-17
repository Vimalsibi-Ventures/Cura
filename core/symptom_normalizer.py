import json
import os
from typing import List, Set
from core.llm_wrapper import get_llm_response
from utils.logger import get_logger

logger = get_logger(__name__)
KG_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'kg.json')

# We only need the set of all official symptom names to provide to the LLM
ALL_SYMPTOM_NAMES: Set[str] = set()
try:
    with open(KG_FILE_PATH, 'r', encoding='utf-8') as f:
        knowledge_graph = json.load(f)
        for condition in knowledge_graph.get("conditions", []):
            symptoms = condition.get("common_symptoms", []) + condition.get("specific_symptoms", [])
            for symptom_obj in symptoms:
                ALL_SYMPTOM_NAMES.add(symptom_obj["name"].lower())
    logger.info(f"Loaded {len(ALL_SYMPTOM_NAMES)} unique official symptom names for CoT normalization.")
except Exception as e:
    logger.error(f"Failed to load symptom names for CoT normalization: {e}")

def normalize_symptoms(user_input: str) -> List[str]:
    """
    Uses a "Chain of Thought" prompt to force the LLM to reliably map
    conversational language to a known list of official symptoms.
    """
    if not ALL_SYMPTOM_NAMES:
        logger.error("Official symptom name list is empty. Cannot normalize.")
        return []

    # Prepare the list of official symptoms for the prompt
    symptom_list_str = ", ".join(f'"{s}"' for s in sorted(list(ALL_SYMPTOM_NAMES)))
    
    system_message = "You are an expert at mapping conversational language to official medical symptoms. You must follow the instructions exactly and provide your final answer only in the specified JSON format."
    
    # This is the multi-step "Chain of Thought" prompt
    prompt = f"""
    Analyze the user's text to find the best matches from a list of official medical symptoms. Follow these steps:
    1.  **Extract:** Read the USER TEXT and identify all phrases that describe medical symptoms.
    2.  **Analyze:** Compare each extracted phrase to the LIST OF OFFICIAL SYMPTOMS provided.
    3.  **Map:** Find the most accurate match from the official list for each phrase.
    4.  **Output:** Return your final answer as a single JSON object with one key, "normalized_symptoms", which contains a list of the official symptom names you identified.

    ### LIST OF OFFICIAL SYMPTOMS:
    [{symptom_list_str}]
    
    ### USER TEXT:
    "{user_input}"
    
    ### FINAL JSON OUTPUT:
    """
    
    llm_response = get_llm_response(prompt, system_message)
    
    if not llm_response:
        logger.warning("LLM provided no response for CoT normalization.")
        return []

    try:
        # Extract the JSON part from the LLM's response
        json_start = llm_response.find('{')
        json_end = llm_response.rfind('}') + 1
        if json_start == -1 or json_end == 0:
            raise json.JSONDecodeError("No JSON object found in the response.", llm_response, 0)
        
        json_str = llm_response[json_start:json_end]
        data = json.loads(json_str)
        
        # Validate the response and return a clean list
        symptoms = data.get("normalized_symptoms", [])
        if isinstance(symptoms, list):
            # Final safety check: ensure all returned symptoms are valid
            valid_symptoms = [s.lower() for s in symptoms if s.lower() in ALL_SYMPTOM_NAMES]
            logger.info(f"CoT normalization successful. Found: {valid_symptoms}")
            return valid_symptoms
        else:
            logger.warning("LLM response JSON did not contain a valid list for 'normalized_symptoms'.")
            return []

    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON from LLM for CoT normalization. Response: {llm_response}")
        return []