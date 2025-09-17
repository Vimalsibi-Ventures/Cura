from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os

# --- Add project root to the Python path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
# -----------------------------------------

from core.emergency_checker import is_emergency
from core.symptom_normalizer import normalize_symptoms
from core.kg_lookup import find_conditions
from core.llm_wrapper import get_llm_response
from utils.logger import get_logger

# Initialize logger and FastAPI app
logger = get_logger(__name__)
app = FastAPI(
    title="Symptom Checker Chatbot API",
    description="An API that uses LLM and a Knowledge Graph for preliminary symptom analysis.",
    version="1.0.0"
)

# --- Define the request and response models ---
class UserQuery(BaseModel):
    text: str

class ChatbotResponse(BaseModel):
    response: str

# --- API Endpoints ---

@app.get("/", tags=["Status"])
def read_root():
    """A simple endpoint to check if the API is running."""
    return {"status": "ok", "message": "Welcome to the Symptom Checker Chatbot API"}

@app.post("/diagnose", response_model=ChatbotResponse, tags=["Chatbot"])
async def diagnose(query: UserQuery):
    """
    Main endpoint to process a user's health query.
    """
    user_input = query.text
    logger.info(f"Received new query: '{user_input}'")

    # 1. First, check for emergencies
    if is_emergency(user_input):
        logger.warning(f"Emergency detected in input: '{user_input}'")
        emergency_message = (
            "Your symptoms may indicate a medical emergency. "
            "Please seek immediate medical attention or contact emergency services."
        )
        return ChatbotResponse(response=emergency_message)

    # 2. Normalize symptoms using the LLM and alias map
    normalized = normalize_symptoms(user_input)
    if not normalized:
        logger.info("Could not normalize any symptoms from input.")
        return ChatbotResponse(response="I'm sorry, I couldn't identify any specific symptoms from your message. Could you please describe them differently?")

    # 3. Look up conditions in the Knowledge Graph
    conditions = find_conditions(normalized)
    if not conditions:
        logger.info("No matching conditions found in the KG.")
        return ChatbotResponse(response="Based on your symptoms, I couldn't find a specific match in my knowledge base. It's always best to consult a doctor for an accurate diagnosis.")

    # 4. Use LLM to wrap the findings into a natural response
    try:
        conditions_summary = ", ".join([cond['name'] for cond in conditions])
        recommendations = "\n".join([f"- For {cond['name']}: {cond['recommendation']}" for cond in conditions])
        
        system_message = (
            "You are a helpful medical assistant. Your role is to present potential medical conditions and their recommendations "
            "in a clear, friendly, and responsible way. You must follow the user's prompt structure exactly."
        )
        
        # --- FINAL, TEMPLATE-BASED PROMPT FOR RELIABILITY ---
        final_prompt = f"""
        You are a helpful AI assistant. Your task is to generate a response based on the provided data. Structure your response EXACTLY as follows.

        ### START OF DATA
        User's original query: "{user_input}"
        Possible conditions my knowledge base suggests: {conditions_summary}
        Recommendations for these conditions:
        {recommendations}
        ### END OF DATA

        ### REQUIRED RESPONSE STRUCTURE
        
        **Part 1: Empathetic Acknowledgment.**
        Start with one or two sentences acknowledging the user's symptoms in a caring tone.
        
        **Part 2: Main Body.**
        Present the possible conditions and their corresponding recommendations from the data provided. You can format this as a list or paragraph.
        
        **Part 3: Mandatory Safety Disclaimer.**
        You MUST end your entire response with the following text block, with no changes:
        "Please remember that I am an AI assistant and not a medical professional. This information is for educational purposes only and is not a substitute for a professional diagnosis. For an accurate assessment, it is essential to consult a qualified healthcare provider. If your symptoms persist or worsen, please seek medical attention promptly."
        ### END OF STRUCTURE

        Now, generate the complete response based on the data and the required structure.
        """
        # --- END OF PROMPT UPDATE ---
        
        final_response = get_llm_response(final_prompt, system_message=system_message)
        if not final_response:
             raise ValueError("LLM failed to generate the final response.")
             
        return ChatbotResponse(response=final_response)

    except Exception as e:
        logger.error(f"Error during final response generation: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred while generating the final response.")