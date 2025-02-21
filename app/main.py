from fastapi import FastAPI
from pydantic import BaseModel
import openai
import redis
import json
from langchain_openai import ChatOpenAI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Connect to Redis
redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

# OpenAI API (Replace with DeepSeek if needed)
openai.api_key = ""
llm = ChatOpenAI(model="gpt-4", openai_api_key=openai.api_key)

# Define form steps - Simplified and conditional
FORM_STEPS = [
    "companyDetails",
    "procurementIntent",
    "roomPlanning",  # Optional - only for fulfillment
    "requirementDetails",
    "procurementModel",
    "budgetTimeline",
    "additionalInfo"
]

# Define which steps are optional
OPTIONAL_STEPS = {"roomPlanning", "additionalInfo"}

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserInput(BaseModel):
    conversation_id: str
    message: str

@app.post("/chat/")
async def chat(input_data: UserInput):
    conversation_id = input_data.conversation_id
    user_message = input_data.message

    conversation_data = get_conversation(conversation_id)
    current_step = conversation_data.get("current_step", FORM_STEPS[0])

    print(f"Before extraction - Conversation data: {conversation_data}")
    
    current_step_data = conversation_data.get(current_step, {})
    extracted_data = extract_fields_ai(user_message, current_step)
    
    print(f"Extracted data: {extracted_data}")
    
    # Ensure we're not nesting data under the step name again
    if current_step in extracted_data:
        extracted_data = extracted_data[current_step]
    
    conversation_data[current_step] = merge_step_data(current_step_data, extracted_data)
    
    print(f"After update - Conversation data: {conversation_data}")
    
    save_conversation(conversation_id, conversation_data)

    if is_step_complete(conversation_data[current_step], current_step):
        next_step = get_next_step(current_step, conversation_data)
        conversation_data["current_step"] = next_step
        save_conversation(conversation_id, conversation_data)

        if next_step:
            ai_question = generate_followup_question_ai(next_step, conversation_data)
            return {"response": ai_question, "form_data": conversation_data}
        return {"response": "All steps completed!", "form_data": conversation_data}

    next_field = get_next_missing_field(conversation_data[current_step], current_step)
    ai_question = generate_followup_question_ai(next_field, conversation_data)
    return {"response": ai_question, "form_data": conversation_data}


def extract_fields_ai(text: str, step: str):
    system_prompt = """You are a helpful assistant that extracts structured data from user input. 
    Always respond with valid JSON only. No other text or explanation.
    Only include fields that you can extract from the text - do not include fields with null values.
    Only extract fields relevant to the current step.
    
    For customerType:
    - If mentioning personal property (house, villa, apartment, home) → set customerType to "personal"
    - If mentioning business (office, company, corporate, hotel) → set customerType to "business"
    
    For procurementIntent:
    - If user mentions "3" or "full project" or "fulfill" → set type to "fulfillment"
    - If user mentions "1" or "purchase products" → set type to "product"
    - If user mentions "2" or "services" → set type to "service"
    
    For requirementDetails:
    - Extract mentioned items and quantities into an array
    - Format: {"requirements": ["2 sofa", "1 table", "3 chairs"]}
    
    For procurementModel:
    - "one-off", "single", "1" → set model to "one-off"
    - "recurring", "regular", "2" → set model to "recurring"
    - "project", "project-based", "3" → set model to "project-based"
    - "tender", "bid", "4" → set model to "tender"
    """
    
    step_fields = {
        "companyDetails": ["customerType", "name", "location"],
        "procurementIntent": ["type"],
        "roomPlanning": ["rooms"],
        "requirementDetails": ["requirements"],
        "procurementModel": ["model"],
        "budgetTimeline": ["budget", "timeline"],
        "additionalInfo": ["additional"]
    }
    
    user_prompt = f"""Extract ONLY the following fields for the current step '{step}': {', '.join(step_fields.get(step, []))}
    
    Text to analyze: {text}
    Current conversation step: {step}
    
    Rules:
    - Only extract fields listed above for the current step
    - For requirementDetails, extract items with quantities as an array
    - Example for requirements: {{"requirements": ["2 sofa", "1 table", "4 chairs"]}}
    - Only include fields you can confidently extract
    - Do not include fields with null values
    - Do not nest the response inside the step name
    
    Return only the JSON object with the extracted values for this step."""

    try:
        response = llm.predict(system_prompt + "\n" + user_prompt)
        response = response.strip()
        if response.startswith('```json'):
            response = response.split('```json')[1]
        if response.endswith('```'):
            response = response[:-3]
        response = response.strip()
        
        parsed_data = json.loads(response) if response else {}
        
        # Handle special cases for each step
        if step == "requirementDetails":
            # Ensure requirements is an array
            if "requirements" in parsed_data and not isinstance(parsed_data["requirements"], list):
                if isinstance(parsed_data["requirements"], dict):
                    # Clean up nested data
                    parsed_data = {"requirements": []}
                else:
                    parsed_data["requirements"] = [parsed_data["requirements"]]
            
            # Extract items from text if not already extracted
            if "requirements" not in parsed_data:
                items = []
                text_lower = text.lower()
                # Look for patterns like "2 sofa" or "three chairs"
                words = text_lower.split()
                for i in range(len(words)-1):
                    if words[i].isdigit() or words[i] in ["one", "two", "three", "four", "five"]:
                        items.append(f"{words[i]} {words[i+1]}")
                if items:
                    parsed_data["requirements"] = items
        
        # Clean up any nested data
        if "procurementIntent" in parsed_data:
            del parsed_data["procurementIntent"]
        if "customerType" in parsed_data and step != "companyDetails":
            del parsed_data["customerType"]
        
        # Ensure we're not nesting data unnecessarily
        if step in parsed_data:
            return parsed_data[step]
        return parsed_data
        
    except Exception as e:
        print(f"Error in extract_fields_ai: {str(e)}")
        return {}


def is_step_complete(step_data: dict, step: str):
    if step in OPTIONAL_STEPS:
        return True
        
    required_fields = {
        "companyDetails": ["customerType", "name", "location"],
        "procurementIntent": ["type"],
        "roomPlanning": ["rooms"],
        "requirementDetails": ["requirements"],
        "procurementModel": ["model"],
        "budgetTimeline": ["budget", "timeline"]
    }
    
    for field in required_fields.get(step, []):
        if field not in step_data or step_data[field] is None:
            return False
    return True


def generate_followup_question_ai(field: str, conversation_data: dict):
    field_questions = {
        "customerType": "Can you please specify if you are seeking our services for personal use or for a business?",
        "name": "What is your name?",
        "location": "What is your location?",
        "type": "Would you like to: \n1. Purchase specific products\n2. Get specific services\n3. Get full project fulfillment (furnishing entire spaces)",
        "rooms": "Which rooms would you like us to furnish? Please list all rooms that need furnishing.",
        "requirements": """Please list all items you need with quantities. For example:
Living Room:
- 2 three-seater sofas
- 4 accent chairs
- 1 coffee table
- 2 side tables
- 1 TV cabinet
- 1 area rug

Bedroom:
- 1 king-size bed
- 2 bedside tables
- 1 wardrobe
- 1 dresser
- 1 vanity set

Please provide your list in a similar format.""",
        "model": "How would you like to proceed with the purchase?\n1. One-off purchase (single transaction)\n2. Recurring order (regular purchases)\n3. Project-based (complete project management)\n4. Tender (competitive bidding)",
        "budget": "What is your budget range for this project? Please specify in your local currency.",
        "timeline": "When would you like this project to be completed? Please specify both preferred start date and completion deadline.",
        "additional": """Do you have any specific requirements for:
1. Style preferences (modern, classic, minimalist, etc.)
2. Color schemes
3. Material preferences (wood type, fabric, etc.)
4. Brand preferences
5. Any specific features needed
6. Installation requirements
Please provide as much detail as possible."""
    }
    
    if field in field_questions:
        # Customize questions based on context
        if field == "type" and "villa" in str(conversation_data).lower():
            return "I understand you want to furnish your villa. Would you like:\n1. To purchase specific furniture items\n2. Full project fulfillment (we handle everything)"
        
        if field == "rooms" and "hotel" in str(conversation_data).lower():
            return """Which hotel areas need furnishing? Please list all areas, such as:
- Guest rooms (specify number of rooms)
- Lobby
- Reception area
- Restaurant/Dining area
- Conference rooms
- Business center
- Spa/Gym
- Other facilities

Please list all areas that need furnishing."""
            
        if field == "requirements":
            intent_type = conversation_data.get("procurementIntent", {}).get("type")
            rooms = conversation_data.get("roomPlanning", {}).get("rooms", [])
            
            if intent_type == "fulfillment":
                if "hotel" in str(conversation_data).lower():
                    return """Please list all furniture and items needed for each area. For example:

Guest Rooms (per room):
- 1 king/queen bed
- 2 bedside tables
- 1 work desk with chair
- 1 lounge chair
- 1 TV with mount
- 1 minibar cabinet
- 1 wardrobe
- 1 luggage rack

Lobby:
- Seating arrangements
- Reception counter
- Decorative items

Please list all items needed with quantities for each area."""
                else:
                    room_prompts = "\n\n".join([f"{room}:\n- Furniture items needed\n- Lighting requirements\n- Storage solutions\n- Decorative elements" for room in rooms]) if rooms else ""
                    return f"""Please list all furniture and items needed for each room with quantities. Be as specific as possible with sizes and any special requirements.

{room_prompts if room_prompts else 'Please list by room:'}"""
            elif intent_type == "product":
                return """Please list all products you want to purchase with:
- Exact quantities
- Preferred dimensions (if applicable)
- Specific materials (if any preference)
- Color preferences
- Any other specific requirements"""
            elif intent_type == "service":
                return """What specific services do you need? Please specify:
- Type of service (design, installation, maintenance, etc.)
- Scope of work
- Specific requirements
- Service frequency (if applicable)"""
                
        if field == "budget":
            intent_type = conversation_data.get("procurementIntent", {}).get("type")
            if intent_type == "fulfillment":
                return """What is your total budget for this furnishing project?
Please specify:
- Total budget range
- Any specific allocation for different areas
- Whether this includes installation and delivery
- Any other cost considerations"""
            elif intent_type == "product":
                return "What is your budget range for these products? Please include any delivery or installation requirements in your budget consideration."
            elif intent_type == "service":
                return "What is your budget allocation for these services? Please specify if this is a one-time budget or recurring budget."
                
        return field_questions[field]
    
    return "Could you please provide more details about your requirements?"


def get_conversation(conversation_id: str):
    data = redis_client.get(conversation_id)
    return json.loads(data) if data else {"current_step": FORM_STEPS[0]}


def save_conversation(conversation_id: str, conversation_data: dict):
    redis_client.set(conversation_id, json.dumps(conversation_data))


def get_next_step(current_step: str, conversation_data: dict):
    current_index = FORM_STEPS.index(current_step)
    while current_index + 1 < len(FORM_STEPS):
        next_step = FORM_STEPS[current_index + 1]
        
        # Skip roomPlanning if not fulfillment
        if next_step == "roomPlanning":
            procurement_type = conversation_data.get("procurementIntent", {}).get("type")
            if procurement_type != "fulfillment":
                current_index += 1
                continue
                
        # Skip optional steps if no data provided
        if next_step in OPTIONAL_STEPS and next_step not in conversation_data:
            current_index += 1
            continue
            
        return next_step
    return None


def get_next_missing_field(step_data: dict, step: str):
    required_fields = {
        "companyDetails": ["customerType", "name", "location"],
        "procurementIntent": ["type"],
        "roomPlanning": ["rooms"],
        "requirementDetails": ["requirements"],
        "procurementModel": ["model"],
        "budgetTimeline": ["budget", "timeline"]
    }
    for field in required_fields.get(step, []):
        if field not in step_data:
            return field
    return None

def merge_step_data(existing_data: dict, new_data: dict) -> dict:
    """Merge new data with existing data, preserving non-null existing values."""
    result = existing_data.copy()
    
    # Remove any fields that shouldn't be in this step
    if "procurementIntent" in result and "type" not in result["procurementIntent"]:
        del result["procurementIntent"]
    
    for key, value in new_data.items():
        # Only update if the new value is not None and either:
        # 1. The key doesn't exist in result, or
        # 2. The existing value is None
        if value is not None and (key not in result or result[key] is None):
            result[key] = value
    return result
