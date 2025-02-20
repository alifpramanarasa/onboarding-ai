from fastapi import FastAPI
from pydantic import BaseModel
import spacy
import openai
import redis
import json
from langchain_openai import ChatOpenAI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
nlp = spacy.load("en_core_web_sm")  # NLP model

# Connect to Redis
redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

# OpenAI API (Replace with DeepSeek if needed)
openai.api_key = ""
llm = ChatOpenAI(model="gpt-4", openai_api_key=openai.api_key)

# Define form steps
FORM_STEPS = [
    "companyDetails",
    "procurementIntent",
    "roomPlanning",
    "scopeComplexity",
    "budgetTimeline",
    "selectedServiceId",
    "additionalSpecs"
]

VALID_CUSTOMER_TYPES = ["company", "home", "villa", "office", "personal"]
VALID_ROLE_TYPES = ["primary", "agent", "other"]
VALID_PROJECT_TYPES = ["one-off", "recurring", "project-based", "tender"]

# Add this with the other constants near the top of the file
VALID_FURNITURE_ITEMS = [
    "sofa", "chair", "table", "desk", "bed", "cabinet", "vase", "lamp", "shelf",
    "wardrobe", "dresser", "mirror", "rug", "carpet", "curtain", "bookcase",
    "sideboard", "ottoman", "stool", "bench", "dining set", "coffee table"
]

# Add this with the other constants at the top
VALID_ROOM_TYPES = [
    "bedroom", "living room", "dining room", "kitchen", "bathroom",
    "office", "study", "guest room", "master bedroom", "family room"
]

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
    """
    Handles user input, extracts structured data, and determines the next step in conversation.
    """
    conversation_id = input_data.conversation_id
    user_message = input_data.message

    # Step 1: Load conversation history from Redis
    conversation_data = get_conversation(conversation_id)

    # Step 2: Determine current step
    current_step = conversation_data.get("current_step", FORM_STEPS[0])

    # Step 3: Extract relevant data from user input
    current_step_data = conversation_data.get(current_step, {})
    extracted_data = extract_fields(user_message, current_step, current_step_data)
    print("extracted_data", extracted_data)
    conversation_data[current_step] = {**current_step_data, **extracted_data}
    
    # Step 4: Store updated conversation in Redis
    save_conversation(conversation_id, conversation_data)

    # Step 5: Check if all required fields for current step are filled
    if is_step_complete(conversation_data[current_step], current_step):
        next_step = get_next_step(current_step, conversation_data)
        conversation_data["current_step"] = next_step
        save_conversation(conversation_id, conversation_data)

        if next_step:
            ai_question = generate_followup_question(next_step, conversation_data)
            return {"response": ai_question, "form_data": conversation_data}

        return {"response": "All steps completed!", "form_data": conversation_data}

    # Step 6: Generate a follow-up question for missing fields
    next_field = get_next_missing_field(conversation_data[current_step], current_step)
    ai_question = generate_followup_question(next_field, conversation_data)

    return {"response": ai_question, "form_data": conversation_data}


def extract_fields(text: str, step: str, step_data: dict):
    """
    Extracts structured data using NLP (spaCy) based on the current form step.
    """
    doc = nlp(text.lower())
    structured_data = {}

    if step == "companyDetails":
        # Use the passed step_data instead of creating an empty dict
        next_field = get_next_missing_field(step_data, step)
        print("next_field", next_field)
        
        if next_field == "customerType":
            # Only look for customer type when specifically asking for it
            for token in doc:
                if token.text in VALID_CUSTOMER_TYPES:
                    structured_data["customerType"] = token.text
            if "company" in text:
                structured_data["customerType"] = "company"
        
        elif next_field == "name":
            structured_data["name"] = text.strip()
        
        elif next_field == "location":
            structured_data["location"] = text.strip()
        
        elif next_field == "role":
            for token in doc:
                if token.text in VALID_ROLE_TYPES:
                    structured_data["role"] = token.text

    elif step == "procurementIntent":
        # Extract furniture items mentioned in the text
        furniture_items = []
        current_quantity = 1
        
        # Dictionary to convert word numbers to integers
        word_to_number = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
        }
        
        # Keep track of potential furniture items not in our list
        unknown_items = []
        
        for token in doc:
            # Check for numbers (both words and digits)
            if token.like_num:
                try:
                    current_quantity = int(token.text)
                except ValueError:
                    if token.text in word_to_number:
                        current_quantity = word_to_number[token.text]
            # Check for furniture items
            elif token.text in VALID_FURNITURE_ITEMS:
                furniture_items.append({"item": token.text, "quantity": current_quantity})
                current_quantity = 1  # Reset quantity
            # If not a number and not in our list, might be a furniture item
            elif token.pos_ in ['NOUN', 'PROPN']:  # Only consider nouns
                unknown_items.append(token.text)
        
        # If we found unknown items, ask AI if they're furniture
        if unknown_items:
            prompt = f"Are any of these items furniture or home decor items? Please respond with only the items that are furniture/decor in a Python list format: {unknown_items}"
            response = llm.predict(prompt)
            try:
                # Try to safely evaluate the response as a Python list
                ai_validated_items = eval(response.strip())
                if isinstance(ai_validated_items, list):
                    for item in ai_validated_items:
                        furniture_items.append({"item": item, "quantity": current_quantity})
                        current_quantity = 1
            except:
                # If eval fails, just continue with what we have
                pass
        
        if furniture_items:
            structured_data["offerings"] = furniture_items

    elif step == "roomPlanning":
        room_info = {}
        dimensions = []
        
        # Extract room type
        for token in doc:
            if token.text in VALID_ROOM_TYPES:
                room_info["type"] = token.text
        
        # Extract dimensions
        numbers = []
        for token in doc:
            if token.like_num:
                try:
                    numbers.append(float(token.text))
                except ValueError:
                    continue
        
        # If we found two numbers, assume they're dimensions
        if len(numbers) >= 2:
            room_info["dimensions"] = {
                "length": numbers[0],
                "width": numbers[1],
                "unit": "meters"  # Default unit, could be made more sophisticated
            }
            room_info["area"] = numbers[0] * numbers[1]
        
        if room_info:
            structured_data["rooms"] = [room_info]

    elif step == "scopeComplexity":
        # Extract style and specifications
        style_info = {}
        
        # Try to extract overall style if mentioned
        style_keywords = ["modern", "traditional", "contemporary", "classic", "minimalist", "rustic"]
        for token in doc:
            if token.text in style_keywords:
                style_info["style"] = token.text
        
        # Use AI to extract detailed specifications
        prompt = f"Extract furniture specifications from this text and return as a Python dictionary with 'specifications' key containing a list of specs: {text}"
        response = llm.predict(prompt)
        try:
            specs_dict = eval(response.strip())
            if isinstance(specs_dict, dict) and "specifications" in specs_dict:
                style_info["specifications"] = specs_dict["specifications"]
        except:
            # If eval fails, store the raw text as type
            style_info["type"] = text.strip()
        
        if style_info:
            structured_data.update(style_info)

    elif step == "budgetTimeline":
        # Extract budget and timeline information
        budget_info = {}
        
        # Extract numbers for budget
        numbers = []
        currency = None
        for token in doc:
            if token.like_num:
                try:
                    numbers.append(float(token.text))
                except ValueError:
                    continue
            # Look for currency indicators
            if token.text in ["rupiah", "idr", "rp", "million", "juta"]:
                currency = token.text

        # Extract timeline
        timeline_words = ["day", "days", "week", "weeks", "month", "months", "year", "years"]
        timeline = None
        timeline_unit = None
        
        for i, token in enumerate(doc):
            if token.text in timeline_words:
                # Look for number before the time unit
                for j in range(max(0, i-3), i):
                    if doc[j].like_num:
                        try:
                            timeline = float(doc[j].text)
                            timeline_unit = token.text
                            break
                        except ValueError:
                            continue
        
        if numbers and currency:
            budget_info["budget"] = {
                "amount": numbers[0],
                "currency": currency
            }
            
        if timeline and timeline_unit:
            budget_info["timeline"] = {
                "duration": timeline,
                "unit": timeline_unit
            }
            
        if budget_info:
            structured_data.update(budget_info)

    return structured_data


def is_step_complete(step_data: dict, step: str):
    """
    Checks if all required fields for a step are filled.
    """
    required_fields = {
        "companyDetails": ["customerType", "name", "location", "role"],
        "procurementIntent": ["offerings"],
        "roomPlanning": ["rooms"],
        "scopeComplexity": ["type"],
        "budgetTimeline": ["budget", "timeline"]
    }
    for field in required_fields.get(step, []):
        if field not in step_data:
            return False
    return True


def get_next_step(current_step: str, conversation_data: dict):
    """
    Determines the next step in the conversation flow.
    """
    current_index = FORM_STEPS.index(current_step)
    if current_index + 1 < len(FORM_STEPS):
        return FORM_STEPS[current_index + 1]
    return None


def get_next_missing_field(step_data: dict, step: str):
    """
    Identifies the next missing field for the current step.
    """
    required_fields = {
        "companyDetails": ["customerType", "name", "location", "role"],
        "procurementIntent": ["offerings"],
        "scopeComplexity": ["type"],
        "budgetTimeline": ["budget", "timeline"]
    }
    for field in required_fields.get(step, []):
        if field not in step_data:
            return field
    return None


def generate_followup_question(field: str, conversation_data: dict):
    """
    Uses AI to generate a follow-up question based on missing fields.
    """
    prompt = f"User has provided {conversation_data}. What is a good follow-up question to ask about {field}?"
    response = llm.predict(prompt)
    return response


# Redis Helpers
def get_conversation(conversation_id: str):
    """
    Retrieves conversation data from Redis.
    """
    data = redis_client.get(conversation_id)
    print(data)
    return json.loads(data) if data else {"current_step": FORM_STEPS[0]}


def save_conversation(conversation_id: str, conversation_data: dict):
    """
    Saves conversation data to Redis.
    """
    redis_client.set(conversation_id, json.dumps(conversation_data))


@app.get("/")
async def root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
