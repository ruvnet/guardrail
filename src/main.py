#              - OpenAI Data Anaylsis & Guiderail Script
#     /\__/\   - main.py
#    ( o.o  )  - v0.0.1
#      >^<     - by @rUv

# Standard library imports
import os
import re
import json
import logging
from typing import List, Literal, Optional, Union, Dict, Any
from urllib.parse import unquote

# FastAPI and related imports
from fastapi import FastAPI, HTTPException, Depends, Query, Body, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse, JSONResponse
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from typing import Union

# Pydantic imports
from pydantic import BaseModel, Field

# External library imports
import requests
import asyncio
import httpx

# Local module imports
import prompts  # Import the prompts module

security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    auth_token = os.getenv("AUTH_TOKEN")
    if credentials.scheme != "Bearer":
        raise HTTPException(status_code=403, detail="Invalid authentication scheme.")
    if credentials.credentials != auth_token:
        raise HTTPException(status_code=403, detail="Invalid token.")
    return credentials.credentials

# FastAPI app initialization
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Deveopment and Production Server Configuration
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="OpenAI Data Analysis & Guiderails Script",
        version="0.0.1",
        description="This API provides advanced data analysis and conditional completions using OpenAI's GPT models. Key features include sentiment analysis, content classification, trend analysis, and the ability to specify different GPT models for tailored text generation and analysis. The script ensures high-quality, context-sensitive AI-generated text, making it ideal for various applications such as content moderation, customer support, and market research.",
        routes=app.routes,
    )

    # Define the development server URL
    dev_url = "yourURL.com"

    # Get the server URL from the environment variable
    server_url = os.getenv("SERVER_URL")

    # Check if SERVER_URL is None, empty, or the string "None", and set to dev_url if it is
    if not server_url or not server_url.strip() or server_url == "None":
        server_url = dev_url
        url_description = "Development server"
    else:
        url_description = "Production server"

    openapi_schema["servers"] = [{
        "url": server_url,
        "description": url_description
    }]

    app.openapi_schema = openapi_schema
    return app.openapi_schema

# Associate the custom OpenAPI function with the FastAPI app
app.openapi = custom_openapi

# OpenAI API base URL
OPENAI_API_BASE_URL = "https://api.openai.com/v1"

# Define the AnalysisResult model
class AnalysisResult(BaseModel):
  analysis: str
  details: Dict[str, Any]
  error: Optional[str]
  raw_openai_response: Optional[Dict[str, Any]] = None  # Add this line

# Model for a message within a completion request
class Message(BaseModel):
    role: str
    content: str

# Model for a completion request
class CompletionRequest(BaseModel):
    model: str = "gpt-3.5-turbo-1106"
    temperature: float = 0
    max_tokens: int = 1000
    top_p: float = 0.1
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    messages: List[Message]
    n: int = 1
    stream: bool = False

# Function to call OpenAI API
async def call_openai_api(endpoint: str, data: dict):
    openai_api_key = os.getenv("OPENAI_API_KEY")  # Retrieve API key from environment variable
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }
    url = f"{OPENAI_API_BASE_URL}/{endpoint}"
    response = None

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=data)
            response.raise_for_status()
            return {"response": response.json(), "raw_response": response.text, "error": None, "status_code": response.status_code}
        except httpx.HTTPError as http_err:
            # Ensure raw response is returned even in case of HTTP errors
            raw_response = response.text if response else None
            return {"response": response.json() if response else None, "raw_response": raw_response, "error": str(http_err), "status_code": response.status_code if response else 500}
        except Exception as exc:
            # Catch any other exceptions and return the error with the raw response if available
            raw_response = response.text if response else None
            return {"response": response.json() if response else None, "raw_response": raw_response, "error": str(exc), "status_code": 500}

# Endpoint for creating completions
@app.post("/proxy_openai_api/completions/", tags=["Chat"])
async def proxy_openai_api_completions(completion_request: CompletionRequest, token: str = Depends(get_current_user)):
    return await call_openai_api("chat/completions", completion_request.dict())

# Condition class for specifying individual conditions in the analysis
class Condition(BaseModel):
    analysis_type: str
    key: str
    threshold: Optional[Union[float, str]] = None  # Making threshold optional
    condition_type: Literal['greater', 'less', 'equal', 'contains', 'exists', 
                           'is_type', 'length_greater', 'length_less', 'length_equal', 
                           'nested_contains', 'regex_match', 'key_value_pair']

# Advanced Analysis
# Model for an analysis request
class AnalysisRequest(BaseModel):
    analysis_type: str = Field(default="sentiment_analysis", example="sentiment_analysis")
    messages: List[Message] = Field(default=[{"role": "user", "content": "I feel incredibly happy and content today!"}], example=[{"role": "user", "content": "I feel incredibly happy and content today!"}])
    token_limit: int = Field(default=1000, example=1000)
    top_p: float = Field(default=0.9, example=0.1)
    temperature: float = Field(default=0.0, example=0.0)


# List of analysis types
analysis_types = [
    "sentiment_analysis",
    "text_summarization",
    "topic_extraction",
    "emotion_detection",
    "language_translation",
    "grammatical_error_check",
    "keyword_extraction",
    "content_classification",
    "trend_analysis",
    "customer_feedback_analysis",
    "brand_sentiment_analysis",
    "product_review_analysis",
    "market_research_analysis",
    "political_bias_detection",
    "fake_news_detection",
    "cultural_trend_analysis",
    "historical_data_analysis",
    "literary_analysis",
    "scientific_research_analysis",
    "social_media_monitoring",
    "psychological_analysis",
    "criminal_intent_detection",
    "behavioral_analysis",
    "relationship_analysis",
    "emotional_intelligence_analysis",
    "ideological_alignment_detection",
    "conflict_resolution_analysis",
    "narrative_analysis",
    "ethical_stance_detection",
    "propaganda_identification",
    "socioeconomic_status_analysis",
    "health_and_wellness_analysis",
    "sarcasm_and_irony_detection",
    "crisis_detection_analysis",
    "cognitive_bias_identification",
    "dialogue_analysis",
    "legal_document_analysis",
    "cultural_analysis",
    "user_experience_feedback_analysis",
    "automated_therapy_session_analysis",
    "stress_level_detection",
    "mood_detection",
    "personality_type_analysis",
    "cognitive_load_measurement",
    "therapeutic_intervention_analysis",
    "empathy_level_assessment",
    "conflict_tendency_analysis",
    "motivational_analysis",
    "mindfulness_meditation_effectiveness",
    "psychological_resilience_assessment",
    "addiction_tendency_analysis",
    "depression_anxiety_detection",
    "self_esteem_assessment",
    "trauma_analysis",
    "life_satisfaction_analysis",
    "sleep_quality_assessment",
    "psychosomatic_symptom_analysis",
    "learning_style_identification",
    "interpersonal_relationship_analysis",
    "cultural_adaptation_analysis"
]

# Dictionary of analysis types with example requests
analysis_examples = {
  analysis_type: {
      "description": f"{analysis_type.replace('_', ' ').capitalize()}",
      "example_request": {
          "analysis_type": analysis_type,
          "messages": [
              {"role": "system", "content": f"You are a helpful assistant capable of {analysis_type.replace('_', ' ')}."},
              {"role": "user", "content": f"Please perform {analysis_type.replace('_', ' ')} on the following text."}
          ]
      }
  }
  for analysis_type in analysis_types
}

# Endpoint for performing analysis
@app.post("/analysis/", response_model=AnalysisResult)
async def perform_analysis(request_data: AnalysisRequest, token: str = Depends(get_current_user)):
    # Special case handling (if you have any specific analysis type like 'message_length')
    if request_data.analysis_type == 'message_length':
        return AnalysisResult(
            analysis="message_length",
            details={"length": len(request_data.messages[0].content)},
            error=None
        )

    # Construct the system message
    system_message = {
        "role": "system",
        "content": prompts.get_system_prompt(request_data.analysis_type)
    }
    messages = [system_message] + [message.dict() for message in request_data.messages]

    # Prepare the payload for the OpenAI API call
    payload = {
        "model": "gpt-3.5-turbo-1106",
        "response_format": {"type": "json_object"},  # Enable JSON mode
        "messages": messages,
        "max_tokens": request_data.token_limit,
        "temperature": request_data.temperature,
        "top_p": request_data.top_p
    }

    # Call the OpenAI API and get both parsed and raw responses
    openai_response_data = await call_openai_api("chat/completions", payload)
    openai_response = openai_response_data.get("response")
    openai_raw_response = openai_response_data.get("raw_response")
    openai_error = openai_response_data.get("error")

    # Check if the response contains the expected data
    if openai_response and 'choices' in openai_response and openai_response['choices']:
        last_message_content = openai_response['choices'][0].get('message', {}).get('content', '')
        try:
            analysis_result = json.loads(last_message_content)
            if isinstance(analysis_result, dict):
                return AnalysisResult(analysis="Successful analysis", details=analysis_result, error=None, raw_openai_response=openai_response)
        except json.JSONDecodeError:
            return AnalysisResult(analysis="Error in processing", details={}, error="Failed to parse the analysis result as JSON.", raw_openai_response=openai_raw_response)
    else:
        # If there is no valid response, return the error and the raw response
        return AnalysisResult(analysis="Error in response format", details={}, error=openai_error or "Invalid response format from OpenAI API.", raw_openai_response=openai_raw_response)


class AnalysisTypeExample(BaseModel):
    description: str
    example_request: Dict[str, Any]

class AnalysisTypesResponse(BaseModel):
  analysis_types: Dict[str, AnalysisTypeExample]

# Endpoint to list all analysis types with example JSON requests
class AnalysisTypeDetail(BaseModel):
  type: str
  json_schema: dict
  example_request: dict  

class AnalysisTypesResponse(BaseModel):
  analysis_types: dict[str, AnalysisTypeDetail]

@app.get("/analysis_types", response_model=AnalysisTypesResponse)
async def get_analysis_types(query: Optional[str] = Query(None, description="Keyword for fuzzy search")):
    def create_example_request(analysis_type):
        return {
            "analysis_type": analysis_type,
            "messages": [
                {"role": "user", "content": "Sample content for " + analysis_type.replace('_', ' ')}
            ]
        }

    # Filter and create analysis types based on the query
    filtered_analysis_types = {
        analysis_type: AnalysisTypeDetail(
            type=analysis_type,
            json_schema=prompts.JSON_SCHEMAS.get(analysis_type, {}),
            example_request=create_example_request(analysis_type)
        )
        for analysis_type in prompts.ANALYSIS_TYPES
        if not query or query.lower() in analysis_type.lower()
    }

    return AnalysisTypesResponse(analysis_types=filtered_analysis_types)

# Helper function to suggest correct key-value pair using the existing call_openai_api function
async def suggest_correct_key(data: dict):
    prompt = f"Based on the following JSON structure, what should be the correct key and value format? JSON: {json.dumps(data)}"
    suggestion_request = {
        "model": "gpt-3.5-turbo-1106",  # Using a more comprehensive model for suggestions
        "prompt": prompt,
        "max_tokens": 500
    }
    return await call_openai_api("completions", suggestion_request)

# Conditional_analysis function with retry logic, feedback loop, and final response output
# Additional import for concurrent execution
from asyncio import gather

# Updated CombinedRequest model to handle multiple conditions
class CombinedRequest(BaseModel):
  request_data: AnalysisRequest
  conditions: List[Condition]

  class Config:
      schema_extra = {
          "example": {
              "request_data": {
                  "analysis_type": "sentiment_analysis",
                  "messages": [
                      {"role": "user", "content": "I am feeling great today!"},
                      {"role": "user", "content": "The weather is sunny and pleasant."}
                  ],
                  "token_limit": 1000,
                  "top_p": 0.9,
                  "temperature": 0.0
              },
              "conditions": [
                  {
                      "analysis_type": "sentiment_analysis",
                      "key": "confidence_score",
                      "threshold": 0.5,
                      "condition_type": "greater"
                  },
                  {
                      "analysis_type": "topic_extraction",
                      "key": "relevance_scores",
                      "threshold": 0.1,
                      "condition_type": "greater"
                  }
              ]
          }
      }


async def perform_analysis_based_on_type(request_data, analysis_type):
  """
  Perform analysis based on the specified analysis type using the existing /analysis endpoint.

  :param request_data: The data to be analyzed.
  :param analysis_type: The type of analysis to perform (e.g., 'sentiment_analysis', 'topic_extraction').
  :return: The result of the analysis.
  """
  # Update the analysis_type in the request_data
  request_data.analysis_type = analysis_type

  # Call the existing /analysis endpoint
  response = await perform_analysis(request_data)

  # Return the response from the /analysis endpoint
  return response

# Conditional_analysis endpoint
@app.post("/conditional_analysis/", response_model=AnalysisResult, tags=["Conditional Analysis"])
async def conditional_analysis(combined_request: CombinedRequest, token: str = Depends(get_current_user)):
    """
    Perform conditional analysis with multiple conditions. This endpoint analyzes the given request data and checks whether it meets all specified conditions.
    """
    request_data = combined_request.request_data
    conditions = combined_request.conditions

    analysis_tasks = [perform_analysis_based_on_type(request_data, condition.analysis_type) for condition in conditions]
    analysis_results = await asyncio.gather(*analysis_tasks)

    condition_responses = []
    for condition, analysis_result in zip(conditions, analysis_results):
        condition_met = check_condition(analysis_result, condition)
        condition_responses.append({
            "condition": condition,
            "result": "Condition met" if condition_met else "Condition not met",
            "details": analysis_result.details,
            "error": analysis_result.error
        })

    all_conditions_met = all(resp["result"] == "Condition met" for resp in condition_responses)
    analysis_summary = "All conditions met" if all_conditions_met else "One or more conditions not met"

    return AnalysisResult(
        analysis=analysis_summary,
        details={"condition_responses": condition_responses},
        error=None if all_conditions_met else "Conditions failed"
    )

# Function to perform analysis based on type
async def perform_analysis_based_on_type(request_data, analysis_type):
    """
    Perform analysis based on the specified analysis type using the existing /analysis endpoint.
    """
    request_data.analysis_type = analysis_type
    return await perform_analysis(request_data)

# Helper function to extract value from nested JSON data
def extract_value(data, key):
    keys = key.split('.')
    for k in keys:
        if '[' in k and ']' in k:  # Handle array indices
            array_key, index = k[:-1].split('[')
            if index == '*':  # Handle wildcard
                # Extract all items for a wildcard
                return [sub_item for sub_item in data.get(array_key, [])]
            else:
                # Extract a specific item by index
                data = data.get(array_key, [])[int(index)]
        else:
            # Proceed to the next key if not an array
            data = data.get(k, {})
    return data

# Function to check if a condition is met
def check_condition(analysis_result, condition):
    """
    Check if a given condition is met based on the analysis result.

    Args:
        analysis_result (AnalysisResult): The result of the analysis.
        condition (Condition): The condition to check against the analysis result.

    Returns:
        bool: True if the condition is met, False otherwise.
    """
    result_value = extract_value(analysis_result.details, condition.key)

    if result_value is None:
        return False

    try:
        if condition.condition_type == 'greater':
            if isinstance(result_value, list):
                return all(float(item) > float(condition.threshold) for item in result_value)
            return float(result_value) > float(condition.threshold)

        elif condition.condition_type == 'less':
            if isinstance(result_value, list):
                return all(float(item) < float(condition.threshold) for item in result_value)
            return float(result_value) < float(condition.threshold)

        elif condition.condition_type == 'equal':
            if isinstance(result_value, dict):
                return any(str(v) == str(condition.threshold) for v in result_value.values())
            if isinstance(result_value, list):
                return any(str(item) == str(condition.threshold) for item in result_value)
            return str(result_value) == str(condition.threshold)

        elif condition.condition_type == 'contains':
            if isinstance(result_value, list):
                return any(str(condition.threshold) in str(item) for item in result_value)
            return str(condition.threshold) in str(result_value)

        elif condition.condition_type == 'exists':
            return result_value is not None

        elif condition.condition_type == 'is_type':
            expected_type = getattr(__builtins__, condition.threshold, None)
            return isinstance(result_value, expected_type)

        elif condition.condition_type in ['length_greater', 'length_less', 'length_equal']:
            if hasattr(result_value, '__len__'):
                length = len(result_value)
                threshold = int(condition.threshold)
                if condition.condition_type == 'length_greater':
                    return length > threshold
                elif condition.condition_type == 'length_less':
                    return length < threshold
                elif condition.condition_type == 'length_equal':
                    return length == threshold

        elif condition.condition_type == 'regex_match':
            return bool(re.match(condition.threshold, str(result_value), re.DOTALL | re.MULTILINE))

        elif condition.condition_type == 'key_value_pair':
            if isinstance(result_value, dict):
                key, val = condition.threshold.split(':', 1)
                return str(result_value.get(key, "")) == val

    except (TypeError, ValueError, IndexError):
        return False

    return False

# Main function to run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
