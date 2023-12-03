# OpenAI Data Analysis & Guiderails Script

## Introduction
The OpenAI Data Analysis & Guiderails Script is an API-driven framework, meticulously crafted to complement and enhance existing AI systems and analysis workflows, including chatbots and intelligent agents. As a versatile tool, it seamlessly integrates with OpenAI's GPT models, offering a robust platform for advanced data analysis and dynamic conditional completions. This script is pivotal in refining AI-powered text outputs, significantly elevating their quality and contextual relevance, making it a vital asset in diverse professional settings.

Equipped with sophisticated features like sentiment analysis, content classification, and trend detection, it provides deep insights into textual data, crucial for areas like content moderation, where it ensures digital spaces are respectful and safe, and customer support, improving responsiveness and effectiveness. Its prowess in market research lies in dissecting and interpreting voluminous text data to identify trends and gauge public opinion, thereby informing strategic decision-making.

Guardrail's adaptability extends to various applications, from academic research to mental health, ensuring high-quality, contextually apt AI contributions. Bridging the gap between AI's raw power and real-world applications, this API-centric framework is an invaluable addition to existing AI and analysis ecosystems across multiple industries.

## AI Guardrails
AI Guardrails, as implemented in the OpenAI Data Analysis & Guiderails Script, are a set of advanced mechanisms and protocols designed to regulate and oversee the performance of AI systems, particularly in generating and analyzing text. The primary purpose of these guardrails is to ensure that the AI operates within a predefined ethical and quality framework, producing results that are not only accurate but also align with societal norms and values.

## 🌟 Key Features and Capabilities

| Feature                                 | Description |
| --------------------------------------- | ----------- |
| 📊 **Sentiment Analysis**               | Determines the emotional tone behind a body of text, assessing whether the sentiment is positive, negative, or neutral. |
| 🏷️ **Content Classification**          | Categorizes text into predefined genres or types, facilitating targeted content handling and organization. |
| 🔍 **Trend Analysis**                   | Identifies current trends, patterns, or emerging themes in text data, useful for market research and content strategy. |
| ⚙️ **Customizable GPT Model Usage**     | Enables the tailoring of text generation and analysis to specific needs, leveraging various GPT model capabilities. |
| 🧮 **Conditional System**               | Implements conditions based on analysis results, allowing for fine-tuned control and contextual responsiveness in output. |
| 🔗 **API-Driven Integration**           | Designed for easy integration with existing AI systems, enhancing chatbots, intelligent agents, and automated workflows. |
| ⏱️ **Real-Time Data Processing**        | Capable of handling and analyzing data in real-time, providing immediate insights and responses. |
| 🌐 **Multi-Lingual Support**            | Offers the ability to process and analyze text in multiple languages, broadening its applicability. |
| 🚫 **Automated Content Moderation**     | Employs AI to automatically detect and handle inappropriate or sensitive content, ensuring safe digital environments. |
| 🧠 **Psychological Analysis**           | Analyzes text to understand psychological aspects, including emotions, behaviors, and mental states. |
| 🔄 **Feedback and Improvement Mechanisms** | Incorporates user feedback for continuous improvement of the system, adapting to evolving requirements and standards. |

# OpenAI Data Analysis & Guiderails Script Capabilities Overview

The OpenAI Data Analysis & Guiderails Script offers a wide range of capabilities, organized into various groups based on tasks or focus areas. These capabilities are designed to analyze text data comprehensively, offering insights across multiple dimensions. The `/analysis_types` endpoint provides a complete list of these functions, with an extensive array of over 50 different analysis types.

## 📊 Text Analysis
- **Sentiment Analysis**: Determine the sentiment (positive, negative, or neutral) and provide a confidence score.
- **Text Summarization**: Condense text, capturing key points and main ideas.
- **Topic Extraction**: Identify and extract main topics or themes.
- **Emotion Detection**: Detect primary emotions and their intensity.
- **Keyword Extraction**: Extract central words or phrases.
- **Content Classification**: Categorize text into predefined genres or categories.
- **Trend Analysis**: Identify current trends, patterns, or emerging themes.
- **Grammatical Error Check**: Identify and suggest corrections for grammatical errors.

## 🌍 Language & Cultural Analysis
- **Language Translation**: Translate text into specified target languages.
- **Cultural Trend Analysis**: Gain insights into cultural trends and public sentiment.
- **Cultural Analysis**: Understand societal norms, values, and practices.
- **Sarcasm and Irony Detection**: Differentiate between literal and figurative language.

## 🏛️ Political & Legal Analysis
- **Political Bias Detection**: Detect political bias and ideological leanings.
- **Fake News Detection**: Assess the likelihood of fake news.
- **Legal Document Analysis**: Interpret legal language and document implications.

## 🧠 Psychological & Behavioral Analysis
- **Psychological Analysis**: Understand emotions, behaviors, and mental states.
- **Behavioral Analysis**: Analyze described or implied behaviors and motivations.
- **Emotional Intelligence Analysis**: Evaluate aspects of emotional intelligence.
- **Cognitive Bias Identification**: Detect cognitive biases and skewed perspectives.
- **Addiction Tendency Analysis**: Evaluate signs of addictive behaviors.

## 🔄 Relationship & Conflict Analysis
- **Relationship Analysis**: Insights into relationships and social dynamics.
- **Conflict Resolution Analysis**: Understand conflict resolution strategies.
- **Conflict Tendency Analysis**: Analyze conflict triggers and patterns.

## 📈 Market & Brand Analysis
- **Market Research Analysis**: Insights into market trends and consumer preferences.
- **Brand Sentiment Analysis**: Assess public sentiment towards brands.
- **Product Review Analysis**: Analyze customer opinions and satisfaction in reviews.
- **Customer Feedback Analysis**: Determine overall satisfaction from customer feedback.

## 🎓 Educational & Learning Analysis
- **Learning Style Identification**: Identify preferred learning styles.
- **Educational Content Analysis**: Analyze educational materials and approaches.

## 🌐 Social Media & Communication Analysis
- **Social Media Monitoring**: Monitor and analyze social media content.
- **Dialogue Analysis**: Understand character interactions and conversational dynamics.

## 🧬 Health & Wellness Analysis
- **Health and Wellness Analysis**: Evaluate health-related content, including medical conditions.
- **Stress Level Detection**: Assess stress levels and identify triggers.
- **Sleep Quality Assessment**: Analyze mentions of sleep quality and issues.
- **Psychosomatic Symptom Analysis**: Assess psychosomatic symptoms and causes.

## 📝 Literary & Historical Analysis
- **Literary Analysis**: Examine literary aspects like themes and narrative style.
- **Historical Data Analysis**: Analyze historical events and patterns.

## 🖥️ User Experience & Feedback
- **User Experience Feedback Analysis**: Evaluate user experience feedback, identifying usability issues.

## 🧘‍♀️ Mindfulness & Therapy Analysis
- **Automated Therapy Session Analysis**: Analyze therapy session transcripts.
- **Mindfulness Meditation Effectiveness**: Analyze mindfulness and meditation techniques.

## 📈 Advanced Analytical Functions
- **Motivational Analysis**: Examine motivational messages and impact.
- **Therapeutic Intervention Analysis**: Evaluate relevance and effectiveness of therapeutic interventions.

Please note that the script includes over 50 distinct analysis functions. For the complete list and detailed descriptions of each type, users can query the `/analysis_types` endpoint.

## Detailed Overview of the Conditional System

The conditional system in the OpenAI Data Analysis & Guiderails Script is designed to execute complex analyses with conditions applied to the results. This system allows for greater control and specificity in handling AI-generated data.

### Simple and Advanced Conditions

- **Simple Conditions**: Basic conditions involve checking a single key-value pair in the analysis results. Example: Check if the sentiment analysis's confidence score is above a certain threshold.
- **Advanced Conditions**: These involve more intricate checks, possibly over multiple keys or complex data structures. Example: Verifying that relevance scores in topic extraction meet certain criteria.

### Sample JSON Formatted Conditions

#### Request Format
```json
{
  "request_data": {
    "analysis_type": "sentiment_analysis",
    "messages": [
      {"role": "user", "content": "I am feeling great today!"},
      {"role": "user", "content": "The weather is sunny and pleasant."}
    ],
    "token_limit": 1000,
    "top_p": 0.1,
    "temperature": 0
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
```
Response Format
```json
{
  "analysis": "All conditions met",
  "details": {
    "condition_responses": [
      {
        "condition": {
          "analysis_type": "sentiment_analysis",
          "key": "confidence_score",
          "threshold": 0.5,
          "condition_type": "greater"
        },
        "result": "Condition met",
        "total_tokens_used": 155,
        "retries": 0,
        "final_openai_response": {
          "id": "chatcmpl-8RoZaFxsQ7osZUEGcZvHPipPk0EMy",
          "object": "chat.completion",
          "created": 1701639950,
          "model": "gpt-4-1106-preview",
          "choices": [
            {
              "index": 0,
              "message": {
                "role": "assistant",
                "content": "{\n  \"sentiment\": \"positive\",\n  \"confidence_score\": 0.95,\n  \"text_snippets\": [\"feeling great\", \"sunny and pleasant\"]\n}"
              },
              "finish_reason": "stop"
            }
          ],
          "usage": {
            "prompt_tokens": 118,
            "completion_tokens": 37,
            "total_tokens": 155
          },
          "system_fingerprint": "fp_a24b4d720c"
        }
      },
      {
        "condition": {
          "analysis_type": "topic_extraction",
          "key": "relevance_scores",
          "threshold": 0.1,
          "condition_type": "greater"
        },
        "result": "Condition met",
        "total_tokens_used": 157,
        "retries": 0,
        "final_openai_response": {
          "id": "chatcmpl-8RoZcw4UY9U19tGuita6CEANw0Cq6",
          "object": "chat.completion",
          "created": 1701639952,
          "model": "gpt-4-1106-preview",
          "choices": [
            {
              "index": 0,
              "message": {
                "role": "assistant",
                "content": "{\n  \"topics\": [\"Emotions\", \"Weather\"],\n  \"relevance_scores\": [0.9, 0.8],\n  \"key_phrases\": [\"feeling great\", \"sunny and pleasant\"]\n}"
              },
              "finish_reason": "stop"
            }
          ],
          "usage": {
            "prompt_tokens": 111,
            "completion_tokens": 46,
            "total_tokens": 157
          },
          "system_fingerprint": "fp_a24b4d720c"
        }
      }
    ]
  },
  "error": null,
  "raw_openai_response": null
}

```
## Understanding the Sample Response

- **Code 200**: Indicates successful execution of the request.
- **Analysis Summary**: "All conditions met" signifies that both conditions (sentiment analysis and topic extraction) passed their respective checks.
- **Details**: The `condition_responses` array contains individual assessments for each condition, including the total tokens used, retries, and the final OpenAI response details.
- **Final OpenAI Response**: Provides the raw response from the OpenAI API, including the model used, tokens details, and the content analyzed.


## Installation
### Requirements
- Python 3.8 or higher
- FastAPI
- Uvicorn
- Pydantic
- Requests

### How to Install
1. Clone the repository to your local machine.
2. Install required Python packages using `pip install -r requirements.txt`.
3. Set environment variables `AUTH_TOKEN` and `OPENAI_API_KEY`.

## How to Use
1. Start the server using `uvicorn main:app --reload`.
2. Access the FastAPI interface at `http://127.0.0.1:8000/docs`.
3. Use the provided endpoints to send requests for data analysis or conditional completions.

## Code Structure
- `main.py`: Contains the FastAPI app initialization, endpoint definitions, and business logic.
- `prompts.py`: Includes predefined prompts and JSON schemas for different types of analysis.
- `security.py`: Manages authentication and security features.

## Contributing
To contribute, please fork the repository, make changes, and submit a pull request. Contributions are welcome to improve the script's functionality and efficiency.

---

Version: v0.0.1  
Author: @rUv
