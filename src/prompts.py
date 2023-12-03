#              - OpenAI Data Anaylsis & Guiderails Script
#     /\__/\   - prompts.py
#    ( o.o  )  - v0.0.1
#      >^<     - by @rUv

ANALYSIS_TYPES = {
    "sentiment_analysis": "Analyze the sentiment of the provided text. Determine whether the sentiment is positive, negative, or neutral and provide a confidence score.",
    "text_summarization": "Summarize the provided text into a concise version, capturing the key points and main ideas.",
    "topic_extraction": "Identify and extract the main topics or themes from the provided text.",
    "emotion_detection": "Detect and identify the primary emotions expressed in the provided text. Provide a score for the intensity of the detected emotion.",
    "language_translation": "Translate the provided text into a specified target language, maintaining the original meaning and context.",
    "grammatical_error_check": "Check the provided text for any grammatical errors, and suggest corrections where necessary.",
    "keyword_extraction": "Extract key words or phrases from the provided text that are central to its meaning.",
    "content_classification": "Classify the provided text into predefined categories or genres based on its content.",
    "trend_analysis": "Analyze the provided text to identify current trends, patterns, or emerging themes.",
    "customer_feedback_analysis": "Analyze customer feedback to determine overall satisfaction, pinpointing areas of strength and those needing improvement.",
    "brand_sentiment_analysis": "Assess public sentiment towards a specific brand based on the provided text, identifying positive, negative, or neutral attitudes.",
    "product_review_analysis": "Analyze product reviews to extract overall customer opinions, satisfaction levels, and key feedback points.",
    "market_research_analysis": "Examine the provided text for insights into market trends, consumer preferences, and competitive landscape.",
    "political_bias_detection": "Detect any political bias in the provided text, identifying leaning towards particular ideologies or parties.",
    "fake_news_detection": "Evaluate the provided text to determine the likelihood of it being fake news, assessing credibility and factual accuracy.",
    "cultural_trend_analysis": "Analyze the text for insights into cultural trends, shifts, and general public sentiment on cultural topics.",
    "historical_data_analysis": "Analyze historical data presented in the text, extracting key events, trends, and patterns.",
    "literary_analysis": "Perform an analysis of literary aspects in the provided text, examining themes, motifs, character development, and narrative style.",
    "scientific_research_analysis": "Analyze scientific content in the text, summarizing research findings, methodology, and conclusions.",
    "social_media_monitoring": "Monitor and analyze content from social media platforms in the provided text, identifying trends, sentiments, and influential discussions.",
    "psychological_analysis": "Analyze the text for psychological insights, understanding emotions, behaviors, and mental states.",
    "criminal_intent_detection": "Assess the text for indications of criminal intent or illicit activities.",
    "behavioral_analysis": "Analyze behaviors described or implied in the text, understanding patterns and motivations.",
    "relationship_analysis": "Examine the text for insights into relationships, interactions, and social dynamics.",
    "emotional_intelligence_analysis": "Evaluate the text for aspects of emotional intelligence, such as empathy, self-awareness, and social skills.",
    "ideological_alignment_detection": "Detect ideological alignment in the text, identifying adherence to specific sets of beliefs or principles.",
    "conflict_resolution_analysis": "Analyze the text for conflict resolution strategies, understanding approaches to solving disputes or disagreements.",
    "narrative_analysis": "Analyze the narrative structure and elements in the text, understanding plot development, storytelling techniques, and thematic elements.",
    "ethical_stance_detection": "Evaluate the text for ethical stances and moral viewpoints.",
    "propaganda_identification": "Identify propaganda techniques and messages within the text, assessing bias and persuasive strategies.",
    "socioeconomic_status_analysis": "Analyze the text for indications of socioeconomic status, understanding economic and social factors.",
    "health_and_wellness_analysis": "Evaluate the text for health and wellness-related content, understanding medical conditions, treatments, and lifestyle factors.",
    "sarcasm_and_irony_detection": "Detect sarcasm and irony in the text, differentiating between literal and figurative language.",
    "crisis_detection_analysis": "Identify and analyze signs of crisis or emergency situations in the text.",
    "cognitive_bias_identification": "Detect cognitive biases in the text, understanding prejudiced thinking or skewed perspectives.",
    "dialogue_analysis": "Analyze dialogues in the text, understanding character interactions, conversational dynamics, and communication styles.",
    "legal_document_analysis": "Examine legal documents in the text, interpreting legal language, clauses, and implications.",
    "cultural_analysis": "Analyze the text for cultural insights, understanding societal norms, values, and practices.",
    "user_experience_feedback_analysis": "Evaluate user experience feedback in the text, identifying usability issues, satisfaction levels, and user preferences.",
    "automated_therapy_session_analysis": "Analyze transcripts of therapy sessions, understanding therapeutic techniques, patient responses, and progress indicators.",
    "stress_level_detection": "Analyze the text to assess stress levels, identifying triggers and intensity of stress.",
    "mood_detection": "Detect the mood of the individual based on textual cues, ranging from happy to sad, calm to angry.",
    "personality_type_analysis": "Analyze text to determine the personality type of the individual, based on standard personality traits.",
    "cognitive_load_measurement": "Measure the cognitive load or mental effort required in the text context.",
    "therapeutic_intervention_analysis": "Analyze text for therapeutic interventions, evaluating their relevance and effectiveness.",
    "empathy_level_assessment": "Assess the level of empathy expressed in the text, identifying empathetic responses and tendencies.",
    "conflict_tendency_analysis": "Analyze text for conflict tendencies, understanding triggers and patterns of conflict.",
    "motivational_analysis": "Examine the text for motivational messages and their impact.",
    "mindfulness_meditation_effectiveness": "Analyze the effectiveness of mindfulness and meditation techniques mentioned in the text.",
    "psychological_resilience_assessment": "Assess the level of psychological resilience presented in the text.",
    "addiction_tendency_analysis": "Evaluate the text for signs of addictive behavior and tendencies.",
    "depression_anxiety_detection": "Detect signs of depression and anxiety in the text, noting severity and contextual factors.",
    "self_esteem_assessment": "Assess the level of self-esteem expressed in the text.",
    "trauma_analysis": "Analyze the text for references to trauma and its psychological impact.",
    "life_satisfaction_analysis": "Evaluate the text for expressions of life satisfaction and fulfillment.",
    "sleep_quality_assessment": "Analyze text for mentions of sleep quality and related issues.",
    "psychosomatic_symptom_analysis": "Assess text for psychosomatic symptoms and their psychological underpinnings.",
    "learning_style_identification": "Identify preferred learning styles mentioned or implied in the text.",
    "interpersonal_relationship_analysis": "Analyze the text for insights into interpersonal relationships and dynamics.",
    "cultural_adaptation_analysis": "Evaluate how individuals or groups adapt to different cultural contexts in the text."
}
# Define a standard template for prompts
STANDARD_PROMPT_TEMPLATE = "You are a data analysis assistant capable of {analysis_type} analysis. {specific_instruction} Respond with your analysis in JSON format. The JSON schema should include '{json_schema}'."

JSON_SCHEMAS = {
     "sentiment_analysis": {
         "sentiment": "string (positive, negative, neutral)",
         "confidence_score": "number (0-1)",
         "text_snippets": "array of strings (specific text portions contributing to sentiment)"
     },
     "text_summarization": {
         "summary": "string",
         "key_points": "array of strings (main points summarized)",
         "length": "number (number of words in summary)"
     },
     "topic_extraction": {
         "topics": "array of strings",
         "relevance_scores": "array of numbers (0-1) (relevance of each topic)",
         "key_phrases": "array of strings (phrases most associated with each topic)"
     },
     "emotion_detection": {
         "emotion": "string (primary emotion detected)",
         "confidence_score": "number (0-1)",
         "secondary_emotions": "array of objects (secondary emotions and their scores)"
     },
     "language_translation": {
         "translated_text": "string",
         "source_language": "string (detected or specified source language)",
         "target_language": "string (language into which text is translated)"
     },
     "grammatical_error_check": {
         "corrected_text": "string",
         "errors": "array of objects (error details including type, position, and suggestions)",
         "total_errors": "number (total count of errors found)"
     },
     "keyword_extraction": {
         "keywords": "array of strings (key phrases or words extracted from the text)",
         "relevance_scores": "array of numbers (0-1) (indicating the relevance of each keyword)",
         "context_snippets": "array of strings (text snippets where each keyword prominently features)",
         "keyword_frequency": "array of numbers (count of occurrences of each keyword in the text)"
     },
     "content_classification": {
         "category": "string",
         "subcategories": "array of strings",
         "confidence_score": "number (0-1)",
         "contextual_details": "array of strings (explanations for classification)"
     },
     "trend_analysis": {
         "trends": "array of objects (each object detailing trend name, relevance score, and description)",
         "emerging_trends": "array of strings (newly identified trends)",
         "trend_lifetime": "array of objects (duration and evolution of each trend)"
     },
     "customer_feedback_analysis": {
         "feedback_summary": "string",
         "sentiment": "string (positive, negative, neutral)",
         "key_feedback_points": "array of strings",
         "customer_satisfaction_index": "number (0-1)"
     },
     "brand_sentiment_analysis": {
         "brand_sentiment": "string (positive, negative, neutral)",
         "confidence_score": "number (0-1)",
         "key_sentiment_drivers": "array of strings",
         "brand_health_index": "number (overall health score of the brand)"
     },
     "product_review_analysis": {
         "review_summary": "string",
         "sentiment": "string (positive, negative, neutral)",
         "product_rating": "number (average rating from reviews)",
         "key_review_topics": "array of strings (main topics mentioned in reviews)"
     },
     "market_research_analysis": {
         "market_trends": "array of objects (trend details including trend name, impact score, and description)",
         "consumer_preferences": "array of objects (preference details including preference type and popularity score)",
         "market_segmentation": "array of objects (segmentation details including segment name and characteristics)"
     },
     "political_bias_detection": {
         "bias": "string (left, right, neutral)",
         "confidence_score": "number (0-1)",
         "bias_indicators": "array of strings (elements indicating bias)",
         "political_alignment_score": "number (quantifying degree of political bias)"
     },
     "fake_news_detection": {
         "credibility": "string (credible, not credible)",
         "confidence_score": "number (0-1)",
         "fact_check_results": "array of objects (details of fact-checking each claim)",
         "reliability_index": "number (overall reliability score)"
     },
     "cultural_trend_analysis": {
         "trends": "array of objects (trend details including trend name, cultural impact score, and description)",
         "cultural_shifts": "array of objects (shift details including shift name, affected areas, and significance)",
         "cultural_health_index": "number (overall health score of cultural aspects)"
     },
     "historical_data_analysis": {
         "key_events": "array of objects (event details including event name, date, and significance)",
         "patterns": "array of objects (pattern details including pattern name, frequency, and implications)",
         "historical_impact_score": "number (quantifying impact of historical events)"
     },
    "literary_analysis": {
        "themes": "array of strings",
        "character_development": "string",
        "narrative_style": "string"
    },
    "scientific_research_analysis": {
        "research_findings": "string",
        "methodology": "string",
        "conclusions": "string"
    },
    "social_media_monitoring": {
        "trending_topics": "array of strings",
        "influential_posts": "array of objects (post details)"
    },
    "psychological_analysis": {
        "emotional_states": "array of strings",
        "behavioral_patterns": "string"
    },
    "criminal_intent_detection": {
        "potential_risks": "array of strings",
        "threat_level": "string"
    },
     "behavioral_analysis": {
         "observed_behaviors": "array of strings",
         "behavioral_patterns": "string",
         "motivations": "string"
     },
     "relationship_analysis": {
         "interaction_types": "array of strings",
         "relationship_dynamics": "string",
         "communication_patterns": "string"
     },
     "emotional_intelligence_analysis": {
         "empathy_levels": "string",
         "self_awareness_assessment": "string",
         "social_skills_evaluation": "string"
     },
     "ideological_alignment_detection": {
         "political_ideologies": "array of strings",
         "alignment_strength": "number (0-1)"
     },
     "conflict_resolution_analysis": {
         "conflict_types": "array of strings",
         "resolution_strategies": "array of strings"
     },
     "narrative_analysis": {
         "plot_structure": "string",
         "character_roles": "array of objects (character details)",
         "thematic_elements": "array of strings"
     },
     "ethical_stance_detection": {
         "ethical_positions": "array of strings",
         "stance_strength": "number (0-1)"
     },
     "propaganda_identification": {
         "propaganda_techniques": "array of strings",
         "persuasive_strength": "number (0-1)"
     },
     "socioeconomic_status_analysis": {
         "economic_indicators": "array of strings",
         "social_factors": "array of strings"
     },
     "health_and_wellness_analysis": {
         "health_conditions_identified": "array of strings",
         "wellness_recommendations": "array of strings"
     },
     "sarcasm_and_irony_detection": {
         "sarcasm_level": "string",
         "irony_type": "string"
     },
     "crisis_detection_analysis": {
         "crisis_signals": "array of strings",
         "urgency_level": "string"
     },
     "cognitive_bias_identification": {
         "biases_identified": "array of strings",
         "bias_impact": "string"
     },
     "dialogue_analysis": {
         "speaking_styles": "array of strings",
         "conversation_themes": "array of strings"
     },
     "legal_document_analysis": {
         "key_clauses": "array of strings",
         "document_legality": "string"
     },
     "cultural_analysis": {
         "cultural_values": "array of strings",
         "societal_norms": "array of strings"
     },
     "user_experience_feedback_analysis": {
         "usability_issues": "array of strings",
         "user_satisfaction_levels": "string"
     },
     "automated_therapy_session_analysis": {
         "therapeutic_techniques_used": "array of strings",
         "patient_response_types": "array of strings"
     },
     "stress_level_detection": {
         "stress_level": "string", 
         "stress_triggers": "array of strings"
     },
     "mood_detection": {
         "mood": "string", 
         "mood_intensity": "number"
     },
     "personality_type_analysis": {
         "personality_type": "string", 
         "trait_scores": "object"
     },
     "cognitive_load_measurement": {
         "cognitive_load_level": "string", 
         "factors_contributing": "array of strings"
     },
     "therapeutic_intervention_analysis": {
         "interventions": "array of strings", 
         "effectiveness": "number"
     },
     "empathy_level_assessment": {
         "empathy_level": "string", 
         "empathetic_responses": "array of strings"
     },
     "conflict_tendency_analysis": {
         "conflict_triggers": "array of strings", 
         "conflict_resolution": "string"
     },
     "motivational_analysis": {
         "motivational_messages": "array of strings", 
         "impact_score": "number"
     },
     "mindfulness_meditation_effectiveness": {
         "techniques_used": "array of strings", 
         "effectiveness_score": "number"
     },
     "psychological_resilience_assessment": {
         "resilience_level": "string", 
         "coping_strategies": "array of strings"
     },
     "addiction_tendency_analysis": {
         "addictive_behaviors": "array of strings", 
         "severity_level": "string"
     },
     "depression_anxiety_detection": {
         "depression_level": "string", 
         "anxiety_level": "string"
     },
     "self_esteem_assessment": {
         "self_esteem_level": "string", 
         "influencing_factors": "array of strings"
     },
     "trauma_analysis": {
         "traumatic_events": "array of strings", 
         "psychological_impact": "string"
     },
     "life_satisfaction_analysis": {
         "satisfaction_level": "string", 
         "key_factors": "array of strings"
     },
     "sleep_quality_assessment": {
         "sleep_quality": "string", 
         "disruptive_factors": "array of strings"
     },
     "psychosomatic_symptom_analysis": {
         "symptoms": "array of strings", 
         "psychological_causes": "array of strings"
     },
     "learning_style_identification": {
         "preferred_styles": "array of strings", 
         "effectiveness": "number"
     },
     "interpersonal_relationship_analysis": {
         "relationship_types": "array of strings", 
         "interaction_patterns": "array of strings"
     },
     "cultural_adaptation_analysis": {
         "adaptation_levels": "array of strings", 
         "challenges_faced": "array of strings"
     }
}

def get_system_prompt(analysis_type: str) -> str:
  # Fetch the specific instruction and JSON schema for the given analysis type
  specific_instruction = ANALYSIS_TYPES.get(analysis_type, "Perform the analysis as per the specified type.")
  json_schema = JSON_SCHEMAS.get(analysis_type, {})

  # Format the JSON schema into a string representation
  json_schema_str = ', '.join([f"'{key}': {value}" for key, value in json_schema.items()])

  # Construct the system prompt using the analysis type, specific instruction, and the formatted JSON schema
  return f"You are a data analysis assistant capable of {analysis_type} analysis. {specific_instruction} Respond with your analysis in JSON format. The JSON schema should include: {{{json_schema_str}}}."
