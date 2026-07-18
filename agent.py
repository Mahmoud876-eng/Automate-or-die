import os
import json

from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder


from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langgraph.graph import StateGraph,  START, END

from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore


app = FastAPI()
load_dotenv()
# -----------------------------------------------------------------------------
# Long-term memory schema
# -----------------------------------------------------------------------------
class LongTermMemory(BaseModel):#TODO: add whatever u like to th elong term memory prefereences fav books 
    user_name: Optional[str] = Field(default=None, description="The name of the user")
# RedisSaver -> checkpointer: stores conversation state/history per thread_id (short-term memory per thread id)
# RedisStore  -> store:       stores arbitrary key/value data per namespace ( long-term memory per user_id)
#                                 
checkpointer = MemorySaver()
store = InMemoryStore()

key=os.getenv("AZURE_AI_INFERENCE_CREDENTIAL")
api= os.getenv("AZURE_AI_INFERENCE_ENDPOINT") 
#here using azure ai foundry  for open source models
llm = AzureAIOpenAIApiChatModel(
        endpoint=api,
        credential=key,
        model="Mistral-Large-3",
    )

gllm = ChatGoogleGenerativeAI(
    model="gemma-4-26b-a4b-it",
    temperature=0.7,
)
# 
long_term_memory_structured_llm = gllm.with_structured_output(LongTermMemory)
@dataclass
class Context:
    user_id: str
# =============================================================================
# TOOLS
# =============================================================================
from typing import Optional
from pydantic import BaseModel, Field


class InsuranceFileMemory(BaseModel):
    claimant_name: Optional[str] = Field(
        default=None,
        description="Full name of the insured person or claimant"
    )

    policy_number: Optional[str] = Field(
        default=None,
        description="Insurance policy number"
    )

    insurance_type: Optional[str] = Field(
        default=None,
        description="Type of insurance such as auto, health, home, travel, life"
    )

    incident_type: Optional[str] = Field(
        default=None,
        description="Type of incident such as accident, theft, fire, injury, flood"
    )

    incident_date: Optional[str] = Field(
        default=None,
        description="Date when the incident occurred"
    )

    incident_location: Optional[str] = Field(
        default=None,
        description="Location where the incident happened"
    )

    incident_description: Optional[str] = Field(
        default=None,
        description="Detailed description of the incident"
    )

    vehicle_information: Optional[str] = Field(
        default=None,
        description="Vehicle details if applicable including make, model, year, and registration"
    )

    damages: list[str] = Field(
        default_factory=list,
        description="List of reported damages, losses, or affected items"
    )

    injuries: list[str] = Field(
        default_factory=list,
        description="List of injuries reported by involved parties"
    )

    documents_provided: list[str] = Field(
        default_factory=list,
        description="Insurance claim documents already provided"
    )

    missing_information: list[str] = Field(
        default_factory=list,
        description="Information required before the claim can be processed"
    )

    claim_status: Optional[str] = Field(
        default=None,
        description="Current claim status such as draft, submitted, pending, approved, rejected"
    )

    claim_priority: Optional[str] = Field(
        default=None,
        description="Priority level of the claim: low, medium, high, urgent"
    )

    next_actions: list[str] = Field(
        default_factory=list,
        description="Recommended next steps for handling the insurance file"
    )
insurance_memory_structured_llm = gllm.with_structured_output(
    InsuranceFileMemory
)
@tool #TODO: a loop to check the missing data
def fillfile(runtime: ToolRuntime[Context]):
    """
    when an agent want to  generate a document here it s 
    """
    messages = runtime.state.get("messages", [])
    user_id=runtime.context.user_id
    namespace=(user_id,"memories")
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            human_message = m
            break
    
    clean_messages = [
        SystemMessage(content="Extract user preferences from this message."),
        HumanMessage(content=human_message.content)
    ]

    
    response = insurance_memory_structured_llm.invoke(clean_messages)
     
    runtime.store.put( 
        namespace,
        "a-memory",
        jsonable_encoder(response),
    )
@tool
async def redirect_to_recomendation(subject: str, topic: str, runtime: ToolRuntime[Context]) -> Command:
    """Redirect user to quiz generation interface.
    
    IF the user asked you of a quiz u must use this tool and no other tool except it.
    
     IMPORTANT: Before calling this tool, if the user mentions "my favorite subject" or personal preferences,
    you MUST call retrieve_long_term_data FIRST to get their profile information.
    Then use that information to determine the subject/topic.
    
    args:
    subject: the subject of the quiz
    topic: the topic of the quiz. if this is missing choose a random topic in the subject.

    """   
    value= {"action": "redirect", "target": f"/recomendation?message={message}"}
    #return {"messages": [ToolMessage(content=json.dumps(value), tool_call_id=runtime.tool_call_id)]}
    return  Command(
        update={
            "messages": [ToolMessage(content=json.dumps(value), tool_call_id=runtime.tool_call_id)],
            "redirect_target": value["target"]  # State for graph logic
        }
    )

@tool
def store_memory(runtime: ToolRuntime[Context]) -> ToolMessage:
    """SAVE USER PROFILE AND LEARNING PREFERENCES TO MEMORY.
    
     CALL THIS TOOL IMMEDIATELY WHEN the user shares:
    - Name, username, or preferred nickname → Extract this for user_name
    - Learning interests and favorite subjects (science, history, math, arts, etc.) → Add to preferences
    - Communication preferences (formal, casual, playful tone) → Set as tone
    - Learning style preferences (visual examples, step-by-step, stories) → Add to learning_style
    - Goals or areas they want to improve → Add to goals
    - Any persistent profile information for personalization
    
    Examples of when to call:
    "My name is Alex" --> store user_name="Alex"
    "I love science and history" --> store preferences=["science", "history"]
    "I prefer a casual tone" --> store tone="casual"
    "I learn best with examples" --> store learning_style="visual examples"
    
    PURPOSE: Enable personalized, contextual learning experiences across sessions.
    This is CRITICAL for the learning companion experience!
    """
    messages = runtime.state.get("messages", [])
    user_id=runtime.context.user_id
    namespace=(user_id,"memories")
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            human_message = m
            break
    
    clean_messages = [
        SystemMessage(content="Extract user preferences from this message."),
        HumanMessage(content=human_message.content)
    ]

    
    response = long_term_memory_structured_llm.invoke(clean_messages)
     
    runtime.store.put( 
        namespace,
        "a-memory",
        jsonable_encoder(response),
    )
    
    return ToolMessage(
        content="data stored inside the db",
        tool_call_id=runtime.tool_call_id  
    )

@tool
def retrieve_memory(runtime: ToolRuntime[Context]) -> ToolMessage:
    """Retrieve user profile and learning preferences from memory.
    
    Call when user asks about their profile, preferences, or references past information.
    Returns stored user data including name, interests, tone preference, learning style, and goals.
    """
    
    user_id=runtime.context.user_id
    namespace=(user_id,"memories")
    personal_data=runtime.store.get(namespace,"a-memory")
    if personal_data is None:
        content = "No profile data found."
    elif isinstance(personal_data, (dict, list)):
        content = json.dumps(personal_data, ensure_ascii=False)
    else:
        content = str(personal_data)

        return ToolMessage(
            content=content,
            tool_call_id=runtime.tool_call_id  
        )

#subscription automatique 
#cimiste 

sys_prompt="""
You are a helpful, friendly conversational assistant with long-term memory.

You have access to two tools:
1. store_memory(content: str, category: str) — Save an important fact, 
   preference, or detail about the user for future conversations.
2. retrieve_memory(query: str) — Search your stored memories for relevant 
   past information about the user.

## When to use store_memory
Use this whenever the user shares something worth remembering long-term:
- Personal facts (name, job, location, family members)
- Preferences (likes/dislikes, communication style, goals)
- Ongoing projects or ongoing situations they mention
- Anything they explicitly ask you to remember

Do NOT store: small talk, one-off questions, or anything the user says not 
to remember.

## When to use retrieve_memory
Use this:
- At the start of a new conversation topic, to check if you already know 
  something relevant about the user
- When the user references something from "before" ("like I told you", 
  "remember when...", "as usual")
- Before answering personal questions about the user (e.g. "what's my job again?")

## Behavior rules
- Never fabricate a memory. If retrieve_memory returns nothing relevant, 
  say you don't have that on record yet — don't guess.
- Be transparent but not robotic: don't say "I am calling store_memory now" 
  — just naturally acknowledge what you're noting, e.g. "Got it, I'll 
  remember that."
- Keep stored memories short, factual, and in third person 
  (e.g. "User works as a backend developer in Tunis" not "I work as...").
- Always retrieve before you assume — don't rely only on the current 
  conversation's context if the user references the past.
- Stay conversational and natural. The memory system should feel invisible 
  to the user, not like a database lookup.

"""
main_tools=[
                retrieve_memory,
                store_memory,
                redirect_to_recomendation,
                fillfile
                ]

tool_long_term=[
                retrieve_memory,
                store_memory,
                ]
run_agent = create_agent(gllm,system_prompt=SystemMessage(content=sys_prompt), context_schema= Context , tools=tool_long_term, checkpointer=checkpointer, store=store)
@app.post('/chatbot/')#we are gonna ue langraph later here
async def chatbot(user_input: str, thr_id:str, usr_id: str):
    
    response= run_agent.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config={"configurable": {"thread_id": thr_id}},
        context=Context(user_id=usr_id),
    )
    return response

@app.post('/recomendation/')
async def recomendation_system(message: str, thr_id:str, usr_id: str):
    response= run_agent.invoke(
        {"messages": [HumanMessage(content=message)]},
        config={"configurable": {"thread_id": thr_id}},
        context=Context(user_id=usr_id),
    )
    return response
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage

# Assumption:
# - create_agent is already available in your project
# - gllm is your initialized LLM
# - checkpointer, store, tool_long_term are already initialized


# ============================================================
# Context Schema
# ============================================================

class Context(BaseModel):
    user_id: Optional[str] = Field(
        default=None,
        description="Unique user identifier"
    )

    session_id: Optional[str] = Field(
        default=None,
        description="Conversation or session identifier"
    )

    workflow_id: Optional[str] = Field(
        default=None,
        description="Current workflow identifier"
    )

    workflow_stage: Optional[str] = Field(
        default=None,
        description="Current stage in the workflow"
    )

    current_node: Optional[str] = Field(
        default=None,
        description="Current agent/node handling the workflow"
    )

    previous_node: Optional[str] = Field(
        default=None,
        description="Previous agent/node in the workflow"
    )

    intent: Optional[str] = Field(
        default=None,
        description="Detected intent for the current user request"
    )

    intent_confidence: Optional[float] = Field(
        default=None,
        description="Confidence score for the detected intent"
    )

    selected_product_id: Optional[int] = Field(
        default=None,
        description="Currently selected insurance product ID"
    )

    customer_profile: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Known customer profile data"
    )

    risk_profile: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Known customer risk profile data"
    )

    tool_results: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Results collected from tools and services"
    )

    missing_information: Optional[List[str]] = Field(
        default=None,
        description="Information still missing to proceed"
    )

    conversation_summary: Optional[str] = Field(
        default=None,
        description="Short memory summary of the conversation"
    )


# ============================================================
# Structured Memory / JSON-like Schemas
# ============================================================

class InsuranceFileMemory(BaseModel):
    claimant_name: Optional[str] = Field(
        default=None,
        description="Full name of the insured person or claimant"
    )

    policy_number: Optional[str] = Field(
        default=None,
        description="Insurance policy number"
    )

    insurance_type: Optional[str] = Field(
        default=None,
        description="Type of insurance such as auto, health, home, travel, life"
    )

    incident_type: Optional[str] = Field(
        default=None,
        description="Type of incident such as accident, theft, fire, injury, flood"
    )

    incident_date: Optional[str] = Field(
        default=None,
        description="Date when the incident occurred"
    )

    incident_location: Optional[str] = Field(
        default=None,
        description="Location where the incident happened"
    )


insurance_memory_structured_llm = gllm.with_structured_output(
    InsuranceFileMemory
)


class IntentOutput(BaseModel):
    intent: Optional[str] = Field(
        default=None,
        description="Detected user intent such as product_discovery, product_comparison, subscription, policy_review, faq, risk_analysis, document_help, unknown"
    )

    confidence: Optional[float] = Field(
        default=None,
        description="Confidence score between 0 and 1"
    )

    missing_information: Optional[List[str]] = Field(
        default=None,
        description="List of missing information required to proceed"
    )

    reasoning_summary: Optional[str] = Field(
        default=None,
        description="Short explanation of why the intent was chosen"
    )

    suggested_next_node: Optional[str] = Field(
        default=None,
        description="Suggested next node in the workflow"
    )


intent_structured_llm = gllm.with_structured_output(
    IntentOutput
)


class ProductItem(BaseModel):
    product_id: Optional[int] = Field(
        default=None,
        description="Unique product identifier"
    )

    name: Optional[str] = Field(
        default=None,
        description="Insurance product name"
    )

    category: Optional[str] = Field(
        default=None,
        description="Product category such as auto, health, home, life, travel"
    )

    monthly_price: Optional[float] = Field(
        default=None,
        description="Monthly product price"
    )

    annual_price: Optional[float] = Field(
        default=None,
        description="Annual product price"
    )

    coverage_summary: Optional[str] = Field(
        default=None,
        description="Short summary of the insurance coverage"
    )


class ProductOutput(BaseModel):
    products_found: Optional[List[ProductItem]] = Field(
        default=None,
        description="List of products found for the user request"
    )

    comparison_mode: Optional[bool] = Field(
        default=None,
        description="Whether the user request is asking for product comparison"
    )

    filters_used: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Filters used to retrieve products"
    )

    missing_information: Optional[List[str]] = Field(
        default=None,
        description="Missing information required to refine product results"
    )

    summary_for_downstream: Optional[str] = Field(
        default=None,
        description="Short summary for downstream agents"
    )


product_structured_llm = gllm.with_structured_output(
    ProductOutput
)


class CustomerOutput(BaseModel):
    customer_found: Optional[bool] = Field(
        default=None,
        description="Whether the customer profile exists"
    )

    profile_completion_status: Optional[str] = Field(
        default=None,
        description="Profile status such as complete, partial, or missing"
    )

    known_fields: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Known customer profile fields"
    )

    missing_fields: Optional[List[str]] = Field(
        default=None,
        description="Missing customer profile fields"
    )

    summary_for_downstream: Optional[str] = Field(
        default=None,
        description="Short summary for downstream agents"
    )


customer_structured_llm = gllm.with_structured_output(
    CustomerOutput
)


class RecommendationItem(BaseModel):
    product_id: Optional[int] = Field(
        default=None,
        description="Recommended product ID"
    )

    score: Optional[float] = Field(
        default=None,
        description="Recommendation score between 0 and 1"
    )

    reason: Optional[str] = Field(
        default=None,
        description="Reason why this product was recommended"
    )


class RecommendationOutput(BaseModel):
    status: Optional[str] = Field(
        default=None,
        description="Recommendation status such as recommended, needs_more_information, or no_match"
    )

    recommended_products: Optional[List[RecommendationItem]] = Field(
        default=None,
        description="List of recommended products"
    )

    missing_information: Optional[List[str]] = Field(
        default=None,
        description="Missing inputs required for a recommendation"
    )

    summary_for_response: Optional[str] = Field(
        default=None,
        description="Summary prepared for the response agent"
    )


recommendation_structured_llm = gllm.with_structured_output(
    RecommendationOutput
)


class RiskOutput(BaseModel):
    risk_profile_available: Optional[bool] = Field(
        default=None,
        description="Whether a risk profile is available"
    )

    risk_scores: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Risk scores across categories"
    )

    risk_summary: Optional[str] = Field(
        default=None,
        description="Summary interpretation of the risk profile"
    )

    missing_information: Optional[List[str]] = Field(
        default=None,
        description="Missing information required for risk understanding"
    )


risk_structured_llm = gllm.with_structured_output(
    RiskOutput
)


class SubscriptionOutput(BaseModel):
    subscription_stage: Optional[str] = Field(
        default=None,
        description="Current stage in the subscription workflow"
    )

    required_fields: Optional[List[str]] = Field(
        default=None,
        description="Fields required to proceed with subscription"
    )

    collected_fields: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Already collected subscription information"
    )

    missing_fields: Optional[List[str]] = Field(
        default=None,
        description="Still missing fields for subscription"
    )

    ready_for_next_step: Optional[bool] = Field(
        default=None,
        description="Whether the workflow is ready to proceed"
    )

    suggested_next_node: Optional[str] = Field(
        default=None,
        description="Suggested next node after subscription agent"
    )


subscription_structured_llm = gllm.with_structured_output(
    SubscriptionOutput
)


class DocumentOutput(BaseModel):
    documents_used: Optional[List[str]] = Field(
        default=None,
        description="Documents used during analysis"
    )

    document_summary: Optional[str] = Field(
        default=None,
        description="Summary of the analyzed documents"
    )

    key_points: Optional[List[str]] = Field(
        default=None,
        description="Important key points extracted from the document"
    )

    missing_documents: Optional[List[str]] = Field(
        default=None,
        description="Documents that are needed but currently missing"
    )

    summary_for_response: Optional[str] = Field(
        default=None,
        description="Summary prepared for the response agent"
    )


document_structured_llm = gllm.with_structured_output(
    DocumentOutput
)


class ResponseOutput(BaseModel):
    text: Optional[str] = Field(
        default=None,
        description="Final conversational response"
    )

    markdown: Optional[str] = Field(
        default=None,
        description="Markdown formatted response if needed"
    )

    suggestion_chips: Optional[List[str]] = Field(
        default=None,
        description="Suggested next actions for the user"
    )

    ui_payload: Optional[Dict[str, Any]] = Field(
        default=None,
        description="UI payload such as cards, forms, banners, or action hints"
    )


response_structured_llm = gllm.with_structured_output(
    ResponseOutput
)


class ClarificationOutput(BaseModel):
    question: Optional[str] = Field(
        default=None,
        description="Clarifying question to ask the user"
    )

    required_field: Optional[str] = Field(
        default=None,
        description="Missing field targeted by the question"
    )


clarification_structured_llm = gllm.with_structured_output(
    ClarificationOutput
)


# ============================================================
# System Prompts
# ============================================================

shared_rules_block = """
You are a specialized agent inside an AI Insurance Assistant orchestrated by LangGraph.

Global rules:
- You do not own business logic.
- You must use available tools through ToolRegistry-compatible tools only.
- You must not bypass services, repositories, or provider abstractions.
- You must update only your own part of the workflow context.
- You must return concise, structured, workflow-oriented outputs.
- If required information is missing, state exactly what is missing.
- Do not fabricate policy, pricing, customer, risk, or document facts.
- Be deterministic, concise, and easy to route.
"""


orchestrator_sys_prompt = f"""
You are the Orchestrator Agent for an AI Insurance Assistant.

{shared_rules_block}

Your role:
- Sit on top of all specialized agents
- Decide which agent should handle the next step
- Route according to intent, confidence, workflow stage, customer profile completeness, tool results, missing information, and available data
- Keep the workflow efficient and minimal
- Ensure the system stays aligned with the platform architecture

Routing principles:
- If intent confidence is low, route to clarification
- If intent is product_discovery, route to product
- If intent is product_comparison, route to product
- If intent is subscription, route to customer first
- If intent is policy_review, route to customer then document
- If intent is faq, route to response
- If intent is risk_analysis, route to risk
- If intent is document_help, route to document
- If profile is incomplete, route to clarification or customer
- If recommendation is needed and enough context exists, route to recommendation
- When enough information is gathered, route to response

Hard rules:
- Do not perform business actions yourself
- Do not retrieve data directly if a specialized agent should do it
- Do not answer the user directly unless the workflow has reached the final response stage
- Prefer the smallest next step that moves the workflow forward

When deciding routing, think in terms of:
- intent
- confidence
- workflow stage
- missing information
- profile completeness
- tool results
- response readiness
"""


intent_sys_prompt = f"""
You are the Intent Agent.

{shared_rules_block}

Goal:
Classify the user's request into one clear workflow intent.

Allowed intents:
- product_discovery
- product_comparison
- subscription
- policy_review
- faq
- risk_analysis
- document_help
- unknown

Responsibilities:
- Read the latest user message and relevant context
- Detect the most likely intent
- Estimate confidence
- Identify missing information required before routing
- Suggest the next best node

Hard rules:
- Do not answer the user directly
- Do not retrieve products
- Do not generate recommendations
- Do not perform subscription actions
- Do not fabricate facts

Expected structured output fields:
- intent
- confidence
- missing_information
- reasoning_summary
- suggested_next_node
"""


product_sys_prompt = f"""
You are the Product Agent.

{shared_rules_block}

Goal:
Retrieve and organize insurance product information relevant to the current request.

Responsibilities:
- Search, filter, and organize products
- Prepare comparison-ready product summaries
- Surface category, pricing, coverage summary, and basic fit signals
- Help downstream recommendation and response steps

Hard rules:
- Do not recommend unless recommendation is explicitly the next workflow step
- Do not invent product details
- Do not bypass tools/services
- If search criteria are unclear, identify missing information

Expected structured output fields:
- products_found
- comparison_mode
- filters_used
- missing_information
- summary_for_downstream
"""


customer_sys_prompt = f"""
You are the Customer Agent.

{shared_rules_block}

Goal:
Retrieve or complete customer profile context needed by downstream workflows.

Responsibilities:
- Retrieve customer profile information
- Determine profile completeness
- Identify known versus missing profile fields
- Prepare profile context for subscription, recommendation, document, or response steps

Hard rules:
- Do not infer unknown customer data
- Do not ask unnecessary questions
- Do not execute subscription, recommendation, or document analysis yourself

Expected structured output fields:
- customer_found
- profile_completion_status
- known_fields
- missing_fields
- summary_for_downstream
"""


recommendation_sys_prompt = f"""
You are the Recommendation Agent.

{shared_rules_block}

Goal:
Produce a structured recommendation based on available context.

Allowed inputs:
- product candidates
- customer profile
- risk profile
- current workflow stage
- user request context

Responsibilities:
- Match products to the user need
- Explain why a product fits
- Return recommendation candidates for user-facing response generation
- Clearly state when more information is needed

Hard rules:
- Do not fabricate prices, exclusions, coverage, or eligibility
- Do not produce the final user-facing answer
- If context is insufficient, return needs_more_information
- Keep reasoning concise and structured

Expected structured output fields:
- status
- recommended_products
- missing_information
- summary_for_response
"""


risk_sys_prompt = f"""
You are the Risk Agent.

{shared_rules_block}

Goal:
Retrieve and summarize available risk profile information.

Responsibilities:
- Read available risk profile context
- Summarize the current risk profile
- Explain category-level risk at a high level
- Support downstream recommendation or response generation

Hard rules:
- Do not invent risk scores
- Do not perform unsupported actuarial modeling
- If risk information is not available, report that clearly

Expected structured output fields:
- risk_profile_available
- risk_scores
- risk_summary
- missing_information
"""


subscription_sys_prompt = f"""
You are the Subscription Agent.

{shared_rules_block}

Goal:
Drive the digital insurance subscription workflow step by step.

Responsibilities:
- Identify the current subscription stage
- Determine required fields
- Track collected information
- Identify missing fields
- Indicate whether the next step is possible

Hard rules:
- Do not finalize a subscription without required fields
- Do not invent legal or policy details
- Do not generate documents yourself
- Keep the workflow structured and minimal

Expected structured output fields:
- subscription_stage
- required_fields
- collected_fields
- missing_fields
- ready_for_next_step
- suggested_next_node
"""


document_sys_prompt = f"""
You are the Document Agent.

{shared_rules_block}

Goal:
Analyze document-related context for policy review and document workflows.

Responsibilities:
- Summarize document content
- Extract key points relevant to the workflow
- Identify missing documents
- Prepare clean downstream summaries for the response step

Hard rules:
- Do not fabricate document content
- Do not claim a document was generated or signed unless the workflow confirms it
- Do not bypass tools/services
- Keep outputs concise and structured

Expected structured output fields:
- documents_used
- document_summary
- key_points
- missing_documents
- summary_for_response
"""


response_sys_prompt = f"""
You are the Response Agent.

{shared_rules_block}

Goal:
Generate the final user-facing response for the AI Insurance Assistant.

Responsibilities:
- Merge outputs from prior agents
- Produce clear conversational text
- Produce markdown when useful
- Produce suggestion chips
- Produce UI payload hints for the frontend

Hard rules:
- Do not invent unsupported facts
- Reflect uncertainty explicitly when data is incomplete
- Keep responses helpful, concise, and action-oriented
- Respect the current workflow stage and context

Expected structured output fields:
- text
- markdown
- suggestion_chips
- ui_payload
"""


clarification_sys_prompt = f"""
You are the Clarification Agent.

{shared_rules_block}

Goal:
Ask the smallest possible clarifying question when confidence is low or required information is missing.

Responsibilities:
- Ask one focused question at a time
- Prioritize the most valuable missing field
- Keep the question short and user-friendly
- Prepare the workflow to continue after the user's answer

Hard rules:
- Do not ask multiple questions at once
- Do not ask for information already present in context
- Do not perform routing yourself
- Keep the question brief

Expected structured output fields:
- question
- required_field
"""


voice_sys_prompt = f"""
You are the Voice Agent.

{shared_rules_block}

Goal:
Support speech-driven interactions by cleaning and normalizing spoken input and preparing spoken-friendly output when required.

Responsibilities:
- Normalize transcribed speech into clear text
- Preserve the user's intent
- Detect likely transcription ambiguity
- Prepare spoken-friendly wording when needed

Hard rules:
- Do not change business meaning
- If transcription is ambiguous, indicate that clarification is needed
- Do not perform orchestration decisions
- Keep output structured and concise
"""


# ============================================================
# Agent Initialization
# ============================================================

orchestrator_agent = create_agent(
    gllm,
    system_prompt=SystemMessage(content=orchestrator_sys_prompt),
    context_schema=Context,
    tools=tool_long_term,
    checkpointer=checkpointer,
    store=store,
)

intent_agent = create_agent(
    gllm,
    system_prompt=SystemMessage(content=intent_sys_prompt),
    context_schema=Context,
    tools=tool_long_term,
    checkpointer=checkpointer,
    store=store,
)

product_agent = create_agent(
    gllm,
    system_prompt=SystemMessage(content=product_sys_prompt),
    context_schema=Context,
    tools=tool_long_term,
    checkpointer=checkpointer,
    store=store,
)

customer_agent = create_agent(
    gllm,
    system_prompt=SystemMessage(content=customer_sys_prompt),
    context_schema=Context,
    tools=tool_long_term,
    checkpointer=checkpointer,
    store=store,
)

recommendation_agent = create_agent(
    gllm,
    system_prompt=SystemMessage(content=recommendation_sys_prompt),
    context_schema=Context,
    tools=tool_long_term,
    checkpointer=checkpointer,
    store=store,
)

risk_agent = create_agent(
    gllm,
    system_prompt=SystemMessage(content=risk_sys_prompt),
    context_schema=Context,
    tools=tool_long_term,
    checkpointer=checkpointer,
    store=store,
)

subscription_agent = create_agent(
    gllm,
    system_prompt=SystemMessage(content=subscription_sys_prompt),
    context_schema=Context,
    tools=tool_long_term,
    checkpointer=checkpointer,
    store=store,
)

document_agent = create_agent(
    gllm,
    system_prompt=SystemMessage(content=document_sys_prompt),
    context_schema=Context,
    tools=tool_long_term,
    checkpointer=checkpointer,
    store=store,
)

response_agent = create_agent(
    gllm,
    system_prompt=SystemMessage(content=response_sys_prompt),
    context_schema=Context,
    tools=tool_long_term,
    checkpointer=checkpointer,
    store=store,
)

clarification_agent = create_agent(
    gllm,
    system_prompt=SystemMessage(content=clarification_sys_prompt),
    context_schema=Context,
    tools=tool_long_term,
    checkpointer=checkpointer,
    store=store,
)

voice_agent = create_agent(
    gllm,
    system_prompt=SystemMessage(content=voice_sys_prompt),
    context_schema=Context,
    tools=tool_long_term,
    checkpointer=checkpointer,
    store=store,
)
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


# ============================================================
# ✅ CONTEXT (STATE)
# ============================================================

class Context(BaseModel):
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    messages: List = []

    orchestrator_data: Optional[dict] = None
    intent_data: Optional[dict] = None
    product_data: Optional[dict] = None
    customer_data: Optional[dict] = None
    recommendation_data: Optional[dict] = None
    risk_data: Optional[dict] = None
    subscription_data: Optional[dict] = None
    document_data: Optional[dict] = None
    response_data: Optional[dict] = None
    clarification_data: Optional[dict] = None


# ============================================================
# ✅ THREAD HELPER
# ============================================================

def get_thread_id(base_thread: str, offset: int) -> str:
    try:
        return str(int(base_thread) + offset)
    except:
        return f"{base_thread}_{offset}"


# ============================================================
# ✅ NODES
# ============================================================

def orchestrator_node(state: Context):
    thread_id = get_thread_id(state.session_id, 0)

    result = orchestrator_agent.invoke(
        {"messages": state.messages},
        config={"configurable": {"thread_id": thread_id}}
    )

    state.orchestrator_data = result
    return state


def intent_node(state: Context):
    thread_id = get_thread_id(state.session_id, 1)

    result = intent_agent.invoke(
        {"messages": state.messages},
        config={"configurable": {"thread_id": thread_id}}
    )

    state.intent_data = result
    return state


def product_node(state: Context):
    thread_id = get_thread_id(state.session_id, 2)

    result = product_agent.invoke(
        {"messages": state.messages},
        config={"configurable": {"thread_id": thread_id}}
    )

    state.product_data = result
    return state


def customer_node(state: Context):
    thread_id = get_thread_id(state.session_id, 3)

    result = customer_agent.invoke(
        {"messages": state.messages},
        config={"configurable": {"thread_id": thread_id}}
    )

    state.customer_data = result
    return state


def recommendation_node(state: Context):
    thread_id = get_thread_id(state.session_id, 4)

    result = recommendation_agent.invoke(
        {"messages": state.messages},
        config={"configurable": {"thread_id": thread_id}}
    )

    state.recommendation_data = result
    return state


def risk_node(state: Context):
    thread_id = get_thread_id(state.session_id, 5)

    result = risk_agent.invoke(
        {"messages": state.messages},
        config={"configurable": {"thread_id": thread_id}}
    )

    state.risk_data = result
    return state


def subscription_node(state: Context):
    thread_id = get_thread_id(state.session_id, 6)

    result = subscription_agent.invoke(
        {"messages": state.messages},
        config={"configurable": {"thread_id": thread_id}}
    )

    state.subscription_data = result
    return state


def document_node(state: Context):
    thread_id = get_thread_id(state.session_id, 7)

    result = document_agent.invoke(
        {"messages": state.messages},
        config={"configurable": {"thread_id": thread_id}}
    )

    state.document_data = result
    return state


def response_node(state: Context):
    thread_id = get_thread_id(state.session_id, 8)

    result = response_agent.invoke(
        {"messages": state.messages},
        config={"configurable": {"thread_id": thread_id}}
    )

    state.response_data = result
    return state


def clarification_node(state: Context):
    thread_id = get_thread_id(state.session_id, 9)

    result = clarification_agent.invoke(
        {"messages": state.messages},
        config={"configurable": {"thread_id": thread_id}}
    )

    state.clarification_data = result
    return state
# ============================================================
# Optional Registry Helpers
# ============================================================
from typing import Literal
def conditional_edge(state: Context) -> Literal[
    "product",
    "customer",
    "recommendation",
    "subscription",
    "risk",
    "document",
    "response",
    "clarification"
]:
    """
    Routing logic based on intent, confidence, and workflow state.
    """

    intent = state.intent
    confidence = state.intent_confidence or 0.0
    missing = state.missing_information or []
    profile = state.customer_profile or {}

    # ------------------------
    # LOW CONFIDENCE → CLARIFY
    # ------------------------
    if confidence < 0.6:
        return "clarification"

    # ------------------------
    # MISSING INFO → CLARIFY
    # ------------------------
    if missing:
        return "clarification"

    # ------------------------
    # INTENT-BASED ROUTING
    # ------------------------
    if intent == "product_discovery":
        return "product"

    if intent == "product_comparison":
        return "product"

    if intent == "subscription":
        # if profile incomplete → customer first
        if not profile:
            return "customer"
        return "subscription"

    if intent == "policy_review":
        return "customer"

    if intent == "risk_analysis":
        return "risk"

    if intent == "document_help":
        return "document"

    if intent == "faq":
        return "response"

    # fallback
    return "clarification"


# ============================================================
# SECONDARY ROUTERS (PER NODE)
# ============================================================

def route_after_product(state: Context) -> Literal[
    "recommendation",
    "response",
    "clarification"
]:
    if state.missing_information:
        return "clarification"

    # assume recommendation needed if products exist
    if state.selected_product_id or state.tool_results:
        return "recommendation"

    return "response"


def route_after_customer(state: Context) -> Literal[
    "subscription",
    "document",
    "clarification",
    "response"
]:
    if state.missing_information:
        return "clarification"

    if state.intent == "subscription":
        return "subscription"

    if state.intent == "policy_review":
        return "document"

    return "response"


def route_after_recommendation(state: Context) -> Literal[
    "response",
    "clarification"
]:
    if state.missing_information:
        return "clarification"
    return "response"


def route_after_subscription(state: Context) -> Literal[
    "recommendation",
    "response",
    "clarification"
]:
    if state.missing_information:
        return "clarification"

    if not state.selected_product_id:
        return "recommendation"

    return "response"


def route_after_document(state: Context) -> Literal[
    "response",
    "clarification"
]:
    if state.missing_information:
        return "clarification"

    return "response"


def route_after_risk(state: Context) -> Literal[
    "recommendation",
    "response"
]:
    if state.risk_profile:
        return "recommendation"

    return "response"



agent_graph = StateGraph(Context)

# ------------------------
# REGISTER NODES (functions ✅)
# ------------------------

agent_graph.add_node("orchestrator", orchestrator_node)
agent_graph.add_node("intent", intent_node)
agent_graph.add_node("product", product_node)
agent_graph.add_node("customer", customer_node)
agent_graph.add_node("recommendation", recommendation_node)
agent_graph.add_node("risk", risk_node)
agent_graph.add_node("subscription", subscription_node)
agent_graph.add_node("document", document_node)
agent_graph.add_node("response", response_node)
agent_graph.add_node("clarification", clarification_node)


# ------------------------
# FLOW
# ------------------------

agent_graph.add_edge(START, "orchestrator")
agent_graph.add_edge("orchestrator", "intent")


agent_graph.add_conditional_edges("intent", conditional_edge)

agent_graph.add_conditional_edges("product", route_after_product)

agent_graph.add_conditional_edges("customer", route_after_customer)

agent_graph.add_conditional_edges("recommendation", route_after_recommendation)

agent_graph.add_conditional_edges("subscription", route_after_subscription)

agent_graph.add_conditional_edges("document", route_after_document)

agent_graph.add_conditional_edges("risk", route_after_risk)

agent_graph.add_edge("clarification", "intent")

agent_graph.add_edge("response", END)


insurance_chain = agent_graph.compile()
from langchain_core.messages import HumanMessage

@app.post('/multiagents/')
def agents(user_input: str, thr_id: str, usr_id: str):

    input_state = {
        "user_id": usr_id,
        "session_id": thr_id,

        # ✅ THIS IS CRITICAL
        "messages": [
            HumanMessage(content=user_input)
        ],

        # initialize fields
        "intent": None,
        "intent_confidence": None,
        "customer_profile": {},
        "risk_profile": {},
        "tool_results": {},
        "missing_information": []
    }

    result = insurance_chain.invoke(
        input_state
    )

    return result