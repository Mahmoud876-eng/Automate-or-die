import os
import json
import traceback
from typing import Optional, List, Dict, Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy
from langgraph.errors import GraphRecursionError
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore


# ============================================================
# APP + ENV
# ============================================================

app = FastAPI()
load_dotenv()

checkpointer = MemorySaver()
store = InMemoryStore()


# ============================================================
# LLM
# ============================================================

# Make sure GOOGLE_API_KEY is available in your .env
# Example:
# GOOGLE_API_KEY=your_key_here

gllm = ChatGoogleGenerativeAI(
    model="gemma-4-26b-a4b-it",
    temperature=0.7,
)


# ============================================================
# RUNTIME CONTEXT FOR TOOLS ONLY
# ============================================================

class RuntimeContext(BaseModel):
    user_id: Optional[str] = None
    session_id: Optional[str] = None


# ============================================================
# GRAPH STATE ONLY
# ============================================================

class AgentState(BaseModel):
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    messages: List[Any] = Field(default_factory=list)

    intent: Optional[str] = None
    intent_confidence: float = 0.0
    missing_information: List[str] = Field(default_factory=list)

    customer_profile: Dict[str, Any] = Field(default_factory=dict)
    risk_profile: Dict[str, Any] = Field(default_factory=dict)
    tool_results: Dict[str, Any] = Field(default_factory=dict)
    selected_product_id: Optional[int] = None

    orchestrator_data: Dict[str, Any] = Field(default_factory=dict)
    intent_data: Dict[str, Any] = Field(default_factory=dict)
    product_data: Dict[str, Any] = Field(default_factory=dict)
    customer_data: Dict[str, Any] = Field(default_factory=dict)
    recommendation_data: Dict[str, Any] = Field(default_factory=dict)
    risk_data: Dict[str, Any] = Field(default_factory=dict)
    subscription_data: Dict[str, Any] = Field(default_factory=dict)
    document_data: Dict[str, Any] = Field(default_factory=dict)
    response_data: Dict[str, Any] = Field(default_factory=dict)
    clarification_data: Dict[str, Any] = Field(default_factory=dict)


# ============================================================
# STRUCTURED OUTPUT SCHEMAS
# ============================================================

class LongTermMemory(BaseModel):
    user_name: Optional[str] = Field(default=None, description="The user's name")
    preferences: List[str] = Field(default_factory=list, description="Stable user preferences")
    goals: List[str] = Field(default_factory=list, description="Stable user goals")


class IntentOutput(BaseModel):
    intent: Optional[str] = Field(
        default="unknown",
        description="Detected intent: product_discovery, product_comparison, subscription, policy_review, faq, risk_analysis, document_help, unknown",
    )
    confidence: float = Field(default=0.0, description="Confidence score between 0 and 1")
    missing_information: List[str] = Field(default_factory=list)
    reasoning_summary: Optional[str] = None
    suggested_next_node: Optional[str] = None


class OrchestratorOutput(BaseModel):
    next_step: Optional[str] = None
    reasoning_summary: Optional[str] = None


class ProductItem(BaseModel):
    product_id: Optional[int] = None
    name: Optional[str] = None
    category: Optional[str] = None
    monthly_price: Optional[float] = None
    annual_price: Optional[float] = None
    coverage_summary: Optional[str] = None


class ProductOutput(BaseModel):
    products_found: List[ProductItem] = Field(default_factory=list)
    comparison_mode: bool = False
    filters_used: Dict[str, Any] = Field(default_factory=dict)
    missing_information: List[str] = Field(default_factory=list)
    summary_for_downstream: Optional[str] = None


class CustomerOutput(BaseModel):
    customer_found: bool = False
    profile_completion_status: Optional[str] = None
    known_fields: Dict[str, Any] = Field(default_factory=dict)
    missing_fields: List[str] = Field(default_factory=list)
    summary_for_downstream: Optional[str] = None


class RecommendationItem(BaseModel):
    product_id: Optional[int] = None
    score: Optional[float] = None
    reason: Optional[str] = None


class RecommendationOutput(BaseModel):
    status: Optional[str] = None
    recommended_products: List[RecommendationItem] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)
    summary_for_response: Optional[str] = None


class RiskOutput(BaseModel):
    risk_profile_available: bool = False
    risk_scores: Dict[str, Any] = Field(default_factory=dict)
    risk_summary: Optional[str] = None
    missing_information: List[str] = Field(default_factory=list)


class SubscriptionOutput(BaseModel):
    subscription_stage: Optional[str] = None
    required_fields: List[str] = Field(default_factory=list)
    collected_fields: Dict[str, Any] = Field(default_factory=dict)
    missing_fields: List[str] = Field(default_factory=list)
    ready_for_next_step: bool = False
    suggested_next_node: Optional[str] = None


class DocumentOutput(BaseModel):
    documents_used: List[str] = Field(default_factory=list)
    document_summary: Optional[str] = None
    key_points: List[str] = Field(default_factory=list)
    missing_documents: List[str] = Field(default_factory=list)
    summary_for_response: Optional[str] = None


class ResponseOutput(BaseModel):
    text: Optional[str] = None
    markdown: Optional[str] = None
    suggestion_chips: List[str] = Field(default_factory=list)
    ui_payload: Dict[str, Any] = Field(default_factory=dict)


class ClarificationOutput(BaseModel):
    question: Optional[str] = None
    required_field: Optional[str] = None


# ============================================================
# STRUCTURED LLMS
# ============================================================

long_term_memory_llm = gllm.with_structured_output(LongTermMemory)
orchestrator_llm = gllm.with_structured_output(OrchestratorOutput)
intent_llm = gllm.with_structured_output(IntentOutput)
product_llm = gllm.with_structured_output(ProductOutput)
customer_llm = gllm.with_structured_output(CustomerOutput)
recommendation_llm = gllm.with_structured_output(RecommendationOutput)
risk_llm = gllm.with_structured_output(RiskOutput)
subscription_llm = gllm.with_structured_output(SubscriptionOutput)
document_llm = gllm.with_structured_output(DocumentOutput)
response_llm = gllm.with_structured_output(ResponseOutput)
clarification_llm = gllm.with_structured_output(ClarificationOutput)


# ============================================================
# TOOLS
# ============================================================

@tool
def store_memory(runtime: ToolRuntime[RuntimeContext]) -> ToolMessage:
    """Save useful long-term user profile information."""
    messages = runtime.state.get("messages", [])
    user_id = runtime.context.user_id or "unknown_user"
    namespace = (user_id, "memories")

    human_message = None
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            human_message = message
            break

    if human_message is None:
        return ToolMessage(
            content="No human message found to store.",
            tool_call_id=runtime.tool_call_id,
        )

    extracted = long_term_memory_llm.invoke([
        SystemMessage(content="Extract stable user profile information. Do not invent anything."),
        HumanMessage(content=human_message.content),
    ])

    runtime.store.put(namespace, "a-memory", jsonable_encoder(extracted))

    return ToolMessage(
        content="User memory saved.",
        tool_call_id=runtime.tool_call_id,
    )


@tool
def retrieve_memory(runtime: ToolRuntime[RuntimeContext]) -> ToolMessage:
    """Retrieve stored user profile and preferences."""
    user_id = runtime.context.user_id or "unknown_user"
    namespace = (user_id, "memories")
    personal_data = runtime.store.get(namespace, "a-memory")

    if personal_data is None:
        content = "No profile data found."
    elif isinstance(personal_data, (dict, list)):
        content = json.dumps(personal_data, ensure_ascii=False)
    else:
        content = str(personal_data)

    return ToolMessage(
        content=content,
        tool_call_id=runtime.tool_call_id,
    )


tool_long_term = [retrieve_memory, store_memory]


# ============================================================
# SYSTEM PROMPTS
# ============================================================

shared_rules_block = """
You are a specialized agent inside an AI Insurance Assistant orchestrated by LangGraph.

Global rules:
- Do not fabricate policy, pricing, customer, risk, or document facts.
- If required information is missing, state exactly what is missing.
- Return concise and structured workflow-oriented outputs.
- Use tools only when appropriate.
"""

orchestrator_sys_prompt = f"""
You are the Orchestrator Agent.
{shared_rules_block}
Decide the next best workflow step. Do not answer the user directly.
"""

intent_sys_prompt = f"""
You are the Intent Agent.
{shared_rules_block}
Classify the latest user request into one of:
- product_discovery
- product_comparison
- subscription
- policy_review
- faq
- risk_analysis
- document_help
- unknown

For product discovery, do not require every detail immediately. Missing information should only include what is absolutely necessary.
"""

product_sys_prompt = f"""
You are the Product Agent.
{shared_rules_block}
Extract the requested insurance product category and useful filters.
Do not invent real products, prices, or coverage.
If product data is unavailable, return an empty product list and summarize what is needed.
"""

customer_sys_prompt = f"""
You are the Customer Agent.
{shared_rules_block}
Identify known customer details from the conversation and missing fields needed for the workflow.
Do not invent customer data.
"""

recommendation_sys_prompt = f"""
You are the Recommendation Agent.
{shared_rules_block}
Use available workflow context to prepare a recommendation summary.
If product data is unavailable or insufficient, return status needs_more_information.
Do not invent prices, eligibility rules, exclusions, or coverage facts.
"""

risk_sys_prompt = f"""
You are the Risk Agent.
{shared_rules_block}
Summarize available risk information only if present in the conversation/context.
Do not invent actuarial scores.
"""

subscription_sys_prompt = f"""
You are the Subscription Agent.
{shared_rules_block}
Track the insurance subscription workflow and identify missing required fields.
Do not finalize any subscription.
"""

document_sys_prompt = f"""
You are the Document Agent.
{shared_rules_block}
Analyze document-related requests and identify needed documents.
Do not claim that documents were generated, signed, or verified unless explicitly provided.
"""

response_sys_prompt = f"""
You are the Response Agent.
{shared_rules_block}
Generate the final user-facing answer from the workflow context.
Be concise, helpful, and clear.
"""

clarification_sys_prompt = f"""
You are the Clarification Agent.
{shared_rules_block}
Ask one short clarifying question for the most important missing field.
Do not ask multiple questions at once.
"""

chatbot_sys_prompt = """
You are a helpful conversational assistant with long-term memory.
Use retrieve_memory when the user references stored information.
Use store_memory when the user shares stable personal preferences, profile facts, or asks you to remember something.
"""


# ============================================================
# SIMPLE MEMORY CHATBOT AGENT
# ============================================================

run_agent = create_agent(
    gllm,
    system_prompt=SystemMessage(content=chatbot_sys_prompt),
    context_schema=RuntimeContext,
    tools=tool_long_term,
    checkpointer=checkpointer,
    store=store,
)


# ============================================================
# HELPERS
# ============================================================

def get_state_value(result: Any, key: str, default: Any = None) -> Any:
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


def make_context_message(state: AgentState) -> HumanMessage:
    human_texts = [m.content for m in state.messages if isinstance(m, HumanMessage)]
    payload = {
        "user_messages": human_texts,
        "intent": state.intent,
        "intent_confidence": state.intent_confidence,
        "missing_information": state.missing_information,
        "product_data": state.product_data,
        "customer_profile": state.customer_profile,
        "customer_data": state.customer_data,
        "recommendation_data": state.recommendation_data,
        "risk_profile": state.risk_profile,
        "risk_data": state.risk_data,
        "subscription_data": state.subscription_data,
        "document_data": state.document_data,
    }
    return HumanMessage(content=json.dumps(payload, ensure_ascii=False))


# ============================================================
# GRAPH NODES
# ============================================================

def orchestrator_node(state: AgentState):
    result = orchestrator_llm.invoke([
        SystemMessage(content=orchestrator_sys_prompt),
        make_context_message(state),
    ])
    state.orchestrator_data = result.model_dump()
    return state


def intent_node(state: AgentState):
    result = intent_llm.invoke([
        SystemMessage(content=intent_sys_prompt),
        *state.messages,
    ])
    state.intent_data = result.model_dump()
    state.intent = result.intent or "unknown"
    state.intent_confidence = result.confidence or 0.0
    state.missing_information = result.missing_information or []
    return state


def product_node(state: AgentState):
    result = product_llm.invoke([
        SystemMessage(content=product_sys_prompt),
        make_context_message(state),
    ])
    state.product_data = result.model_dump()
    state.missing_information = result.missing_information or []
    return state


def customer_node(state: AgentState):
    result = customer_llm.invoke([
        SystemMessage(content=customer_sys_prompt),
        make_context_message(state),
    ])
    state.customer_data = result.model_dump()
    state.customer_profile = result.known_fields or {}
    state.missing_information = result.missing_fields or []
    return state


def recommendation_node(state: AgentState):
    result = recommendation_llm.invoke([
        SystemMessage(content=recommendation_sys_prompt),
        make_context_message(state),
    ])
    state.recommendation_data = result.model_dump()
    state.missing_information = result.missing_information or []
    return state


def risk_node(state: AgentState):
    result = risk_llm.invoke([
        SystemMessage(content=risk_sys_prompt),
        make_context_message(state),
    ])
    state.risk_data = result.model_dump()
    state.risk_profile = result.risk_scores or {}
    state.missing_information = result.missing_information or []
    return state


def subscription_node(state: AgentState):
    result = subscription_llm.invoke([
        SystemMessage(content=subscription_sys_prompt),
        make_context_message(state),
    ])
    state.subscription_data = result.model_dump()
    state.missing_information = result.missing_fields or []
    return state


def document_node(state: AgentState):
    result = document_llm.invoke([
        SystemMessage(content=document_sys_prompt),
        make_context_message(state),
    ])
    state.document_data = result.model_dump()
    state.missing_information = result.missing_documents or []
    return state


def response_node(state: AgentState):
    result = response_llm.invoke([
        SystemMessage(content=response_sys_prompt),
        make_context_message(state),
    ])
    state.response_data = result.model_dump()
    return state


def clarification_node(state: AgentState):
    result = clarification_llm.invoke([
        SystemMessage(content=clarification_sys_prompt),
        make_context_message(state),
    ])
    state.clarification_data = result.model_dump()
    return state


# ============================================================
# ROUTERS
# ============================================================

def conditional_edge(state: AgentState) -> Literal[
    "product",
    "customer",
    "subscription",
    "risk",
    "document",
    "response",
    "clarification",
]:
    intent = state.intent
    confidence = state.intent_confidence or 0.0
    missing = state.missing_information or []

    if confidence < 0.6:
        return "clarification"

    if missing:
        return "clarification"

    if intent in ["product_discovery", "product_comparison"]:
        return "product"

    if intent == "subscription":
        if not state.customer_profile:
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

    return "clarification"


def route_after_product(state: AgentState) -> Literal["recommendation", "response", "clarification"]:
    if state.missing_information:
        return "clarification"
    if state.intent in ["product_discovery", "product_comparison"]:
        return "recommendation"
    return "response"


def route_after_customer(state: AgentState) -> Literal["subscription", "document", "response", "clarification"]:
    if state.missing_information:
        return "clarification"
    if state.intent == "subscription":
        return "subscription"
    if state.intent == "policy_review":
        return "document"
    return "response"


def route_after_recommendation(state: AgentState) -> Literal["response", "clarification"]:
    if state.missing_information:
        return "clarification"
    return "response"


def route_after_subscription(state: AgentState) -> Literal["recommendation", "response", "clarification"]:
    if state.missing_information:
        return "clarification"
    if not state.selected_product_id:
        return "recommendation"
    return "response"


def route_after_document(state: AgentState) -> Literal["response", "clarification"]:
    if state.missing_information:
        return "clarification"
    return "response"


def route_after_risk(state: AgentState) -> Literal["recommendation", "response"]:
    if state.risk_profile:
        return "recommendation"
    return "response"


# ============================================================
# GRAPH BUILD
# ============================================================

agent_graph = StateGraph(AgentState)

agent_graph.add_node("orchestrator", orchestrator_node, retry_policy=RetryPolicy(max_attempts=3))
agent_graph.add_node("intent", intent_node, retry_policy=RetryPolicy(max_attempts=3))
agent_graph.add_node("product", product_node, retry_policy=RetryPolicy(max_attempts=3))
agent_graph.add_node("customer", customer_node, retry_policy=RetryPolicy(max_attempts=3))
agent_graph.add_node("recommendation", recommendation_node, retry_policy=RetryPolicy(max_attempts=3))
agent_graph.add_node("risk", risk_node, retry_policy=RetryPolicy(max_attempts=3))
agent_graph.add_node("subscription", subscription_node, retry_policy=RetryPolicy(max_attempts=3))
agent_graph.add_node("document", document_node, retry_policy=RetryPolicy(max_attempts=3))
agent_graph.add_node("response", response_node, retry_policy=RetryPolicy(max_attempts=3))
agent_graph.add_node("clarification", clarification_node, retry_policy=RetryPolicy(max_attempts=3))

agent_graph.add_edge(START, "orchestrator")
agent_graph.add_edge("orchestrator", "intent")

agent_graph.add_conditional_edges("intent", conditional_edge)
agent_graph.add_conditional_edges("product", route_after_product)
agent_graph.add_conditional_edges("customer", route_after_customer)
agent_graph.add_conditional_edges("recommendation", route_after_recommendation)
agent_graph.add_conditional_edges("subscription", route_after_subscription)
agent_graph.add_conditional_edges("document", route_after_document)
agent_graph.add_conditional_edges("risk", route_after_risk)

# IMPORTANT: clarification must END.
# Do not do clarification -> intent, because that creates an infinite loop without a new user answer.
agent_graph.add_edge("clarification", END)
agent_graph.add_edge("response", END)

insurance_chain = agent_graph.compile()


# ============================================================
# API ENDPOINTS
# ============================================================

@app.post("/multiagents/")
def agents(user_input: str, thr_id: str, usr_id: str):
    input_state = AgentState(
        user_id=usr_id,
        session_id=thr_id,
        messages=[HumanMessage(content=user_input)],
        intent=None,
        intent_confidence=0.0,
        missing_information=[],
        customer_profile={},
        risk_profile={},
        tool_results={},
    )

    try:
        result = insurance_chain.invoke(input_state, {"recursion_limit": 25})

        return jsonable_encoder({
            "status": "success",
            "intent": get_state_value(result, "intent"),
            "intent_confidence": get_state_value(result, "intent_confidence"),
            "missing_information": get_state_value(result, "missing_information", []),
            "clarification": get_state_value(result, "clarification_data", {}),
            "response": get_state_value(result, "response_data", {}),
            "product": get_state_value(result, "product_data", {}),
            "customer": get_state_value(result, "customer_data", {}),
            "recommendation": get_state_value(result, "recommendation_data", {}),
            "risk": get_state_value(result, "risk_data", {}),
            "subscription": get_state_value(result, "subscription_data", {}),
            "document": get_state_value(result, "document_data", {}),
        })

    except GraphRecursionError as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"LangGraph recursion error: {str(e)}")

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chatbot/")
def chatbot(user_input: str, thr_id: str, usr_id: str):
    try:
        result = run_agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config={"configurable": {"thread_id": thr_id}},
            context=RuntimeContext(user_id=usr_id, session_id=thr_id),
        )
        return jsonable_encoder(result)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
