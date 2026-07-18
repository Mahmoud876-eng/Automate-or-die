from langgraph.graph import StateGraph, START, END
from typing import Literal

# your imports (adapt paths if needed)
from ..models import Context
from ..core import checkpointer

# agents (from your previous file)
from ..agents import (
    orchestrator_agent,
    intent_agent,
    product_agent,
    customer_agent,
    recommendation_agent,
    risk_agent,
    subscription_agent,
    document_agent,
    response_agent,
    clarification_agent,
)

from langgraph.types import RetryPolicy


# ============================================================
# CONDITIONAL EDGE (CORE ROUTER)
# ============================================================

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


# ============================================================
# GRAPH DEFINITION
# ============================================================

agent_graph = StateGraph(Context)


# ------------------------
# NODES
# ------------------------

agent_graph.add_node("orchestrator", orchestrator_agent)
agent_graph.add_node("intent", intent_agent)
agent_graph.add_node("product", product_agent)
agent_graph.add_node("customer", customer_agent)
agent_graph.add_node("recommendation", recommendation_agent)
agent_graph.add_node("risk", risk_agent)
agent_graph.add_node("subscription", subscription_agent)
agent_graph.add_node("document", document_agent)
agent_graph.add_node("response", response_agent)
agent_graph.add_node("clarification", clarification_agent)


# ------------------------
# MAIN FLOW
# ------------------------

agent_graph.add_edge(START, "orchestrator")
agent_graph.add_edge("orchestrator", "intent")

# intent → conditional routing
agent_graph.add_conditional_edges("intent", conditional_edge)


# ------------------------
# PRODUCT FLOW
# ------------------------

agent_graph.add_conditional_edges("product", route_after_product)


# ------------------------
# CUSTOMER FLOW
# ------------------------

agent_graph.add_conditional_edges("customer", route_after_customer)


# ------------------------
# RECOMMENDATION FLOW
# ------------------------

agent_graph.add_conditional_edges("recommendation", route_after_recommendation)


# ------------------------
# SUBSCRIPTION FLOW
# ------------------------

agent_graph.add_conditional_edges("subscription", route_after_subscription)


# ------------------------
# DOCUMENT FLOW
# ------------------------

agent_graph.add_conditional_edges("document", route_after_document)


# ------------------------
# RISK FLOW
# ------------------------

agent_graph.add_conditional_edges("risk", route_after_risk)


# ------------------------
# CLARIFICATION LOOP
# ------------------------

agent_graph.add_edge("clarification", "intent")


# ------------------------
# FINAL NODE
# ------------------------

agent_graph.add_edge("response", END)


# ============================================================
# COMPILE GRAPH
# ============================================================

insurance_chain = agent_graph.compile(
    checkpointer=checkpointer,
    retry_policy=RetryPolicy(max_attempts=2)
)