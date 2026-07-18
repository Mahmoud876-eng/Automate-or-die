# AI Insurance Assistant with LangGraph, FastAPI, Long-Term Memory & Multi-Agent Architecture

## 📌 Overview

This project is a **production-ready AI Insurance Assistant** built using:

* **LangGraph**
* **LangChain Agents**
* **Google Gemma Models**
* **Azure AI Foundry Models**
* **FastAPI**
* **Structured Outputs (Pydantic)**
* **Long-Term Memory**
* **Multi-Agent Orchestration**
* **Insurance Workflow Automation**

The system combines conversational AI, persistent memory, insurance file extraction, recommendation engines, workflow orchestration, and specialized agents into a single architecture.

***

# Architecture

```text
                         User
                           │
                           ▼
                    FastAPI Endpoint
                           │
                           ▼
                    LangGraph Router
                           │
        ┌──────────────────┴──────────────────┐
        │                                     │
        ▼                                     ▼
 Simple Memory Agent              Multi-Agent Workflow
        │                                     │
        │                                     ▼
        │                           Orchestrator Agent
        │                                     │
        │                                     ▼
        │                              Intent Agent
        │                                     │
        │                     ┌───────────────┴──────────────┐
        │                     │                              │
        ▼                     ▼                              ▼
 Long-Term Memory      Product Agent                Customer Agent
        │                     │                              │
        │                     ▼                              ▼
        │              Recommendation Agent       Subscription Agent
        │
        ▼
 Insurance File
 Extraction Tool

```

***

# Features

## ✅ Conversational AI

The assistant can:

* Answer insurance questions
* Guide customers
* Explain policies
* Help in product selection
* Handle subscription workflows

***

## ✅ Long-Term User Memory

The system stores user information across sessions.

Examples:

```text
"My name is Mahmoud"

"I prefer life insurance"

"I learn through examples"
```

Stored memory:

```json
{
  "user_name": "Mahmoud"
}
```

Uses:

* Personalization
* Recommendations
* Profile recall
* Workflow continuity

***

## ✅ Insurance File Extraction

The system can automatically transform user conversations into structured insurance claims.

Example:

```text
I had a car accident yesterday in Tunis.
My car is a 2023 Toyota Corolla.
```

Output:

```json
{
  "insurance_type": "auto",
  "incident_type": "accident",
  "incident_location": "Tunis",
  "vehicle_information": "2023 Toyota Corolla"
}
```

***

## ✅ Structured Outputs Everywhere

Every agent produces predictable JSON outputs using Pydantic.

Benefits:

* Reliable orchestration
* Strong validation
* Easy frontend integration
* Tool interoperability

***

## Technology Stack

### Backend

* FastAPI
* LangGraph
* LangChain

### Models

#### Azure AI Foundry

```python
AzureAIOpenAIApiChatModel(
    model="Mistral-Large-3"
)
```

Used for:

* Reasoning
* Insurance workflows
* Production inference

***

#### Google Gemini / Gemma

```python
ChatGoogleGenerativeAI(
    model="gemma-4-26b-a4b-it"
)
```

Used for:

* Structured extraction
* Memory generation
* Agent execution

***

### Persistence

#### Short-Term Memory

```python
MemorySaver()
```

Stores conversation history by:

```text
thread_id
```

***

#### Long-Term Memory

```python
InMemoryStore()
```

Stores:

```text
user_id
```

namespaced memories.

Example:

```python
(user_id, "memories")
```

***

# Available Tools

***

## store\_memory

Stores important user information.

Example:

```text
"My name is Sarah"
```

Triggers:

```python
store_memory()
```

Output:

```json
{
  "user_name": "Sarah"
}
```

***

## retrieve\_memory

Searches saved user memory.

Example:

```text
"What's my name?"
```

Returns:

```json
{
  "user_name": "Sarah"
}
```

***

## fillfile

Creates structured insurance claim files.

Extracted fields include:

* Claimant Name
* Policy Number
* Insurance Type
* Incident Type
* Date
* Location
* Damage Reports
* Injuries
* Documents
* Missing Information
* Next Actions

***

## redirect\_to\_recomendation

Routes users to recommendation workflows.

Used when:

```text
Generate recommendations
Suggest policies
Find matching insurance plans
```

***

# Insurance Memory Schema

The claim extraction engine produces:

```python
InsuranceFileMemory
```

Fields:

| Field               | Description                |
| ------------------- | -------------------------- |
| claimant\_name      | Claimant                   |
| policy\_number      | Policy number              |
| insurance\_type     | Insurance category         |
| incident\_type      | Accident, theft, fire, etc |
| incident\_date      | Incident date              |
| incident\_location  | Incident location          |
| damages             | Damages list               |
| injuries            | Injuries list              |
| documents\_provided | Submitted documents        |
| claim\_status       | Current state              |
| claim\_priority     | Priority level             |
| next\_actions       | Recommended actions        |

***

# Agent Architecture

The platform uses specialized agents.

***

## 1. Orchestrator Agent

Responsible for:

* Workflow routing
* Agent selection
* State management

Never answers the user directly.

***

## 2. Intent Agent

Detects:

```text
product_discovery
product_comparison
subscription
policy_review
risk_analysis
document_help
faq
```

Returns:

```json
{
  "intent": "subscription",
  "confidence": 0.95
}
```

***

## 3. Product Agent

Handles:

* Product search
* Product filtering
* Product comparison

Returns:

```json
{
  "products_found": [...]
}
```

***

## 4. Customer Agent

Retrieves:

* Profile information
* Missing user data
* Customer completeness

***

## 5. Recommendation Agent

Generates:

```json
{
  "recommended_products": [...]
}
```

Based on:

* Customer profile
* Product catalog
* Risk profile

***

## 6. Risk Agent

Analyzes:

* Risk data
* Risk scores
* Eligibility indicators

***

## 7. Subscription Agent

Handles onboarding workflows:

```text
Quote
Application
Verification
Submission
```

***

## 8. Document Agent

Analyzes:

* Policies
* Claims
* Uploaded documents

Produces:

```json
{
  "document_summary": "...",
  "key_points": [...]
}
```

***

## 9. Clarification Agent

Used when:

```text
Confidence < 0.6
Missing information exists
```

Example:

```text
"What type of insurance are you interested in?"
```

***

## 10. Response Agent

Creates the final user-facing response.

Produces:

```json
{
  "text": "...",
  "markdown": "...",
  "suggestion_chips": [...]
}
```

***

## Workflow Routing

```text
START
  │
  ▼
Orchestrator
  │
  ▼
Intent
  │
  ├─► Product
  ├─► Customer
  ├─► Risk
  ├─► Document
  ├─► Subscription
  ├─► Clarification
  └─► Response
```

Secondary agents can route to:

```text
Recommendation
Response
Clarification
```

until the workflow completes.

***

# REST API Endpoints

## Chat Endpoint

```http
POST /chatbot/
```

Parameters:

```json
{
  "user_input": "Hello",
  "thr_id": "123",
  "usr_id": "u001"
}
```

***

## Recommendation Endpoint

```http
POST /recomendation/
```

Example:

```json
{
  "message": "Find me travel insurance",
  "thr_id": "123",
  "usr_id": "u001"
}
```

***

## Multi-Agent Endpoint

```http
POST /multiagents/
```

Example:

```json
{
  "user_input": "I need auto insurance",
  "thr_id": "123",
  "usr_id": "u001"
}
```

***

# Environment Variables

Create a `.env` file:

```env
AZURE_AI_INFERENCE_CREDENTIAL=YOUR_API_KEY
AZURE_AI_INFERENCE_ENDPOINT=YOUR_ENDPOINT
GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
```

***

# Installation

```bash
git clone https://github.com/your-org/insurance-ai-assistant.git

cd insurance-ai-assistant

pip install -r requirements.txt
```

***

# Run the API

```bash
uvicorn main:app --reload
```

Server:

```text
http://localhost:8000
```

Swagger:

```text
http://localhost:8000/docs
```

***

# Future Improvements

* Redis Memory Store
* PostgreSQL Persistence
* Vector Database Integration
* RAG for Policy Documents
* MCP Server Support
* Voice Agent Integration
* Insurance CRM Integration
* Underwriting Agents
* Fraud Detection Agent
* Multi-Tenant Architecture
* Human-in-the-Loop Approval Workflows

***

# Example Use Cases

### Insurance Product Recommendation

```text
User → I need family health insurance
↓
Intent Agent
↓
Product Agent
↓
Recommendation Agent
↓
Response Agent
```

### Insurance Claim Creation

```text
User → My house was damaged by flooding
↓
fillfile()
↓
InsuranceFileMemory
↓
Claim Record
```

### Subscription Workflow

```text
User → I want auto insurance
↓
Intent
↓
Customer
↓
Subscription
↓
Recommendation
↓
Response
```

***

# License

MIT License

***

Built with ❤️ using **LangGraph, LangChain, FastAPI, Azure AI Foundry, Google GenAI, and Multi-Agent Architecture for Insurance Automation**.
