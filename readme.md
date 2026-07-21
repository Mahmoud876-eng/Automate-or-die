=# README.md

# Multi-Agent Company Assistant & Insurance Memory Platform
## run the code 
fastapi dev deep.py
## Overview

This project is a **LangGraph + DeepAgents + FastAPI** application that demonstrates two different AI-powered systems:

1. **Multi-Agent Company Assistant**
   * Uses multiple specialized AI agents coordinated by a supervisor.
   * Helps with products, customers, policies, documents, and risk analysis.
   * Demonstrates delegation, routing, and answer synthesis.

2. **Insurance Memory Assistant**
   * Collects, stores, retrieves, and manages insurance claim information.
   * Maintains long-term memory per user.
   * Extracts structured insurance files automatically using Pydantic schemas.
   * Can later generate insurance recommendation documents.

This project was built as a **hackathon prototype** to demonstrate:

* Multi-agent orchestration
* Tool calling
* Long-term memory
* Structured extraction
* FastAPI APIs
* LangGraph persistence

***

# Architecture

```
                    +----------------+
                    |     User       |
                    +-------+--------+
                            |
                            v
                  +------------------+
                  |     FastAPI      |
                  +------------------+
                            |
          --------------------------------------
          |                                    |
          v                                    v

+-----------------------+     +------------------------+
| Multi-Agent Assistant |     | Insurance Assistant    |
+-----------------------+     +------------------------+

            |                            |
            v                            v

+------------------+        +------------------------+
| Supervisor Agent |        | Long Term Memory Agent |
+------------------+        +------------------------+

            |
            v

   +-------------------+
   | Specialist Agents |
   +-------------------+

      Product Agent
      Customer Agent
      Document Agent
      Risk Agent
      Answer Agent
```

***

# Main Technologies

### AI Frameworks

* LangGraph
* LangChain
* DeepAgents

### Models

* GPT-4.1 (Azure OpenAI)
* Mistral Large 3 (Azure AI Inference)

### APIs

* FastAPI

### Memory

* InMemoryStore
* InMemorySaver

### Validation

* Pydantic

***

# Part 1: Multi-Agent Company Assistant

***

## Purpose

The company assistant simulates an internal AI team.

Instead of one AI answering everything, tasks are delegated to specialists.

Example request:

> ACME wants 20 laptops and a 15% discount. Is this allowed?

The system may:

1. Check products.
2. Check customer profile.
3. Check company policies.
4. Run risk analysis.
5. Generate a final recommendation.

***

# Available Demo Data

***

## Products

### Laptop Pro 14

* Price: $1299
* Stock: 18
* Best for:
  * Developers
  * Analysts
  * Business users

***

### Tablet Air

* Price: $599
* Stock: 42
* Best for:
  * Sales teams
  * Field workers

***

### Secure Router X

* Price: $249
* Stock: 7
* Best for:
  * Small offices
  * Security-focused teams

***

## Customers

### ACME Corp

* Enterprise customer
* Active contract
* Security review required for purchases above $1000

### StartupX

* Startup customer
* Trial account
* Limited commitment approval

***

## Internal Documents

The demo contains policies including:

* Return policy
* Discount policy
* Security policy
* Shipping policy

***

# Specialist Agents

***

## Product Specialist

Responsible for:

* Product recommendations
* Pricing
* Availability
* Stock checking

Uses:

```python
search_products()
```

***

## Customer Specialist

Responsible for:

* Customer lookup
* Customer status
* Region
* Contracts

Uses:

```python
lookup_customer()
```

***

## Document Specialist

Responsible for:

* Internal policies
* Shipping rules
* Return policies
* Discount rules

Uses:

```python
search_documents()
```

***

## Risk Specialist

Responsible for:

* Compliance review
* Approval requirements
* Security checks
* Risk flags

Uses:

```python
run_risk_check()
```

***

## Answer Synthesizer

Responsible for:

* Combining all specialist outputs
* Producing the final user response

***

# Supervisor Agent

The supervisor is the brain of the system.

Prompt:

```text
SUPERVISOR_PROMPT
```

Responsibilities:

### 1. Understand Request

Classify:

* Product question
* Customer question
* Policy question
* Risk request
* Decision support

***

### 2. Ask Clarifying Questions

If information is missing:

Example:

> Recommend a laptop

Supervisor asks:

> What will the laptop be used for?

***

### 3. Route Work

Delegates to relevant specialists.

***

### 4. Gather Results

Collects internal notes.

***

### 5. Generate Final Answer

Calls:

```python
answer_synthesizer
```

***

# Multi-Agent Endpoint

### POST

```http
/multiagents/
```

### Parameters

```text
user_input
thr_id
usr_id
```

### Example

```http
POST /multiagents/

user_input="ACME wants laptops with a discount"
thr_id="thread_123"
usr_id="user_456"
```

***

# Part 2: Insurance Assistant

***

## Purpose

The insurance assistant acts as an AI insurance advisor capable of:

* Collecting claim information
* Storing files
* Updating files
* Retrieving files
* Persisting memory
* Generating insurance documents

***

# Insurance Claim Schema

The project uses a structured model:

```python
InsuranceFileMemory
```

Fields include:

### Personal Information

* claimant\_name
* policy\_number

### Insurance Details

* insurance\_type

Examples:

* Auto
* Health
* Home
* Life
* Travel

### Incident Information

* incident\_type
* incident\_date
* incident\_location
* incident\_description

### Damage Information

* damages
* injuries

### Supporting Information

* documents\_provided

### Workflow Information

* claim\_status
* claim\_priority
* next\_actions
* missing\_information

***

# Automatic Claim Extraction

When users provide information such as:

```text
I had a car accident yesterday.
My name is John Doe.
My policy number is 12345.
The accident happened in Tunis.
```

The system automatically extracts:

```json
{
  "claimant_name": "John Doe",
  "policy_number": "12345",
  "insurance_type": "auto",
  "incident_type": "accident"
}
```

***

# Insurance Tools

***

## Store Insurance File

Tool:

```python
store_insurance_file()
```

Purpose:

* Extract claim information
* Create file
* Update file
* Save long-term insurance data

Stored Under:

```python
(user_id, "insurance_files")
```

Key:

```python
current_insurance_file
```

***

## Retrieve Insurance File

Tool:

```python
retrieve_insurance_file()
```

Purpose:

* Resume a claim
* Check missing information
* View collected information

Returns:

```json
{
  "status": "found",
  "insurance_file": {}
}
```

***

# Long-Term User Memory

***

## Memory Schema

```python
LongTermMemory
```

Stores:

* User name
* Preferences
* Profile information

***

## Store Memory

Tool:

```python
store_memory()
```

Examples:

```text
My name is Mahmoud.
```

or

```text
I like science and technology.
```

***

## Retrieve Memory

Tool:

```python
retrieve_memory()
```

Returns stored user profile information.

***

# Recommendation Redirect Tool

Tool:

```python
redirect_to_recomendation()
```

Purpose:

Redirects workflow toward recommendation generation.

Returns:

```json
{
  "action":"redirect",
  "target":"/recomendation/"
}
```

***

# Insurance Document Generation

Tool:

```python
generate_insurance_document()
```

Purpose:

Generate a customer-facing insurance report.

The final document may include:

* Customer profile
* Insurance needs
* Risk summary
* Coverage recommendation
* Premium estimates
* Advisor recommendations

Current implementation:

* Redirects to document endpoint
* Retrieves stored memory
* Returns available user data

***

# Insurance API Endpoints

***

## Chatbot

### POST

```http
/chatbot/
```

Parameters:

```text
user_input
thr_id
usr_id
```

Purpose:

Main insurance assistant endpoint.

***

## Recommendation

### POST

```http
/recomendation/
```

Purpose:

Handles recommendation workflow.

***

## Insurance Document

### POST

```http
/generate_insurance_document/
```

Purpose:

Returns stored customer information that can later be transformed into:

* PDF
* DOCX
* Insurance Proposal
* Recommendation Report

***

# Memory System

The application uses two memory layers.

***

## Short-Term Memory

Uses:

```python
InMemorySaver()
```

Purpose:

Conversation history.

Stored by:

```python
thread_id
```

Example:

```python
thread_123
```

***

## Long-Term Memory

Uses:

```python
InMemoryStore()
```

Purpose:

Persistent user information.

Stored by:

```python
user_id
```

Example:

```python
(user_456, "insurance_files")
```

***

# Example Insurance Flow

### Step 1

User:

```text
I had a car accident in Tunis yesterday.
```

***

### Step 2

AI calls:

```python
store_insurance_file()
```

***

### Step 3

Data stored:

```json
{
  "incident_type": "accident",
  "incident_location": "Tunis"
}
```

***

### Step 4

User:

```text
What information is still missing?
```

***

### Step 5

AI calls:

```python
retrieve_insurance_file()
```

***

### Step 6

Response:

```json
{
  "missing_information": [
    "policy number",
    "claimant name",
    "damages"
  ]
}
```

***

# Future Improvements

This hackathon demo can easily be extended with:

### Persistence

* PostgreSQL
* Redis
* MongoDB

### AI

* RAG
* Vector Databases
* Knowledge Bases

### Insurance Features

* Product Recommendation Engine
* Premium Calculation
* Risk Scoring
* Claim Processing

### Documents

* PDF Generation
* DOCX Generation
* Digital Signatures

### Integrations

* CRM
* Insurance Core Systems
* Payment Providers
* Government APIs

***

# Summary

This application combines:

✅ Multi-Agent AI Orchestration  
✅ Supervisor + Specialist Pattern  
✅ LangGraph Memory Management  
✅ Insurance Claim Data Extraction  
✅ Long-Term User Memory  
✅ FastAPI APIs  
✅ Structured Pydantic Outputs  
✅ Recommendation Workflows  
✅ Insurance Document Generation

It is designed as a hackathon-ready foundation for building a production-grade AI insurance advisor and enterprise decision-support platform.