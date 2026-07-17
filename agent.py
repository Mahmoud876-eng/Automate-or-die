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
tool_long_term=[
                retrieve_memory,
                store_memory
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
