import os
import sys
import operator
from typing import TypedDict, Literal, List, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage

PORT = 8111
print(f"\nConnecting to MedGemma Triage Agent (Port {PORT})...")

try:
    classifier_llm = ChatOpenAI(
        base_url=f"http://localhost:{PORT}/v1",
        api_key="sk-no-key-required",
        model="medgemma",
        temperature=0.0, 
        max_tokens=10,  
        streaming=False 
    )

    responder_llm = ChatOpenAI(
        base_url=f"http://localhost:{PORT}/v1",
        api_key="sk-no-key-required",
        model="medgemma",
        temperature=0.2,
        max_tokens=512,
        streaming=True 
    )
    print("LLM Clients initialized.")
except Exception as e:
    sys.exit(1)

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    severity: str  

def triage_node(state: AgentState):
    last_message = state["messages"][-1]
    user_text = last_message.content
    print(f"\n[Triage] Analyzing severity for: '{user_text}'")
    
    prompt = f"""
    You are an emergency triage nurse. Analyze the input.
    - If the symptom implies immediate life threat (heart attack, stroke, heavy bleeding, inability to breathe), output only: CRITICAL
    - Otherwise (cold, headache, diabetes info), output only: NORMAL
    - If the user is asking a question (e.g., "Is this bad?", "Could it be flu?"), output: NORMAL

    Examples:
    User: "I have a headache and runny nose." -> Output: NORMAL
    User: "My chest feels heavy and left arm hurts." -> Output: CRITICAL
    User: "I cut my finger, it stopped bleeding." -> Output: NORMAL
    User: "I am choking and cannot breathe." -> Output: CRITICAL
    User: "Could it be the flu?" -> Output: NORMAL
    
    Input: {user_text}
    Output:"""
    
    messages = [HumanMessage(content=prompt)]
    
    try:
        response = classifier_llm.invoke(messages)
        severity = response.content.strip().upper()
        if "NORMAL" in severity:
            severity = "NORMAL"
        elif "CRITICAL" in severity:
            severity = "CRITICAL"
        else:
            severity = "NORMAL"
            
        print(f"Severity Assessed: {severity}")
        return {"severity": severity}
        
    except Exception as e:
        print(f"Error in triage: {e}")
        return {"severity": "NORMAL"} 

def critical_response_node(state: AgentState):
    print(f"[Critical] Generating Emergency Protocol...")
    print(f"--- Stream Output: ---\n")
    
    system_prompt = SystemMessage(content="You are an Emergency Response System. The user is in danger. Output ONLY immediate action steps. Do NOT explain. Do NOT say what you are thinking. BUT: If the user is just asking a question (e.g., Is it flu?), answer it normally and calmly. Do NOT repeat yourself.")
    messages = [system_prompt] + state["messages"]
    
    full_response = ""
    try:
        for chunk in responder_llm.stream(messages):
            content = chunk.content
            if content:
                print(content, end="", flush=True)
                full_response += content
    except Exception as e:
        print(f"(Stream Error: {e})")

    if not full_response.strip():
        fallback = "Critical Error: Please call Emergency Services immediately."
        print(fallback)
        full_response = fallback
            
    return {"messages": [AIMessage(content=full_response)]}

def normal_response_node(state: AgentState):
    print(f"[Normal] Generating Medical Advice...")
    print(f"--- Stream Output: ---\n")
    
    system_prompt = SystemMessage(content="You are a helpful medical assistant. Answer concisely. Do NOT output any internal thoughts, thought tags, or reasoning steps.")
    messages = [system_prompt] + state["messages"]
    
    full_response = ""
    try:
        for chunk in responder_llm.stream(messages):
            content = chunk.content
            if content:
                print(content, end="", flush=True)
                full_response += content
    except Exception as e:
        print(f"(Stream Error: {e})")

    if not full_response.strip():
        fallback = "I apologize, I couldn't generate a response. Please consult a doctor."
        print(fallback)
        full_response = fallback
            
    return {"messages": [AIMessage(content=full_response)]}

def router_logic(state: AgentState) -> Literal["critical", "normal"]:
    if state["severity"] == "CRITICAL":
        return "critical"
    else:
        return "normal"

def build_graph():
    memory = MemorySaver()
    workflow = StateGraph(AgentState)
    
    workflow.add_node("triage", triage_node)
    workflow.add_node("critical_care", critical_response_node)
    workflow.add_node("general_advice", normal_response_node)
    
    workflow.set_entry_point("triage")
    workflow.add_conditional_edges(
        "triage",
        router_logic,
        {
            "critical": "critical_care",
            "normal": "general_advice"
        }
    )
    
    workflow.add_edge("critical_care", END)
    workflow.add_edge("general_advice", END)
    
    return workflow.compile(checkpointer=memory)

def get_next_session_id():
    counter_file = "session_counter.txt"
    
    if os.path.exists(counter_file):
        with open(counter_file, "r") as f:
            try:
                content = f.read().strip()
                current_num = int(content) if content else 0
            except ValueError:
                current_num = 0
    else:
        current_num = 0 
        
    next_num = current_num + 1
    
    with open(counter_file, "w") as f:
        f.write(str(next_num))
        
    return f"patient_session_{next_num:03d}"

if __name__ == '__main__':
    app = build_graph()
    
    thread_id = get_next_session_id()
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"\nMedGemma Agent Started (Session: {thread_id})")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 50)
    
    print(f"\nStarting Triage Agent...")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Bye!")
                break
            
            input_message = HumanMessage(content=user_input)
            app.invoke({"messages": [input_message]}, config=config)
            
            print("\n" + "-" * 50)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")