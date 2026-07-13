from typing import Dict, List
from openai import OpenAI

def generate_response(openai_key: str, user_message: str, context: str, 
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context"""
    system_prompt = (
        "You are a NASA mission operations expert. Answer user questions using ONLY the provided context. "
        "Cite source attributions when available (e.g., source: <filename> or mission). If the answer is not in the "
        "context, say you don't know and suggest what information is missing. Be concise and factual."
    )

    # Build messages: system, optional conversation history, then context+user question
    messages = []
    messages.append({"role": "system", "content": system_prompt})

    # append provided conversation history (if any)
    if conversation_history:
        for turn in conversation_history:
            # expect dicts like {"role": "user/assistant", "content": "..."}
            if "role" in turn and "content" in turn:
                messages.append({"role": turn["role"], "content": turn["content"]})

    # Add the retrieved context as a system-level context message for grounding
    if context:
        messages.append({"role": "system", "content": f"Context for answering questions:\n{context}"})

    # Finally add the user question
    messages.append({"role": "user", "content": user_message})

    client = OpenAI(api_key=openai_key)

    try:
        resp = client.chat.completions.create(model=model, messages=messages, max_tokens=512, temperature=0.0)
        # New SDK returns choices with message
        if resp and getattr(resp, "choices", None):
            choice = resp.choices[0]
            content = getattr(choice, "message", {}).get("content") if isinstance(getattr(choice, "message", {}), dict) else None
            if not content:
                # fallback to text field
                content = getattr(choice, "text", None)
            return content or ""
    except Exception as e:
        return f"Error generating response: {e}"

    return ""