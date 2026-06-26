import os
import json
import gradio as gr
from openai import OpenAI

with open("config.json") as f:
    config = json.load(f)

ASSISTANT_NAME = config["assistant_name"]
ASSISTANT_INSTRUCTIONS = config["assistant_instructions"]
MODEL = config["model"]
VECTOR_STORE_ID = config["vector_store_id"]

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

assistant = client.beta.assistants.create(
    name=ASSISTANT_NAME,
    instructions=ASSISTANT_INSTRUCTIONS,
    model=MODEL,
    tools=[{"type": "file_search"}],
    tool_resources={"file_search": {"vector_store_ids": [VECTOR_STORE_ID]}},
)

def chat(message, history):
    thread = client.beta.threads.create()

    for user_msg, assistant_msg in history:
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=user_msg,
        )
        if assistant_msg:
            client.beta.threads.messages.create(
                thread_id=thread.id,
                role="assistant",
                content=assistant_msg,
            )

    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=message,
    )

    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )

    messages = client.beta.threads.messages.list(thread_id=thread.id, order="desc")
    for msg in messages.data:
        if msg.role == "assistant":
            text = ""
            for block in msg.content:
                if block.type == "text":
                    text += block.text.value
            return text

    return "I'm sorry, I couldn't generate a response. Please try again!"


demo = gr.ChatInterface(
    fn=chat,
    title=ASSISTANT_NAME,
    description="Your friendly AI tutor — ask me anything about your school material!",
    examples=[
        "Can you help me understand fractions?",
        "What is photosynthesis?",
        "How do I solve 12 + 7?",
    ],
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=5000)
