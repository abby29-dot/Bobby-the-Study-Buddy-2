import os
import json
import mimetypes
import gradio as gr
from openai import OpenAI

with open("config.json") as f:
    config = json.load(f)

ASSISTANT_NAME = config["assistant_name"]
ASSISTANT_INSTRUCTIONS = config["assistant_instructions"]
MODEL = config["model"]
VECTOR_STORE_ID = config["vector_store_id"]

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

assistant = client.beta.assistants.create(
    name=ASSISTANT_NAME,
    instructions=ASSISTANT_INSTRUCTIONS,
    model=MODEL,
    tools=[{"type": "file_search"}],
    tool_resources={"file_search": {"vector_store_ids": [VECTOR_STORE_ID]}},
)


def upload_file(path):
    ext = os.path.splitext(path)[1].lower()
    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    with open(path, "rb") as f:
        response = client.files.create(file=(os.path.basename(path), f, mime), purpose="assistants")
    return response.id, ext


def build_user_content(text, file_paths):
    content = []
    attachments = []

    for path in file_paths:
        file_id, ext = upload_file(path)
        if ext in IMAGE_TYPES:
            content.append({"type": "image_file", "image_file": {"file_id": file_id}})
        else:
            attachments.append({"file_id": file_id, "tools": [{"type": "file_search"}]})

    if text:
        content.append({"type": "text", "text": text})

    if not content:
        content.append({"type": "text", "text": "(file uploaded)"})

    return content, attachments


def chat(message, history):
    if isinstance(message, dict):
        text = message.get("text") or ""
        files = message.get("files") or []
    else:
        text = message
        files = []

    thread = client.beta.threads.create()

    for entry in history:
        if isinstance(entry, dict):
            role = entry.get("role")
            raw = entry.get("content")
            if role not in ("user", "assistant") or not raw:
                continue
            if isinstance(raw, list):
                safe = [b for b in raw if isinstance(b, dict) and b.get("type") in ("text", "image_file", "image_url")]
                if not safe:
                    continue
                content = safe
            else:
                content = str(raw)
            client.beta.threads.messages.create(
                thread_id=thread.id,
                role=role,
                content=content,
            )
        else:
            user_msg, assistant_msg = entry
            if user_msg:
                client.beta.threads.messages.create(
                    thread_id=thread.id,
                    role="user",
                    content=str(user_msg),
                )
            if assistant_msg:
                client.beta.threads.messages.create(
                    thread_id=thread.id,
                    role="assistant",
                    content=str(assistant_msg),
                )

    content, attachments = build_user_content(text, files)

    msg_kwargs = {"thread_id": thread.id, "role": "user", "content": content}
    if attachments:
        msg_kwargs["attachments"] = attachments

    client.beta.threads.messages.create(**msg_kwargs)

    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )

    messages = client.beta.threads.messages.list(thread_id=thread.id, order="desc")
    for msg in messages.data:
        if msg.role == "assistant":
            text_out = ""
            for block in msg.content:
                if block.type == "text":
                    text_out += block.text.value
            return text_out

    return "I'm sorry, I couldn't generate a response. Please try again!"


demo = gr.ChatInterface(
    fn=chat,
    title=ASSISTANT_NAME,
    description="Your friendly AI tutor — ask me anything about your school material! You can also upload files (PDFs, images, documents) for Bobby to read.",
    multimodal=True,
    examples=[
        {"text": "Can you help me understand fractions?"},
        {"text": "What is photosynthesis?"},
        {"text": "How do I solve 12 + 7?"},
    ],
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=5000)
