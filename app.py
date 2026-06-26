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


def respond(message, chat_history, thread_id):
    if isinstance(message, dict):
        text = message.get("text") or ""
        files = message.get("files") or []
    else:
        text = str(message)
        files = []

    if not text and not files:
        return gr.MultimodalTextbox(value=None), chat_history, thread_id

    if thread_id is None:
        thread = client.beta.threads.create()
        thread_id = thread.id

    content, attachments = build_user_content(text, files)

    msg_kwargs = {"thread_id": thread_id, "role": "user", "content": content}
    if attachments:
        msg_kwargs["attachments"] = attachments

    client.beta.threads.messages.create(**msg_kwargs)

    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread_id,
        assistant_id=assistant.id,
    )

    response_text = "I'm sorry, I couldn't generate a response. Please try again!"
    messages = client.beta.threads.messages.list(thread_id=thread_id, order="desc")
    for msg_obj in messages.data:
        if msg_obj.role == "assistant":
            text_out = ""
            for block in msg_obj.content:
                if block.type == "text":
                    text_out += block.text.value
            response_text = text_out
            break

    display_user = text if text else f"📎 {len(files)} file(s) uploaded"
    chat_history = chat_history + [
        (display_user, response_text),
    ]

    return gr.MultimodalTextbox(value=None), chat_history, thread_id


def clear_chat():
    return [], None


with gr.Blocks(title=ASSISTANT_NAME) as demo:
    gr.Markdown(f"# {ASSISTANT_NAME}")
    gr.Markdown(
        "Your friendly AI tutor — ask me anything about your school material! "
        "You can also upload files (PDFs, images, documents) for Bobby to read."
    )

    thread_state = gr.State(value=None)

    chatbot = gr.Chatbot(
        height=480,
        show_label=False,
    )

    msg_input = gr.MultimodalTextbox(
        placeholder="Type a message or attach a file…",
        show_label=False,
        file_count="multiple",
    )

    clear_btn = gr.Button("🗑️ Clear conversation", size="sm", variant="secondary")

    gr.Examples(
        examples=[
            {"text": "Can you help me understand fractions?"},
            {"text": "What is photosynthesis?"},
            {"text": "How do I solve 12 + 7?"},
        ],
        inputs=msg_input,
    )

    msg_input.submit(
        respond,
        inputs=[msg_input, chatbot, thread_state],
        outputs=[msg_input, chatbot, thread_state],
    )

    clear_btn.click(clear_chat, outputs=[chatbot, thread_state])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=5000)
