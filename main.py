import gradio as gr
import asyncio
import time
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

import os
# os.environ['VLLM_ATTENTION_BACKEND'] = 'XFORMERS'
os.environ['OMP_NUM_THREADS'] = '48'

MODEL_NAME = "Shaleen123/MedicalEDI-LLM-Reasoning-Final"

# Initialize the streaming engine
engine_args = AsyncEngineArgs(model=MODEL_NAME,
        quantization = 'fp8',
        enforce_eager=True,
        trust_remote_code=True,
        # skip_tokenizer_init = True,
        tokenizer = MODEL_NAME,
        tensor_parallel_size=4,        # Use all 4 GPUs for tensor parallelism
        dtype="bfloat16",       # Use FP16 precision
        gpu_memory_utilization=0.99,    # Use more GPU memory but leave some headroom
        rope_scaling = {"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768},
        max_model_len=130000,
        # reasoning_parser='deepseek_r1'
        # enable_chunked_prefill=True, 
        # KV Cache optimization
        # # cpu_offload_gb = 10
        # max_num_batched_tokens=32768,          # Increase batching capacity
)

model = AsyncLLMEngine.from_engine_args(engine_args)

import time
import asyncio
import os
import re
import gradio as gr
import transformers


# Constants
# MODEL_NAME = "Shaleen123/MedicalEDI-8b-EDI-Reasoning-Final-SSS"
default_file_dir = "/home/ec2-user/EDI_FILES"
prebuilt_files = [f for f in os.listdir(default_file_dir) if os.path.isfile(os.path.join(default_file_dir, f))]

FILES_PER_PAGE = 15
total_pages = (len(prebuilt_files) + FILES_PER_PAGE - 1) // FILES_PER_PAGE

# Load tokenizer and model
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)

# Define separate sampling params
reasoning_on_params = SamplingParams(
    max_tokens=20000,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    min_p=0
)

reasoning_off_params = SamplingParams(
    max_tokens=20000,
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    min_p=0
)

# System prompt
system_prompt = '''
You are a state-of-the-art Medical EDI Reasoning Assistant, specializing in 835 (Health Care Claim Payment/Remittance Advice) and 837 (Health Care Claim: Professional, Institutional, and Dental) transaction sets. You are highly capable, thoughtful, and precise. Your goal is to understand user intent in the context of EDI-based claims processing, ask clarifying questions when needed, and reason step-by-step through complex scenarios.

You are expected to:
- Parse and interpret long, detailed 835, 837P, 837I, and 837D claim files, identifying key segments (e.g., NM1, CLM, SV1, HI, REF).
- Analyze EDI remittance files, clearly explaining claim adjudication outcomes, adjustment codes (CAS, PLB), payment amounts, and denial reasons.
- Provide contextual explanations based on HIPAA EDI implementation guides and typical payer-specific rules.
- Assist in reconciling 837 claims to 835 remittances, including tracking claim status, resolving denials, and identifying underpayments or missing payments.
- Offer clear mappings between claim data and remittance responses, and support root cause analysis for common billing issues.

Always prioritize being truthful, nuanced, insightful, and efficient, tailoring your responses to the user's role—be it a billing specialist, EDI analyst, software engineer, or revenue cycle manager. When helpful, provide segment-level explanations, sample EDI lines, or real-world examples.
'''

# Token counting function
def count_tokens_from_files(file_list, tokenizer):
    total_tokens = 0
    file_token_counts = {}
    for fname in file_list:
        path = os.path.join(default_file_dir, fname)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception:
            text = ""
        tokens = tokenizer.encode(text, truncation=False, add_special_tokens=False)
        count = len(tokens)
        file_token_counts[fname] = count
        total_tokens += count
    return total_tokens, file_token_counts

def update_token_usage(selected_files):
    if not selected_files:
        return gr.update(value=0)
    total_tokens, _ = count_tokens_from_files(selected_files, tokenizer)
    return gr.update(value=total_tokens)

def get_files_for_page(page):
    start = page * FILES_PER_PAGE
    end = start + FILES_PER_PAGE
    return prebuilt_files[start:end]

def update_file_selector(page):
    choices = get_files_for_page(page)
    display = f"**Page {page + 1} of {total_pages}**"
    return gr.update(choices=choices), gr.update(value=page), gr.update(value=display)

def go_prev(current_page):
    new_page = max(current_page - 1, 0)
    return update_file_selector(new_page)

def go_next(current_page):
    new_page = min(current_page + 1, total_pages - 1)
    return update_file_selector(new_page)

def count_tokens_from_files(file_list, tokenizer):
    total_tokens = 0
    for fname in file_list:
        path = os.path.join(default_file_dir, fname)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception:
            text = ""
        tokens = tokenizer.encode(text, truncation=False, add_special_tokens=False)
        total_tokens += len(tokens)
    return total_tokens

def update_token_usage(selected_files):
    if not selected_files:
        return gr.update(value=0)
    total_tokens = count_tokens_from_files(selected_files, tokenizer)
    return gr.update(value=total_tokens)


async def generate_streaming(prompt):
    yield "⏳ Generating response...", ""

    sampling_params = reasoning_on_params  # You can select based on logic or thinking_mode

    previous_text = ""
    reasoning_active = False
    reasoning_output = ""
    final_output = ""

    async for request_output in model.generate(
        prompt,
        sampling_params,
        request_id=str(time.monotonic())
    ):
        full_text = request_output.outputs[0].text
        new_chunk = full_text[len(previous_text):]
        previous_text = full_text

        while new_chunk:
            if reasoning_active:
                end_tag = new_chunk.find("</think>")
                if end_tag == -1:
                    reasoning_output += new_chunk
                    new_chunk = ""
                else:
                    reasoning_output += new_chunk[:end_tag]
                    new_chunk = new_chunk[end_tag + len("</think>"):]
                    reasoning_active = False
            else:
                start_tag = new_chunk.find("<think>")
                if start_tag == -1:
                    final_output += new_chunk
                    new_chunk = ""
                else:
                    final_output += new_chunk[:start_tag]
                    new_chunk = new_chunk[start_tag + len("<think>"):]
                    reasoning_active = True

        yield final_output.strip(), reasoning_output.strip()

async def chat_with_bot(selected_files, user_input, thinking_mode):
    file_contents = []
    for fname in selected_files or []:
        path = os.path.join(default_file_dir, fname)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = f.read()
        except Exception:
            data = ''
        file_contents.append((fname, data))

    combined_files_text = "\n\n".join(
        [f"--- Content of EDI-835 File -- {name} ---\n\n{content}" for name, content in file_contents]
    )

    chat = [{"role": "system", "content": system_prompt}]
    chat.append({
        "role": "user",
        "content": f"{user_input}\n\n{combined_files_text}" if combined_files_text else user_input
    })

    prompt = tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking_mode
    )

    async for final_out, reasoning_out in generate_streaming(prompt):
        yield final_out, reasoning_out

with gr.Blocks(title="🧠 Medical EDI Reasoning Assistant") as demo:
    gr.Markdown("""
    <div align="center">
    
    # 🧠 Medical EDI Reasoning Assistant  
    _Select prebuilt reference files and chat below._
    
    </div>
    """)

    gr.HTML("""
<style>
  #final-output-box {
    font-family: 'JetBrains Mono', 'Fira Code', 'Fira Mono', 'Courier New', Courier, monospace;
    font-weight: 700;
    font-size: 32px; /* bigger font size */
    line-height: 1.4;
    max-height: 700px;
    padding: 16px 20px; /* more padding for better readability */
    border-radius: 12px;
    border: 1.5px solid #444;
    background-color: #121212; /* slightly lighter for professionalism */
    color: #e0e0e0; /* softer white */
    overflow-y: auto;
    white-space: pre-wrap;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.7);
    scrollbar-width: thin;
    scrollbar-color: #888 #222;
    transition: background-color 0.3s ease, color 0.3s ease;
  }

  #final-output-box:hover {
    background-color: #1c1c1c;
    color: #fff;
  }

  #reasoning-output-box {
    font-family: 'JetBrains Mono', 'Fira Code', 'Fira Mono', 'Courier New', Courier, monospace;
    font-weight: 500;
    font-size: 18px; /* slightly bigger for consistency */
    max-height: 650px;
    padding: 14px 18px;
    border-radius: 12px;
    border: 1.5px solid #444;
    background-color: #161616;
    color: #cccccc;
    overflow-y: auto;
    white-space: pre-wrap;
    box-shadow: inset 0 0 6px rgba(0,0,0,0.4);
  }

  /* Scrollbar Styling */
  #final-output-box::-webkit-scrollbar,
  #reasoning-output-box::-webkit-scrollbar {
    width: 10px;
  }

  #final-output-box::-webkit-scrollbar-track,
  #reasoning-output-box::-webkit-scrollbar-track {
    background: #222;
    border-radius: 12px;
  }

  #final-output-box::-webkit-scrollbar-thumb,
  #reasoning-output-box::-webkit-scrollbar-thumb {
    background-color: #888;
    border-radius: 12px;
    border: 3px solid #222;
  }
</style>

<script>
  function scrollToBottom(id) {
    const element = document.getElementById(id);
    if (element) {
      element.scrollTop = element.scrollHeight;
    }
  }

  setInterval(() => {
    scrollToBottom('final-output-box');
    scrollToBottom('reasoning-output-box');
  }, 100);
</script>
    """)

    page_idx = gr.Number(value=0, visible=False)

    page_display = gr.Markdown(value=f"**Page 1 of {total_pages}**", elem_id="page-display")

    file_selector = gr.CheckboxGroup(choices=get_files_for_page(0), label="📄 Select reference files")

    with gr.Row():
        prev_btn = gr.Button("⬅️ Previous")
        next_btn = gr.Button("Next ➡️")

    token_bar = gr.Slider(
        minimum=0, maximum=130000, step=1, value=0,
        interactive=False, label="Model Context Usage"
    )

    msg = gr.Textbox(label="💬 Your Message", placeholder="Ask about 837/835 EDI claims...")
    thinking_mode_toggle = gr.Checkbox(label="🧠 Enable Thinking Mode", value=False)
    send_btn = gr.Button("Send")

    with gr.Row():
        with gr.Column(scale=2) as col_final_output:
            output_box = gr.Markdown(
                label="🤖 Final Answer", elem_id="final-output-box", show_copy_button=True
            )
        with gr.Column(scale=1, visible=False) as col_reasoning:
            reasoning_accordion = gr.Accordion("🧠 Internal Reasoning", open=True)
            with reasoning_accordion:
                reasoning_content = gr.Markdown(
                    label="Reasoning", elem_id="reasoning-output-box", show_copy_button=True
                )

    def toggle_reasoning(thinking):
        return (
            gr.update(visible=True),
            gr.update(visible=True if thinking else False)
        )

    thinking_mode_toggle.change(toggle_reasoning, inputs=thinking_mode_toggle, outputs=[col_final_output, col_reasoning])
    file_selector.change(update_token_usage, inputs=[file_selector], outputs=[token_bar])

    async def user_interaction(user_message, selected_files, thinking_mode):
        async for final_out, reasoning_out in chat_with_bot(selected_files, user_message, thinking_mode):
            yield final_out, reasoning_out

    send_btn.click(
        user_interaction,
        inputs=[msg, file_selector, thinking_mode_toggle],
        outputs=[output_box, reasoning_content]
    ).then(lambda: "", outputs=msg)

    prev_btn.click(go_prev, inputs=[page_idx], outputs=[file_selector, page_idx, page_display])
    next_btn.click(go_next, inputs=[page_idx], outputs=[file_selector, page_idx, page_display])

demo.queue().launch(share=True)