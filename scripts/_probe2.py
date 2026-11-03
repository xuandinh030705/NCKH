import os, glob, sys, time, json
venv = os.path.dirname(os.path.dirname(os.path.abspath(sys.executable)))
sp = os.path.join(venv, "Lib", "site-packages")
for d in glob.glob(os.path.join(sp, "nvidia", "*", "bin")):
    os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]
import llama_cpp
from llama_cpp import Llama
from jinja2 import Template
import jinja2.exceptions

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "models", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
llm = Llama(model_path=MODEL, n_ctx=2048, n_gpu_layers=-1, n_batch=512, verbose=False)
tpl = llm.metadata["tokenizer.chat_template"]
print("HAS enable_thinking:", "enable_thinking" in tpl)
sysmsg = ("Bạn là một gia sư dạy Toán tiểu học tận tình, nói năng cụ thể và rõ ràng vào bài toán.\n"
          "QUY TẮC SƯ PHẠM BẮT BUỘC (TUYỆT ĐỐI TUÂN THỦ):\n"
          "1. Xưng hô: 'thầy/cô' - 'em'.\n"
          "2. CẤM TUYỆT ĐỐI: Không dùng từ 'phương trình', 'hai vế', 'chuyển vế'.\n"
          "6. KHÔNG đưa ra đáp số cuối cùng cho đến khi học sinh tự tính ra.")
msgs = [{"role": "system", "content": sysmsg},
        {"role": "user", "content": "Tìm x, biết: x * 3 - 4 = 11"}]
t = Template(tpl)
try:
    prompt = t.render(messages=msgs, add_generation_prompt=True, enable_thinking=False, bos_token="", eos_token="")
except jinja2.exceptions.TemplateError as e:
    print("no enable_thinking ->", e)
    prompt = t.render(messages=msgs, add_generation_prompt=True, bos_token="", eos_token="")
print("PROMPT>>>", repr(prompt[-300:]))
t0 = time.time(); ttft = None; chunks = []
for out in llm.create_completion(prompt=prompt, max_tokens=300, temperature=0.25, top_p=0.85,
                                repeat_penalty=1.15, stop=["<|im_end|>", "<|endoftext|>"], stream=True):
    c = out["choices"][0]["text"]
    if c and ttft is None:
        ttft = time.time() - t0
    chunks.append(c)
total = time.time() - t0
text = "".join(chunks)
n = len(llm.tokenize(text.encode(), add_bos=False))
print("TTFT", round(ttft, 2), "total", round(total, 2), "tokens", n, "tok/s", round(n / total, 1))
print("TEXT>>>", text)
