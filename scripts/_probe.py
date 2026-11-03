import os, glob, sys, time
venv = os.path.dirname(os.path.dirname(os.path.abspath(sys.executable)))
sp = os.path.join(venv, "Lib", "site-packages")
for d in glob.glob(os.path.join(sp, "nvidia", "*", "bin")):
    os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]
    try: os.add_dll_directory(d)
    except Exception: pass
import llama_cpp
from llama_cpp import Llama

MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
t0 = time.time()
llm = Llama(model_path=MODEL, n_ctx=2048, n_gpu_layers=-1, n_batch=512, verbose=False)
print("load_s", round(time.time() - t0, 2))
md = llm.metadata
a = md.get("general.architecture")
print("arch", a, "layers", md.get(f"{a}.block_count"), "embd", md.get(f"{a}.embedding_length"),
      "heads", md.get(f"{a}.attention.head_count"), "kvheads", md.get(f"{a}.attention.head_count_kv"))
print("n_tokens", llm.n_tokens, "size", os.path.getsize(MODEL) / 1e9)
tpl = md.get("tokenizer.chat_template") or ""
print("TPL_LEN", len(tpl))
print(tpl[:3000])

