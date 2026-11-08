"""下載實驗要用的模型到 Hugging Face 快取，並顯示下載進度。

用法：python src/download_models.py            （下載全部）
     python src/download_models.py bertimbau-base mbert
"""
import sys

from huggingface_hub import snapshot_download

MODELS = {
    "bertimbau-base": "neuralmind/bert-base-portuguese-cased",    # 約 0.4 GB
    "bertimbau-large": "neuralmind/bert-large-portuguese-cased",  # 約 1.3 GB
    "mbert": "google-bert/bert-base-multilingual-cased",          # 約 0.7 GB
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",                     # 約 0.9 GB（RQ2 LLM）
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",                     # 約 2.9 GB（RQ2 LLM）
    "qwen-3b": "Qwen/Qwen2.5-3B-Instruct",                         # 約 6.2 GB（加深實驗）
}
# 只下載一種權重格式與 tokenizer，避免重複下載 TensorFlow／Flax 權重
PATTERNS = ["*.json", "*.txt", "pytorch_model.bin", "model.safetensors", "model-*.safetensors"]  # 大模型的權重會分成多個檔

names = sys.argv[1:] or [k for k in MODELS if not k.startswith("qwen")]
for name in names:
    repo = MODELS[name]
    print(f"\n=== {name}  ({repo}) ===", flush=True)
    path = snapshot_download(repo, allow_patterns=PATTERNS)
    print(f"完成：{path}", flush=True)
print("\n全部模型都已下載。")
