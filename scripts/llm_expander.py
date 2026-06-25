import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

COMBINED_SYSTEM = (
    "Bạn là trợ lý pháp lý Việt Nam. tasks:\n"
    "1. Viết lại câu hỏi bằng thuật ngữ pháp lý (1 câu duy nhất).\n"
    "2. Viết đoạn ngắn mô tả câu trả lời pháp lý (2-3 câu, trích dẫn điều khoản).\n"
    "Trả lời đúng 2 dòng: dòng 1 = câu viết lại, dòng 2 = đoạn mô tả."
)

_model = None
_tokenizer = None


def load_llm():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
    )
    _model.eval()
    return _model, _tokenizer


def _generate(system_prompt, user_prompt, max_new_tokens=100):
    model, tokenizer = load_llm()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    response = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def expand_and_hyde(query):
    raw = _generate(COMBINED_SYSTEM, query, max_new_tokens=120)
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    expanded = lines[0] if lines else query
    hyde = lines[1] if len(lines) > 1 else expanded
    return expanded, hyde
