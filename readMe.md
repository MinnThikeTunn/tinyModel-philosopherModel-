# 🧘‍♂️ Philosopher Bot (Fine-Tuned OPT-2.7B)

This is an initial version of a **philosopher-style conversational bot**, fine-tuned using [facebook/opt-2.7b](https://huggingface.co/facebook/opt-2.7b) via parameter-efficient fine-tuning (LoRA).  
It tries to respond in a deep, philosophical tone — but let’s be honest:  
> ⚠️ **It’s pretty inefficient and not very smart. Lots of room for improvement.**

---

## 🚀 Model Details

- **Base Model**: `facebook/opt-2.7b`
- **Quantization**: 4-bit with `bitsandbytes`
- **PEFT**: LoRA with `r=16`, `alpha=32`
- **Training Data**: Instruction-output pairs designed to emulate philosophical reasoning.

---


## 🛠️ Fine-Tuning Tips (How to Improve It)

Here’s what you can do to **make it way smarter and more aligned**:

### 🔁 1. Better Dataset

- Use **larger and richer instruction-output data**, e.g.:
  - [Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
  - [Philosophy Quotes]([https://huggingface.co/datasets/](https://huggingface.co/datasets/datastax/philosopher-quotes)) if available
- Use diverse prompts: quotes, metaphors, paradoxes, Socratic dialogues, etc.

### 🧠 2. Smarter Preprocessing

- Ensure each prompt follows consistent structure like:
  ```txt
  Instruction: What is free will?
  Output: Free will is the capacity of...

  
## 📦 How to Use

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("your-username/philosopher-bot")
model = AutoModelForCausalLM.from_pretrained("your-username/philosopher-bot")

prompt = "What is the meaning of life?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
