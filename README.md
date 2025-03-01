# Tiny LLM with Lightning AI

A minimal implementation of a Tiny LLM (Lightweight Language Model) using PyTorch Lightning. This project demonstrates how to build, train, and optimize a small-scale Transformer model from scratch.

## 🚀 Features
- Transformer-based language model with PyTorch & Lightning AI
- Small & efficient architecture for low-resource environments
- Uses Tiny Shakespeare dataset for training
- Optimized training with Lightning Trainer
- Supports GPU acceleration

## 📂 Project Structure
```
📦 tiny-llm
├── data/               # Dataset storage
├── models/             # Transformer model implementation
├── notebooks/          # Jupyter Notebooks for experiments
├── scripts/            # Training & evaluation scripts
├── checkpoints/        # Saved model checkpoints
├── README.md           # Project documentation
└── requirements.txt    # Dependencies
```

## 🔧 Installation
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/your-username/tiny-llm.git
cd tiny-llm
```

### 2️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

## 📊 Training the Model
Run the training script:
```sh
python scripts/train.py --epochs 10 --batch_size 32 --lr 1e-3
```

For GPU training, use:
```sh
python scripts/train.py --accelerator gpu --devices 1
```

## 📈 Model Evaluation
Run the evaluation script:
```sh
python scripts/evaluate.py --checkpoint checkpoints/tiny_llm.pth
```

## 🔥 Inference
Use the trained model to generate text:
```python
from models.tiny_transformer import TinyTransformer
import torch

# Load model
model = TinyTransformer.load_from_checkpoint("checkpoints/tiny_llm.pth")
model.eval()

# Sample input
input_text = "Once upon a time"
tokens = tokenizer.encode(input_text, return_tensors="pt")
output = model(tokens)
print(output)
```

## 🛠 Optimizations
- **Reduce Model Size** → Fewer layers, smaller hidden size
- **Quantization** → `torch.quantization.quantize_dynamic()`
- **LoRA / Flash Attention** → Speed up training
- **Dataset Expansion** → Fine-tune on custom text corpora

## 📜 License
AppCache licence. See `LICENSE` for details.

## 🙌 Acknowledgments
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)
- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT)
- [Hugging Face Datasets](https://huggingface.co/datasets/)

## ⭐ Contribute
Pull requests are welcome! If you find issues, open an [issue](https://github.com/your-username/tiny-llm/issues).

### 📬 Contact
For questions, reach out via GitHub Issues or email: `your-email@example.com`

