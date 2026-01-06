# ✍️ Handwritten Manuscript Recognition (English + Kannada)

A modular implementation of a Swin Transformer–based handwritten manuscript recognition model inspired by **Donut** — supporting **English and Kannada** characters.

This project is designed to be:

- 🔧 configurable  
- 🧩 modular  
- 🚀 easy to train, test, and extend  

---

## 📁 Project Structure

handwritten-recognition/
├── config.yaml
├── requirements.txt
├── train.py
├── inference.py
├── data/
│ ├── init.py
│ ├── tokenizer.py
│ └── dataset.py
├── models/
│ ├── init.py
│ ├── encoder.py
│ ├── decoder.py
│ └── model.py
└── utils/
├── init.py
├── preprocessing.py
├── metrics.py
└── training.py

yaml
Copy code

---

## 🚀 Setup

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

pip install -r requirements.txt

touch data/__init__.py models/__init__.py utils/__init__.py
🏋️ Training
bash
Copy code
python train.py --config config.yaml
Resume training:

bash
Copy code
python train.py --config config.yaml
Monitor logs:

bash
Copy code
tail -f logs/training.log
🔍 Inference
bash
Copy code
python inference.py --checkpoint checkpoints/best_model.pth --image test.png
Batch directory:

bash
Copy code
python inference.py --checkpoint checkpoints/best_model.pth --image_dir images/
Save results:

bash
Copy code
python inference.py --checkpoint checkpoints/best_model.pth --image_dir images/ --output results.txt
Control decoding:

bash
Copy code
python inference.py --checkpoint checkpoints/best_model.pth --image test.png --temperature 0.7 --top_k 5
⚙️ Quick config.yaml Edits
yaml
Copy code
data:
  train_path: "/your/path/here"
  batch_size: 4

training:
  epochs: 100
  learning_rate: 3e-4
💡 Reduce batch_size if GPU runs out of memory.

🐛 Common Fixes
bash
Copy code
touch data/__init__.py models/__init__.py utils/__init__.py
pip install -r requirements.txt
Check dataset path in config.yaml.

If CUDA error → lower batch size.

📊 Metrics
Metric	Meaning	Good	Needs Work
Loss	Model error	< 0.5	> 2.0
Accuracy	Correct outputs	> 90%	< 50%
CER	Character Error Rate	< 5%	> 20%

📂 Important Outputs
bash
Copy code
checkpoints/best_model.pth
logs/training.log
checkpoints/config.yaml
🔄 Typical Workflow
bash
Copy code
pip install -r requirements.txt
nano config.yaml

python train.py --config config.yaml
tail -f logs/training.log

python inference.py --checkpoint checkpoints/best_model.pth --image test.png
🛠 Troubleshooting
bash
Copy code
cat logs/training.log
ls /path/to/dataset/training_images/
nvidia-smi

python -c "from data.tokenizer import CharTokenizer; print('OK')"
python -c "from models.model import HandwrittenDonut; print('OK')"
🎯 Performance Tuning
yaml
Copy code
data:
  batch_size: 16
  num_workers: 4

training:
  epochs: 100
  early_stopping_patience: 20

model:
  decoder_layers: 4
  decoder_heads: 8
💾 Checkpoint Management
bash
Copy code
# Best model
checkpoints/best_model.pth

# Resume specific checkpoint
# resume_from: "checkpoints/checkpoint_epoch_20.pth"

# Clean extras
rm checkpoints/checkpoint_epoch_*.pth
✅ Notes
Always run inference using best_model.pth

Keep __init__.py files (imports break otherwise)

Validate on a separate dataset

Use nvidia-smi to watch GPU
