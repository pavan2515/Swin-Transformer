✍️ Handwritten Manuscript Recognition (English + Kannada)

A modular implementation of a Swin-Transformer–based model inspired by Donut, designed for handwritten text recognition in English and Kannada.

This repo is structured to be clean, configurable, and easy to extend — whether you’re experimenting or moving toward production.

📁 Project Structure
handwritten-recognition/
├── config.yaml              # Main configuration
├── requirements.txt         # Dependencies
├── train.py                 # Training script
├── inference.py             # Inference script
├── data/
│   ├── __init__.py
│   ├── tokenizer.py
│   └── dataset.py
├── models/
│   ├── __init__.py
│   ├── encoder.py
│   ├── decoder.py
│   └── model.py
└── utils/
    ├── __init__.py
    ├── preprocessing.py
    ├── metrics.py
    └── training.py

🚀 Setup
# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Ensure packages import correctly
touch data/__init__.py models/__init__.py utils/__init__.py

🏋️ Training
# Basic training
python train.py --config config.yaml

# Resume training (set resume_from in config.yaml)
python train.py --config config.yaml

# Monitor logs
tail -f logs/training.log

🔍 Inference
# Single image
python inference.py --checkpoint checkpoints/best_model.pth --image test.png

# Batch directory
python inference.py --checkpoint checkpoints/best_model.pth --image_dir images/

# Save results to file
python inference.py --checkpoint checkpoints/best_model.pth --image_dir images/ --output results.txt

# Control decoding behavior
python inference.py --checkpoint checkpoints/best_model.pth --image test.png --temperature 0.7 --top_k 5

⚙️ Quick config.yaml Tweaks
data:
  train_path: "/your/path/here"   # ← Set dataset path
  batch_size: 4                   # Lower if GPU OOM

training:
  epochs: 100
  learning_rate: 3e-4


Tip:
Increase LR to train faster, decrease LR for more stable training.

🐛 Common Issues & Fixes
# Missing imports
touch data/__init__.py models/__init__.py utils/__init__.py

# CUDA OOM
# Reduce batch_size in config.yaml

# Dataset not found
# Check path in config.yaml

# Missing dependencies
pip install -r requirements.txt

📊 Metrics Explained
Metric	Meaning	Good	Needs Work
Loss	Model error	< 0.5	> 2.0
Accuracy	Correct outputs	> 90%	< 50%
CER	Character Error Rate	< 5%	> 20%
📂 Important Outputs
checkpoints/best_model.pth   # Best saved model
logs/training.log            # Training history
checkpoints/config.yaml      # Config used when training

💡 Tips

1️⃣ Always run inference using best_model.pth
2️⃣ If something breaks — check logs first
3️⃣ Start with a small dataset to test pipelines
4️⃣ Watch GPU usage with:

nvidia-smi


5️⃣ Lower batch_size when memory is low

🔄 Typical Workflow
pip install -r requirements.txt
touch data/__init__.py models/__init__.py utils/__init__.py

nano config.yaml   # Set dataset path

python train.py --config config.yaml
tail -f logs/training.log

python inference.py --checkpoint checkpoints/best_model.pth --image test.png

🛠 Troubleshooting
cat logs/training.log
ls /path/to/dataset/training_images/
nvidia-smi

python -c "from data.tokenizer import CharTokenizer; print('OK')"
python -c "from models.model import HandwrittenDonut; print('OK')"

cat config.yaml

🎯 Performance Tuning
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
# Use best model
checkpoints/best_model.pth

# Resume from specific checkpoint
# config.yaml:
# resume_from: "checkpoints/checkpoint_epoch_20.pth"

# Clean up old checkpoints
rm checkpoints/checkpoint_epoch_*.pth

📣 Notes

Keep your config versioned

Always validate on a held-out dataset

Don’t delete __init__.py files — imports will break
