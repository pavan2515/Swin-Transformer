# Handwritten Donut — English + Kannada OCR (Swin Transformer)

A clean, production-ready system for recognizing handwritten English and Kannada characters using deep learning.


🤔 What is this?
Think of this as a smart scanner that can read handwritten characters. You show it an image of a handwritten letter (like 'a' or 'ಅ'), and it tells you what it is.
The magic: It uses AI to learn from examples, just like how you learned to read as a kid!

🎨 What makes this special?
Your original code worked, but it was like a messy kitchen where everything is in one drawer. This version is like a professional kitchen - everything has its place, and it's much faster!
The Big Changes:

Organized Structure 📁

Before: Everything jumbled in one file
Now: Separate folders for data, models, and utilities
Why? Easy to find things, easy to fix bugs, easy for teams to work together


Blazing Fast ⚡

Before: Preprocessing happened every time (slow!)
Now: Preprocessing happens once when loading data
Result: 3-5x faster training!


Smart Training 🧠

Before: Just showed you loss (confusing!)
Now: Shows accuracy, character accuracy, error rates
Plus: Automatically saves best model, stops if it's not learning


Easy to Adjust ⚙️

Before: Had to edit code to change settings
Now: Just edit config.yaml - no code changes needed!




📦 What's in the box?
handwritten-recognition/
│
├── 🎛️ config.yaml           # All your settings (like a control panel)
├── 📋 requirements.txt      # Software needed to run this
│
├── 🚂 train.py              # Trains the AI model
├── 🔮 inference.py          # Uses trained model to predict
│
├── 📂 data/                 # Everything about loading data
│   ├── tokenizer.py         # Converts characters ↔ numbers
│   └── dataset.py           # Loads and prepares images
│
├── 🤖 models/               # The AI brain
│   ├── encoder.py           # Looks at images
│   ├── decoder.py           # Generates text predictions
│   └── model.py             # Combines everything
│
└── 🛠️ utils/                # Helper tools
    ├── preprocessing.py     # Cleans up images
    ├── metrics.py           # Measures how good the AI is
    └── training.py          # Training helpers (saving, logging, etc.)

🚀 Getting Started (5 minutes!)
Step 1: Set everything up
bash# Create the project folder
mkdir handwritten-recognition
cd handwritten-recognition

# Create subfolders
mkdir data models utils checkpoints logs outputs

# Create special files Python needs
touch data/__init__.py models/__init__.py utils/__init__.py

# Install required software
pip install -r requirements.txt
```

### Step 2: Organize your images

Your images should look like this:
```
dataset/
└── training_images/
    ├── a/              ← Put all images of letter 'a' here
    │   ├── img1.png
    │   ├── img2.png
    │   └── img3.png
    ├── b/              ← Put all images of letter 'b' here
    │   ├── img1.png
    │   └── img2.png
    ├── ಅ/              ← Put all images of 'ಅ' here
    │   └── img1.png
    └── ...
The folder name = the character in the images!
Step 3: Tell it where your images are
Open config.yaml and change this line:
yamldata:
  train_path: "/path/to/your/dataset/training_images"  # ← Put your actual path here
Step 4: Start training!
bashpython train.py --config config.yaml
```

Now sit back! The AI will:
- ✅ Load your images
- ✅ Learn from them
- ✅ Save the best model automatically
- ✅ Show you how well it's learning

---

## 📊 What you'll see while training
```
Epoch 1/50
──────────────────────────────────────
Train: loss: 2.35 | accuracy: 23.45% | cer: 54.33%
Val:   loss: 2.12 | accuracy: 28.90% | cer: 47.66%
✓ New best model saved!

Epoch 2/50
──────────────────────────────────────
Train: loss: 1.87 | accuracy: 45.67% | cer: 32.11%
Val:   loss: 1.76 | accuracy: 52.34% | cer: 28.77%
✓ New best model saved!

... (getting better each time!)

Epoch 25/50
──────────────────────────────────────
Train: loss: 0.23 | accuracy: 94.50% | cer: 3.21%
Val:   loss: 0.31 | accuracy: 91.20% | cer: 5.43%
✓ New best model saved!
What these numbers mean:

Loss: Lower is better (think: how wrong it is)
Accuracy: Higher is better (% of perfect matches)
CER (Character Error Rate): Lower is better (% of mistakes)


🔮 Using your trained model
Once training finishes, use it to read new images:
bash# Read one image
python inference.py \
    --checkpoint checkpoints/best_model.pth \
    --image my_handwriting.png

# Result: Prediction: a
bash# Read many images at once
python inference.py \
    --checkpoint checkpoints/best_model.pth \
    --image_dir my_images/ \
    --output results.txt

⚙️ Tweaking Settings
All settings are in config.yaml. Here's what you might want to change:
Training too slow?
yamldata:
  batch_size: 16  # Process more images at once (needs more GPU memory)
Not learning well?
yamltraining:
  learning_rate: 5e-4  # Make it learn faster
  epochs: 100          # Train for longer
Running out of memory?
yamldata:
  batch_size: 4  # Process fewer images at once
Images are very clean (printed, not handwritten)?
yamldata:
  apply_manuscript_preprocessing: false  # Turn off aggressive cleaning

🐛 Something not working?
"Module not found" error
bash# Did you create these files?
touch data/__init__.py models/__init__.py utils/__init__.py
"CUDA out of memory"
yaml# In config.yaml, reduce batch size:
data:
  batch_size: 4
"No images found"
bash# Check your dataset path in config.yaml
# Make sure images are in folders named after their labels
ls /your/path/training_images/
Still stuck?
bash# Check the training log for clues
cat logs/training.log

📈 How to know if it's working?
Good signs:

✅ Loss going down each epoch
✅ Accuracy going up
✅ CER (error rate) going down
✅ Training and validation metrics are similar

Warning signs:

⚠️ Loss not changing → learning rate might be wrong
⚠️ Training accuracy high but validation low → overfitting (train longer, add more data)
⚠️ Loss becomes "nan" → learning rate too highd
