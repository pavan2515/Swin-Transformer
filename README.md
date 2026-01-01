# Swin-TransformerHandwritten Donut — English + Kannada OCR (Swin Transformer)

This project implements a lightweight OCR model for handwritten text using:

Swin Transformer as the visual encoder

A custom Transformer decoder

A character-level tokenizer supporting English + Kannada

Built-in manuscript preprocessing (CLAHE, denoising, adaptive thresholding, morphology)

The model learns to predict the label name of each image (one character token at a time) based on folder names.

✨ Features

✔️ Works on handwritten manuscripts

✔️ Internal noise removal & contrast enhancement

✔️ Character tokenizer (English + Kannada)

✔️ Simple dataset structure

✔️ Trainable end-to-end

✔️ Easy inference on new images

📂 Dataset Structure

Place your images like this:

dataset/
 └── training_images/
      ├── apple/
      │    ├── img1.jpg
      │    ├── img2.png
      │
      ├── ಅಮ್ಮ/
      │    ├── img3.jpg
      │    ├── img4.png
      │
      └── hello/
           ├── img5.jpg
           ├── img6.png


👉 Folder name = label text (what the model learns to predict)

🔧 Installation

Install dependencies:

pip install transformers timm torchvision pillow opencv-python

▶️ Training

Run the script:

python train.py


What happens during training:

Images are resized to 224×224

Preprocessing improves readability (denoise, CLAHE, threshold)

Images go through Swin Transformer encoder

Transformer decoder predicts characters

Cross-Entropy loss updates weights

Training runs for 10 epochs by default.

🔍 Inference (Prediction)

Use the predict() function:

predict("/content/test_image.jpg")


Example output:

Prediction: hello

🧠 Model Architecture
Component	Role
Swin Transformer	Extracts visual features
Transformer Decoder	Generates text tokens
Character Tokenizer	Maps English + Kannada characters
Manuscript Preprocessing	Improves readability
🧾 Tokenizer Details

Special tokens:

Token	Meaning
<pad>	padding
<s>	start
</s>	end
<unk>	unknown

Both English letters and Kannada characters are supported.

✏️ Preprocessing Pipeline

The script automatically performs:

Grayscale conversion

Noise removal (fastNlMeans)

CLAHE contrast boost

Adaptive thresholding

Morphological opening (remove artifacts)

This improves OCR accuracy on noisy manuscripts.

⚙️ Hyperparameters
Parameter	Value
Optimizer	AdamW
LR	3e-4
Batch Size	8
Loss	CrossEntropy (ignore pad)
Epochs	10
💡 Notes & Tips

More data = better accuracy

Keep handwriting centered & cropped

Balance classes (avoid one label dominating)

Increase epochs if loss is still high

📌 Future Improvements (Optional)

Beam search decoding

Multi-line text handling

Dataset augmentation

Save / load trained weights

🛠️ Requirements

Python 3.8+

GPU recommended (but CPU works)

PyTorch + Transformers

📜 License

Use freely for research, learning, and educational projects.
