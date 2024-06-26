#  ****  AI that can generate accurate and detailed captions for an image  ****
#  **************     by MAHIR MITTAL    *********************

from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image

# Load the pre-trained model
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Load the feature extractor and tokenizer
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

def load_image(image_path):
    return Image.open(image_path).convert("RGB")

def preprocess_image(image):
    return feature_extractor(images=image, return_tensors="pt").pixel_values

def generate_caption(model, pixel_values, tokenizer):
    # Generate captions
    output_ids = model.generate(pixel_values, max_length=16, num_beams=4, num_return_sequences=1, early_stopping=True)
    
    # Decode the generated captions to text
    captions = [tokenizer.decode(output_id, skip_special_tokens=True) for output_id in output_ids]
    return captions[0]

# Load and preprocess the image
image_path = "image1.jpg"
image = load_image(image_path)
pixel_values = preprocess_image(image)

# Generate the caption
caption = generate_caption(model, pixel_values, tokenizer)
print("Generated Caption:", caption)
