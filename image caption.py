# TASK3
#image caption
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image

# Load model
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Your image path (already correct)
image_path = r"C:\Users\HP\Downloads\pexels-muhammed-mahsum-tunc-859110584-35389656.jpg"

# Open image
image = Image.open(image_path)

if image.mode != "RGB":
    image = image.convert(mode="RGB")

# Process image
pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
pixel_values = pixel_values.to(device)

# Generate caption
output_ids = model.generate(pixel_values, max_length=16, num_beams=4)

caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("\nGenerated Caption:", caption)
