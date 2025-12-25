import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Search for CUDA device #;;
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)
model.to(device)

# Example text
text = "This is an example sentence."

# Tokenize and preprocess the text
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
inputs.to(device)

# Inference
with torch.no_grad():
    outputs = model(**inputs)

# Get predicted label
predicted_label = torch.argmax(outputs.logits, dim=1).item()

# Print the result
print("Predicted Label:", predicted_label)