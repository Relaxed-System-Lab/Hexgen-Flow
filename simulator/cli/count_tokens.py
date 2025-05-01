import os
from transformers import AutoTokenizer
from huggingface_hub import login
# Load the tokenizer for llama3.1-70b
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B-Instruct")

# Path to the templates folder
templates_folder = "./templates"

# Iterate through all .txt files in the folder
for root, _, files in os.walk(templates_folder):
    for file in files:
        if file.endswith(".txt"):
            file_path = os.path.join(root, file)
            
            # Read the content of the file
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Tokenize the content and calculate token length
            tokens = tokenizer.encode(content, add_special_tokens=False)
            token_length = len(tokens)
            
            # Print the token length for the file
            print(f"File: {file_path}, Token Length: {token_length}")
