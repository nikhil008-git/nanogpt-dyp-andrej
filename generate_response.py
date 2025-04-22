import torch
from gpt import model, encode, decode, device

# Set the model to evaluation mode (so it doesn't perform training)
model.eval()

# Ask your question
context = torch.tensor([encode('What programs does the college offer?')], dtype=torch.long, device=device)

# Generate a response (without training logs)
response = decode(model.generate(context, max_new_tokens=200)[0].tolist())
print(response)
