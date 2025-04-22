import torch
from gpt import model, encode, decode, device

def ask_question(question):
    # Convert question to model input
    context = torch.tensor([encode(question)], dtype=torch.long, device=device)
    
    # Generate response
    response = decode(model.generate(context, max_new_tokens=200)[0].tolist())
    return response

# Test questions
test_questions = [
    "What programs does the college offer?",
    "How are the placements?",
    "Tell me about the hostel facilities",
    "What sports facilities are available?",
    "How to get admission?"
]

print("\nTesting DY Patil College AI Model")
print("--------------------------------")
for question in test_questions:
    print(f"\nQ: {question}")
    print(f"A: {ask_question(question)}")
    print("-" * 50)

# Interactive mode625ys ik niki is    a k s i  a is a a  a s
print("\nNow you can ask your own questions!")
print("Type 'quit' to exit")
while True:
    user_question = input("\nYour question: ")
    if user_question.lower() == 'quit':
        break
    print(f"\nAnswer: {ask_question(user_question)}") 