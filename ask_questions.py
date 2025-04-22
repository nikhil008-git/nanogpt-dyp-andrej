import torch
from gpt import model, encode, decode, device

print("\nDY Patil College of Engineering AI Assistant")
print("Type 'quit' to exit")
print("\nYou can ask questions about:")
print("1. Academic Programs")
print("2. Campus Facilities")
print("3. Admissions")
print("4. Placements")
print("5. Faculty")
print("6. Campus Life")
print("7. Infrastructure")
print("8. Hostel Facilities")
print("9. Sports Facilities")
print("10. Transportation")
print("\nExample questions:")
print("- What programs does the college offer?")
print("- How are the placements?")
print("- Tell me about the hostel facilities")
print("- What sports facilities are available?")
print("- How to get admission?")

while True:
    user_input = input("\nYour question about DY Patil College: ")
    if user_input.lower() == 'quit':
        break
    
    # Convert user input to model input
    context = torch.tensor([encode(user_input)], dtype=torch.long, device=device)
    
    # Generate response
    response = decode(model.generate(context, max_new_tokens=200)[0].tolist())
    print("\nAI:", response) 