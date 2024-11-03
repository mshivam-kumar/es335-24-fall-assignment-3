import streamlit as st
import torch
import torch.nn as nn
import requests

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download and read the dataset
url = "https://cs.stanford.edu/people/karpathy/char-rnn/warpeace_input.txt"
response = requests.get(url)

# Write the dataset to a file
with open("names-long.txt", "w", encoding='utf-8') as file:
    file.write(response.text)

# Read the text file
with open('names-long.txt', 'r') as file:
    text = file.read()

# Process the text
words = text.lower().strip().split()
words = [word for word in words if 0 < len(word) < 10 and word.isalpha()]

# Use the trained vocabulary size to avoid mismatch
trained_vocab_size = 12076
unique_words = sorted(set(words))
stoi = {w: i + 1 for i, w in enumerate(unique_words[:trained_vocab_size - 1])}  # limit to trained vocab size
stoi['.'] = 0  # End token
itos = {i: w for w, i in stoi.items()}

# Define the RNN model
class NextWord(nn.Module):
    def __init__(self, block_size, vocab_size, emb_dim, hidden_size):
        super().__init__()
        self.block_size = block_size  # Add this line to store block_size as an attribute
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.RNN(emb_dim, hidden_size, batch_first=True)
        self.lin = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h=None):
        x = self.emb(x)
        out, h = self.rnn(x, h)
        x = out[:, -1, :]
        x = self.lin(x)
        return x, h

def load_trained_model(params):
    # Initialize the model with the specified parameters
    model = NextWord(
        block_size=params["block_size"], 
        vocab_size=trained_vocab_size, 
        emb_dim=params["embedding_dim"], 
        hidden_size=params["hidden_size"]
    ).to(device)
    
    # Load the entire checkpoint
    checkpoint = torch.load(params["model_path"], map_location=device)
    print(checkpoint.keys())  # Check what keys are present in the checkpoint

    # Adjust the loading based on the checkpoint structure
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)  # Use this line if the key exists
    else:
        model.load_state_dict(checkpoint, strict=False)  # Or this if the state dict is the entire checkpoint
    
    # Set the model to evaluation mode
    model.eval()
    return model



# Function to generate text
def generate_text(model, context, itos, stoi, k=5, max_len=20):
    context_words = context.lower().strip().split()
    context_indices = [stoi.get(w, 0) for w in context_words]

    if len(context_indices) < model.block_size:
        context_indices = [0] * (model.block_size - len(context_indices)) + context_indices

    context_tensor = torch.tensor(context_indices[-model.block_size:]).unsqueeze(0).to(device)
    generated = context_indices.copy()  
    h = None

    for _ in range(max_len):
        out, h = model(context_tensor, h)
        prob = torch.softmax(out, dim=-1)
        ix = torch.multinomial(prob, num_samples=1).item()

        if ix == stoi['.']:  # End if the end token is generated
            break

        generated.append(ix)
        context_tensor = torch.tensor([[ix]]).to(device)  

    generated_words = [itos.get(ix, '<UNK>') for ix in generated]
    return ' '.join(generated_words)

# Streamlit UI
st.title('Next Word Prediction App')

# User input for the text context
context = st.text_input("Input text for prediction")

# Add sliders for user inputs
activation_function = st.selectbox("Select Activation Function", ["ReLU", "Tanh"])
context_length = st.selectbox("Select Context Length", [5, 10, 15])
embedding_size = st.selectbox("Select Embedding Size", [64, 128])


# Set model parameters based on user selections
model_params = {
    "block_size": context_length,
    "embedding_dim": embedding_size,
    "hidden_size": 1024,
    "activation_function": activation_function.lower(),  # Use lowercase for nn.RNN
    "model_path": f"relu_{embedding_size}_{context_length}.pth"  # Update this to the actual model path
}


# Load the model based on selected parameters
model = load_trained_model(model_params)

# Generate and display text when user inputs a text
if st.button('Generate Next Words'):
    generated_text = generate_text(model, context, itos, stoi, k=5)
    st.write("Generated Text:")
    st.write(generated_text)
