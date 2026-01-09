import torch

# Training Hyperparameters
MODEL_CHECKPOINT = "distilbert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-5

# O = Outside, B-KEY = Beginning of Keyword, I-KEY = Inside Keyword
LABEL2ID = {"O": 0, "B-KEY": 1, "I-KEY": 2}
ID2LABEL = {0: "O", 1: "B-KEY", 2: "I-KEY"}

# Device configuration (Auto-detect)
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps") 
    else:
        return torch.device("cpu")