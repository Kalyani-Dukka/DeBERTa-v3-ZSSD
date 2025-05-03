import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DebertaV2Tokenizer, DebertaV2Config, DebertaV2Model, AdamW, get_scheduler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

# Set up GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# File paths
train_file = r"subtaskA\mixed\raw_train_all_onecol.csv"
valid_file = r"subtaskA\mixed\raw_val_all_onecol.csv"
test_files = [r"subtaskA\claim\raw_test_all_onecol.csv", r"subtaskA\noun_phrase\raw_test_all_onecol.csv", r"subtaskA\mixed\raw_test_all_onecol.csv"]
model_save_path = "./saved_model"
output_root = "./results"

# Disable W&B logging
os.environ["WANDB_DISABLED"] = "true"

# Load tokenizer
tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")

# Custom Dataset class
class StanceDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        text, target = self.texts[idx]
        encoding = self.tokenizer(
            text, target,
            padding="max_length",
            truncation="only_first",
            max_length=128,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": self.labels[idx]
        }

# Load datasets
def load_dataset(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None, None
    df = pd.read_csv(file_path)
    df.dropna(subset=["Text", "Target 1", "Stance 1"], inplace=True)
    label_map = {"FAVOR": 0, "AGAINST": 1, "NONE": 2}
    df["stance_label"] = df["Stance 1"].map(label_map)
    texts = df[["Text", "Target 1"]].to_numpy()
    labels = df["stance_label"].to_numpy()
    return texts, labels

# Load train and validation data
train_texts, train_labels = load_dataset(train_file)
valid_texts, valid_labels = load_dataset(valid_file)

# Create dataset objects
train_dataset = StanceDataset(train_texts, train_labels, tokenizer)
valid_dataset = StanceDataset(valid_texts, valid_labels, tokenizer)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

# Define the classifier
class DebertaClassifier(nn.Module):
    def __init__(self, num_labels, model_name="microsoft/deberta-v3-base", dropout=0.3):
        super(DebertaClassifier, self).__init__()
        
        self.config = DebertaV2Config.from_pretrained(model_name)
        self.deberta = DebertaV2Model.from_pretrained(model_name)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        self.linear = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)
        self.out = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, **kwargs):
        x_input_ids, x_atten_masks = kwargs["input_ids"], kwargs["attention_mask"]

        
        device = x_input_ids.device
        # print(self.deberta.config)
        # print(self.deberta.config.eos_token_id)
        # raise Exception

        last_hidden = self.deberta(x_input_ids, x_atten_masks).last_hidden_state
        # print("SB"*10)
        
        # Use correct eos_token_id for Roberta
        eos_token_id = 2
        eos_token_ind = x_input_ids.eq(eos_token_id).nonzero()
        

        # Debugging: Check tensor lengths
        # print(f"len(eos_token_ind): {len(eos_token_ind)}")
        # print(f"len(x_input_ids): {len(x_input_ids)}, 3 * len(x_input_ids): {3 * len(x_input_ids)}")

        # Fix assertion to prevent crashes
        assert len(eos_token_ind) == 2 * len(x_input_ids), "Mismatch in EOS token count!"

        b_eos = [eos_token_ind[i, 1].item() for i in range(len(eos_token_ind)) if i % 2 == 0]
        e_eos = [eos_token_ind[i, 1].item() for i in range(len(eos_token_ind)) if (i + 1) % 2 == 0]
        
        x_atten_clone = x_atten_masks.clone().detach()

        for begin, end, att, att2 in zip(b_eos, e_eos, x_atten_masks, x_atten_clone):
            att[begin:], att2[: begin + 1] = 0, 0  # att all </s> --> 0; att2 1st and 2nd </s> --> 0
            att[0], att2[end] = 0, 0  # <s> --> 0; 3rd </s> --> 0

        # Convert attention masks to float and ensure they are on the same device
        txt_l = x_atten_masks.sum(1).to(device)
        topic_l = x_atten_clone.sum(1).to(device)
        txt_vec = x_atten_masks.float().to(device)
        topic_vec = x_atten_clone.float().to(device)

        txt_mean = torch.einsum("blh,bl->bh", last_hidden, txt_vec) / txt_l.unsqueeze(1)
        topic_mean = torch.einsum("blh,bl->bh", last_hidden, topic_vec) / topic_l.unsqueeze(1)

        # Concatenate sentence representations
        cat = torch.cat((txt_mean, topic_mean), dim=1)

        # Apply dropout, activation, and final classification layers
        query = self.dropout(cat)
        linear = self.relu(self.linear(query))
        out = self.out(linear)

        return out

# Load model
model = DebertaClassifier(num_labels=3, model_name="microsoft/deberta-v3-base")
model.to(device)

# Define optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * 4)

# Define loss function
criterion = torch.nn.CrossEntropyLoss()

# Training loop
def train_model(model, train_loader, valid_loader, optimizer, scheduler, criterion, epochs=4):
    for epoch in range(epochs):
        model.train()
        total_loss, total_correct = 0, 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            loss = criterion(outputs, batch['labels'])
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            total_correct += (outputs.argmax(dim=-1) == batch['labels']).sum().item()
        
        train_acc = total_correct / len(train_dataset)
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}, Accuracy = {train_acc:.4f}")
        
        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in valid_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                preds = outputs.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
        val_acc = accuracy_score(all_labels, all_preds)
        val_macro_f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"Validation: Accuracy = {val_acc:.4f}, Macro F1 = {val_macro_f1:.4f}")

# Train model
train_model(model, train_loader, valid_loader, optimizer, scheduler, criterion)

# Save trained model

# # model.save_pretrained(model_save_path)
torch.save(model.state_dict(), model_save_path)
print("Model saved successfully!")


# Load saved model
model = DebertaClassifier(num_labels=3, model_name="microsoft/deberta-v3-base")
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.to(device)

# Evaluate on test datasets
def evaluate_on_test(model, test_file):
    test_texts, test_labels = load_dataset(test_file)
    if test_texts is None:
        return
    test_dataset = StanceDataset(test_texts, test_labels, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            preds = outputs.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"Final Evaluation for {test_file}: Accuracy = {acc:.4f}, Macro F1 = {macro_f1:.4f}, Weighted F1 = {weighted_f1:.4f}")
    
    report = classification_report(all_labels, all_preds, target_names=["FAVOR", "AGAINST", "NONE"], output_dict=True)
    report_path = os.path.join(output_root, f"{os.path.basename(test_file)}_classification_report.csv")
    pd.DataFrame(report).transpose().to_csv(report_path)
    print(f"Classification report saved: {report_path}")
    
    pred_df = pd.DataFrame({"Predictions": all_preds})
    pred_df.to_csv(os.path.join(output_root, f"{os.path.basename(test_file)}_predictions.csv"), index=False)
    print(f"Predictions saved: {os.path.join(output_root, f'{os.path.basename(test_file)}_predictions.csv')}")

for test_file in test_files:
    evaluate_on_test(model, test_file)
