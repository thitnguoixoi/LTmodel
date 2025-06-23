import pandas as pd
import logging
import base64
import math
import nltk
from nltk.tokenize import RegexpTokenizer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from LTModel import LTModel
from bilstm import BiLSTMModel
import HTokenizer
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns


def main():

    pd.set_option("display.max_columns", None)  # Hiển thị tất cả các cột
    pd.set_option("display.max_rows", None)     # Hiển thị tất cả các dòng
    # Độ rộng dòng đủ lớn để không bị xuống dòng
    pd.set_option("display.width", 1000)

    # Load data & vocab
    df = pd.read_csv("alldata_balanced.csv", low_memory=False)
    for col in df.columns:
        df[col] = df[col].astype('object').fillna('')
    logger.info(f"Loaded data shape: {df.shape}")

    vocab = {}
    with open("final_filtered_vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    logger.info(f"Vocabulary size: {vocab_size}")

    # Đọc nhãn từ file CSV
    labels = df["request.Attack_Tag"]
    df_without_class = df.drop(columns=['request.Attack_Tag'], errors='ignore')
    logger.info(f"Data without class shape: {df_without_class.shape}")
    # Process data
    logger.info("first 5 rows of data without class:")
    logger.info(df_without_class.head())
    logger.info("first 5 labels:")
    logger.info(labels.head())
    logger.info("Processing data...")

    tokenized_data = process_dataframe(df_without_class, vocab)
    logger.info("First 5 tokenized sequences:")
    for i, seq in enumerate(tokenized_data[:5]):
        logger.info(f"  [{i}] {seq}")
    # Convert tokens to IDs
    logger.info("Converting tokens to IDs...")
    token_ids_data = convert_tokens_to_ids(tokenized_data, vocab)
    logger.info("First 5 token ID sequences:")
    for i, ids in enumerate(token_ids_data[:5]):
        logger.info(f"  [{i}] {ids}")
    maxlen = 512

    # Prepare data loaders
    train_loader, test_loader = prepare_data(
        token_ids_data, labels, vocab, max_len=maxlen, batch_size=128)
    logger.info(f"Label mapping: \n'Normal': 0,\n'Directory Traversal': 1,\n'SQL Injection': 2,\n'XSS': 3,\n'Log Forging': 4,\n'Cookie Injection': 5,\n'RCE': 6,\n'LOG4J': 7")
    # Train LTModel
    model = LTModel(
        vocab_size=len(vocab),
        hidden_size=128,
        num_layers=2,
        num_classes=8
    )
    if hasattr(model, 'position_encoding') and hasattr(model.position_encoding, 'max_len'):
        model.position_encoding.max_len = maxlen
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = FocalLoss(gamma=2)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=7, gamma=0.1)
    model = model.to(device)
    logger.info("Training LTModel...")
    history_att = train_optimized_model(
        model, train_loader, optimizer, scheduler, device,
        epochs=15, accumulation_steps=16, batch_chunks=2, criterion=criterion
    )
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': model.token_embedding.embedding.num_embeddings if hasattr(model.token_embedding.embedding, 'num_embeddings') else None,
        'embedding_dim': model.token_embedding.embedding.embedding_dim if hasattr(model.token_embedding.embedding, 'embedding_dim') else None,
        'num_layers': len(model.encoder.layers) if hasattr(model.encoder, 'layers') else None,
        'ff_dim': model.encoder.layers[0].feed_forward.linear1.out_features if hasattr(model.encoder.layers[0].feed_forward, 'linear1') else None,
        'dropout': model.position_encoding.dropout.p if hasattr(model.position_encoding, 'dropout') else None,
        'num_classes': model.classifier.out_features if hasattr(model.classifier, 'out_features') else None
    }, 'final_model_complete.pt')
    logger.info("LTModel saved successfully!")

    # Train BiLSTM
    bilstm_model = BiLSTMModel(
        len(vocab), 128, 128, 2, 8, 0.1
    ).to(device)
    bilstm_optimizer = optim.AdamW(
        bilstm_model.parameters(), lr=1e-4, weight_decay=1e-5)
    bilstm_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        bilstm_optimizer, T_0=5, T_mult=1, eta_min=1e-5)
    logger.info("Training BiLSTM model...")
    bilstm_model, history_bilstm= train_bilstm(
        bilstm_model, train_loader, bilstm_optimizer, bilstm_scheduler,
        epochs=15, device=device, criterion=criterion
    )

    # Evaluate và vẽ confusion matrix
    logger.info("Evaluating Attention model on test data...")
    acc_att, report_att, conf_matrix_att = evaluate_model(model, test_loader, device=device)
    plot_confusion_matrix(conf_matrix_att, labels=['Normal','Directory Traversal','SQL Injection','XSS','Log Forging','Cookie Injection','RCE','LOG4J'], title="LTModel Confusion Matrix")

    logger.info("Evaluating BiLSTM model on test data...")
    acc_bilstm, report_bilstm, conf_matrix_bilstm = evaluate_model(bilstm_model, test_loader, device=device)
    plot_history(history_att, title_prefix="LTModel", save_path="ltmodel_train.png")
    plot_history(history_bilstm, title_prefix="BiLSTM", save_path="bilstm_train.png")
    plot_confusion_matrix(conf_matrix_att, labels=[...], title="LTModel Confusion Matrix", save_path="ltmodel_confmat.png")
    plot_confusion_matrix(conf_matrix_bilstm, labels=[...], title="BiLSTM Confusion Matrix", save_path="bilstm_confmat.png")
    torch.save({
        'model_state_dict': bilstm_model.state_dict(),
        'vocab_size': len(vocab),
        'embedding_dim': 128,
        'hidden_dim': 128,
        'num_layers': 2,
        'num_classes': 8,
        'dropout': 0.1
    }, 'bilstm_model.pt')
    logger.info("BiLSTM model saved successfully!")
# Prepare data for training


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        # Compute standard cross entropy loss (per sample)
        ce_loss = F.cross_entropy(
            input, target, weight=self.weight, reduction='none')
        # Compute the probability of the true class
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def plot_history(history, title_prefix="", save_path=None):
    epochs = range(1, len(history["loss"]) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["loss"], label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["accuracy"], label="Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{title_prefix} Accuracy")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_confusion_matrix(conf_matrix, labels, title="Confusion Matrix", save_path=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
def prepare_data(token_ids_data, labels, vocab, max_len=128, test_size=0.2, batch_size=32):
    """
    Prepare data for training by padding sequences, splitting into train/test sets,
    and creating DataLoader objects.

    Args:
        token_ids_data: List of lists containing token IDs
        labels: List of labels (0 for valid, 1 for anomalous)
        max_len: Maximum sequence length for padding
        test_size: Proportion of data to use for testing
        batch_size: Batch size for training

    Returns:
        train_loader: DataLoader for training data
        test_loader: DataLoader for testing data
    """
    # Ensure token_ids_data and labels have the same length
    if len(token_ids_data) != len(labels):
        raise ValueError(
            f"Mismatch between token_ids_data length ({len(token_ids_data)}) and labels length ({len(labels)})")

    # Convert labels to numeric values if they're strings
    if isinstance(labels[0], str):
        label_map = {
            'Normal': 0,
            'Directory Traversal': 1,
            'SQL Injection': 2,
            'XSS': 3,
            'Log Forging': 4,
            'Cookie Injection': 5,
            'RCE': 6,
            'LOG4J': 7
        }
        labels = [label_map[label] for label in labels]

    # Pad sequences to the same length
    padded_data = []
    for seq in token_ids_data:
        if len(seq) < max_len:
            padded_seq = seq + [vocab['<pad>']] * (max_len - len(seq))
        else:
            padded_seq = seq[:max_len]
        padded_data.append(padded_seq)

    # Convert to PyTorch tensors
    X = torch.tensor(padded_data, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)

    # Create DataLoader objects
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def process_dataframe(df, vocab):
    """
    Process all data in the dataframe and extract tokens using RegexpTokenizers
    """
    tokenizer = HTokenizer.HTokenizer()
    tokenized_data = []
    for index, row in df.iterrows():
        tokens = tokenizer.tokenize_df(row, columns=df.columns)
        filtered_tokens = filter_tokens_by_vocab(tokens, vocab)
        tokenized_data.append(filtered_tokens)
    return tokenized_data


def filter_tokens_by_vocab(tokens, vocab):
    """
    Filter tokens based on vocabulary and add UNK for unknown tokens.
    """
    result = []
    for token in tokens:
        if token in vocab:
            result.append(token)
        else:
            result.append('<unk>')
    return result


def convert_tokens_to_ids(tokenized_data, vocab):
    """
    Convert list of token lists into list of ID lists based on vocabulary.
    """
    token_ids_data = []
    for tokens in tokenized_data:
        token_ids = [vocab[token] if token in vocab else vocab['<unk>']
                     for token in tokens]
        token_ids_data.append(token_ids)
    return token_ids_data


def train_optimized_model(model, train_loader, optimizer, scheduler, device, epochs=10, accumulation_steps=4, batch_chunks=2, criterion=None, max_len=160):
    """
    Optimized training function for LTModel with gradient accumulation and batch chunking

    Args:
        model: LTModel instance
        train_loader: DataLoader for training data
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler
        epochs: Number of training epochs
        accumulation_steps: Number of batches to accumulate gradients before updating weights
        batch_chunks: Number of chunks to split each batch into to reduce memory usage
    """
    model.train()
    # Check if CUDA is available for mixed precision training
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    from tqdm.auto import tqdm
    history = {"loss": [], "accuracy": []}  # Thêm dòng này

    for epoch in range(epochs):
        total_loss = 0
        epoch_correct = 0
        epoch_total = 0
        optimizer.zero_grad()
        batch_count = 0

        # Create progress bar for the current epoch
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in progress_bar:
            inputs, labels = batch
            batch_size, seq_len = inputs.shape

            # Truncate sequences if needed
            if seq_len > max_len:
                inputs = inputs[:, :max_len]
                seq_len = max_len

            # Split batch into smaller chunks to reduce memory usage
            chunk_size = batch_size // batch_chunks
            if chunk_size == 0:
                chunk_size = 1

            for i in range(0, batch_size, chunk_size):
                # Get the current chunk
                end_idx = min(i + chunk_size, batch_size)
                inputs_chunk = inputs[i:end_idx].to(device)
                labels_chunk = labels[i:end_idx].to(device)

                # Create attention mask (1 for tokens, 0 for padding)
                # 1 is [PAD] token
                attention_mask = (inputs_chunk != 1).float()

                # Use mixed precision training if available
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        # Get logits directly from the model output
                        outputs = model(inputs_chunk)
                        # Check if outputs is a dictionary and extract logits
                        if isinstance(outputs, dict) and 'logits' in outputs:
                            logits = outputs['logits']
                        else:
                            logits = outputs

                        loss = criterion(logits, labels_chunk)
                        # Scale the loss by the number of chunks and accumulation steps
                        loss = loss / (batch_chunks * accumulation_steps)

                    # Optimized backward pass with scaler
                    scaler.scale(loss).backward()
                else:
                    # Standard training path
                    outputs = model(inputs_chunk)
                    # Check if outputs is a dictionary and extract logits
                    if isinstance(outputs, dict) and 'logits' in outputs:
                        logits = outputs['logits']
                    else:
                        logits = outputs

                    loss = criterion(logits, labels_chunk)
                    # Scale the loss by the number of chunks and accumulation steps
                    loss = loss / (batch_chunks * accumulation_steps)
                    loss.backward()

                # Calculate accuracy
                _, predicted = torch.max(logits, 1)
                batch_correct = (predicted == labels_chunk).sum().item()
                batch_total = labels_chunk.size(0)

                # Track epoch metrics
                epoch_correct += batch_correct
                epoch_total += batch_total

                total_loss += loss.item() * batch_chunks * accumulation_steps

                # Free up memory
                del inputs_chunk, labels_chunk, outputs, loss, attention_mask
                if 'logits' in locals():
                    del logits
                torch.cuda.empty_cache()

            batch_count += 1

            # Update weights after accumulation_steps
            if batch_count % accumulation_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=1.0)
                    optimizer.step()

                optimizer.zero_grad()

                # Update progress bar with current loss and accuracy
                current_accuracy = 100 * epoch_correct / epoch_total if epoch_total > 0 else 0
                current_loss = total_loss / batch_count
                progress_bar.set_postfix({
                    'loss': f'{current_loss:.6f}',
                    'accuracy': f'{current_accuracy:.2f}%'
                })

        # Update learning rate after each epoch
        scheduler.step()

        # Calculate final accuracy for the epoch
        epoch_accuracy = 100 * epoch_correct / epoch_total if epoch_total > 0 else 0
        avg_loss = total_loss / len(train_loader)
        logger.info(
            f'Epoch {epoch+1}/{epochs} completed - Loss: {avg_loss:.6f}, Accuracy: {epoch_accuracy:.2f}%')
        history["loss"].append(avg_loss)
        history["accuracy"].append(epoch_accuracy)
    return history

def train_bilstm(model, train_loader, optimizer, scheduler, epochs, device, criterion):
    model.train()
    scaler = torch.amp.GradScaler('cuda')  # For mixed precision training

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        history = {"loss": [], "accuracy": []}  # Thêm dòng này

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')

        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass with mixed precision
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            # Backward pass and optimize with gradient scaling
            scaler.scale(loss).backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # Track statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            accuracy = 100. * correct / total
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{accuracy:.2f}%'
            })

        # Update learning rate based on scheduler
        scheduler.step()

        logger.info(
            f'Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        history["loss"].append(avg_loss)
        history["accuracy"].append(accuracy)


    return model, history


def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            if isinstance(outputs, dict):
                logits = outputs.get("logits", None)
                if logits is None:
                    raise ValueError("Output dict does not contain 'logits'")
            else:
                logits = outputs

            _, predicted = logits.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    accuracy = accuracy_score(all_targets, all_preds)
    report = classification_report(
        all_targets, all_preds, target_names=['Normal',
                                              'Directory Traversal',
                                              'SQL Injection',
                                              'XSS',
                                              'Log Forging',
                                              'Cookie Injection',
                                              'RCE',
                                              'LOG4J'])
    conf_matrix = confusion_matrix(all_targets, all_preds)

    logger.info(f"Evaluation Accuracy: {accuracy:.4f}")
    logger.info("\nClassification Report:\n" + report)
    logger.info("\nConfusion Matrix:\n" + str(conf_matrix))

    return accuracy, report, conf_matrix


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler("train_attention.log",
                                mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    main()
