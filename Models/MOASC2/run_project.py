# run_project.py

import os
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix

from transformers import AdamW

# Your local imports
from prepare_datasets import *
from Parameters import *
from DataSet import *
from SentimentClassifier import *

################################################################################
# PLOT FUNCTIONS
################################################################################
def plot_confusion_matrix(true_labels, pred_labels, classes, output_dir):
    """Saves a confusion matrix image to `output_dir`."""
    cm = confusion_matrix(true_labels, pred_labels)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, None] + 1e-7)

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()

    save_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_curves(train_loss, val_loss, test_loss, output_dir):
    """Plots train/val/test loss vs. epoch and saves the figure."""
    plt.figure()
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.plot(test_loss, label="Test Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train / Validation / Test Loss')
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(output_dir, 'train_val_test_loss_curve.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Train/val/test loss curve saved to {save_path}")

################################################################################
# TEXT CLASSIFIER FUNCTION
################################################################################
def run_text_sentiment_classifier(
    train_loader, validation_loader, test_loader, params, device, writer, output_dir
):
    """
    Example with text-only classifier (TextSentimentClassifier).
    """
    model = TextSentimentClassifier(freeze_bert=True).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    train_loss, val_loss, test_loss = [], [], []
    val_acc_list, test_acc_list = [], []

    def evaluate(model, dataloader):
        model.eval()
        mean_acc, mean_l = 0.0, 0.0
        count = 0

        all_labels = []
        all_preds = []

        with torch.no_grad():
            for sequences, attn_masks, labels in dataloader:
                sequences = sequences.to(device)
                attn_masks = attn_masks.to(device)
                labels = labels.to(device)

                logits = model(sequences, attn_masks)
                l_val = F.cross_entropy(logits, labels.long()).item()

                mean_l += l_val
                mean_acc += get_accuracy_from_logits(logits, labels)
                count += 1

                preds = logits.argmax(dim=-1).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())

        mean_acc /= count
        mean_l /= count

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        return mean_acc, mean_l, all_labels, all_preds

    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=1)
    best_accuracy = 0.0
    global_step = 0

    for epoch in range(params.epochs):
        print(f"\n>>> Epoch {epoch+1}/{params.epochs}")
        model.train()

        epoch_loss = 0.0
        total_samples = 0
        epoch_start_time = time.time()

        for it, (sequences, attn_masks, labels) in enumerate(train_loader):
            sequences = sequences.to(device)
            attn_masks = attn_masks.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(sequences, attn_masks)

            # Balanced class weights
            class_w = class_weight.compute_class_weight(
                class_weight='balanced',
                classes=np.unique(labels.cpu()),
                y=labels.cpu().numpy()
            )
            class_w = torch.tensor(class_w, dtype=torch.float).to(device)

            loss = F.cross_entropy(logits, labels.long(), weight=class_w)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            epoch_loss += loss.item() * batch_size
            total_samples += batch_size
            global_step += 1

            writer.add_scalar('Loss/train_batch', loss.item(), global_step)

            if (it + 1) % params.print_every == 0:
                acc = get_accuracy_from_logits(logits, labels)
                print(f"    Iteration {it+1}, Loss: {loss.item():.4f}, Accuracy: {acc*100:.2f}%")

        epoch_loss /= total_samples
        train_loss.append(epoch_loss)

        val_acc, val_l, _, _ = evaluate(model, validation_loader)
        val_loss.append(val_l)
        val_acc_list.append(val_acc)

        test_acc, t_loss, _, _ = evaluate(model, test_loader)
        test_loss.append(t_loss)
        test_acc_list.append(test_acc)

        scheduler.step(val_l)

        writer.add_scalar('Loss/train_epoch', epoch_loss, epoch)
        writer.add_scalar('Loss/val_epoch', val_l, epoch)
        writer.add_scalar('Loss/test_epoch', t_loss, epoch)

        print(f"Epoch {epoch+1} complete. "
              f"Train Loss: {epoch_loss:.4f}, Val Loss: {val_l:.4f}, Val Acc: {val_acc*100:.2f}%, "
              f"Test Loss: {t_loss:.4f}, Test Acc: {test_acc*100:.2f}%")

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            print(f"  [*] New best val_acc = {val_acc:.4f}")

        print(f"Epoch time: {time.time() - epoch_start_time:.2f}s")

    # Plot curves
    plot_curves(train_loss, val_loss, test_loss, output_dir)

    # Evaluate one last time for confusion matrix
    _, _, y_true, y_pred = evaluate(model, test_loader)
    classes = ["0", "1", "2"]  # Adjust if needed
    plot_confusion_matrix(y_true, y_pred, classes, output_dir)

################################################################################
# MULTIMODAL CLASSIFIER FUNCTION
################################################################################
def run_sentiment_classifier(
    train_loader, validation_loader, test_loader, params, device, writer, output_dir
):
    """
    Example with the SentimentClassifier that uses images + text.
    """
    model = SentimentClassifier(freeze_bert=False).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

    train_loss, val_loss, test_loss = [], [], []
    val_acc_list, test_acc_list = [], []

    def evaluate(model, dataloader):
        model.eval()
        mean_acc, mean_l = 0., 0.
        count = 0

        all_labels = []
        all_preds = []

        with torch.no_grad():
            for seqs, attn_masks, ims, labels in dataloader:
                seqs = seqs.to(device)
                attn_masks = attn_masks.to(device)
                ims = ims.to(device)
                labels = labels.to(device)

                logits = model(seqs, attn_masks, ims)
                l_val = F.cross_entropy(logits, labels.long()).item()
                mean_l += l_val
                mean_acc += get_accuracy_from_logits(logits, labels)

                preds = logits.argmax(dim=-1).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())

                count += 1

        mean_acc /= count
        mean_l /= count

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        return mean_acc, mean_l, all_labels, all_preds

    best_accuracy = 0.0
    best_test_accuracy = 0.0
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=1)
    global_step = 0

    for epoch in range(params.epochs):
        print(f"\n>>> Epoch {epoch+1}/{params.epochs}")
        epoch_start_time = time.time()

        model.train()
        epoch_loss = 0.0
        samples_count = 0

        for it, (seqs, masks, ims, lbls) in enumerate(train_loader):
            seqs = seqs.to(device)
            masks = masks.to(device)
            ims = ims.to(device)
            lbls = lbls.to(device)

            optimizer.zero_grad()

            logits = model(seqs, masks, ims)

            # Balanced class weights
            class_w = class_weight.compute_class_weight(
                class_weight='balanced',
                classes=np.unique(lbls.cpu()),
                y=lbls.cpu().numpy()
            )
            class_w = torch.tensor(class_w, dtype=torch.float).to(device)

            if len(class_w) == 2:
                loss = F.cross_entropy(logits, lbls.long())
            else:
                loss = F.cross_entropy(logits, lbls.long(), weight=class_w)

            loss.backward()
            optimizer.step()

            batch_size = lbls.size(0)
            epoch_loss += loss.item() * batch_size
            samples_count += batch_size
            global_step += 1

            writer.add_scalar('Loss/train_batch', loss.item(), global_step)

            if (it + 1) % params.print_every == 0:
                acc = get_accuracy_from_logits(logits, lbls)
                print(f"    Iteration {it+1}, Loss: {loss.item():.4f}, Acc: {acc*100:.2f}%")

        epoch_loss /= samples_count
        train_loss.append(epoch_loss)

        # Evaluate on validation
        val_acc, val_l, _, _ = evaluate(model, validation_loader)
        val_loss.append(val_l)
        val_acc_list.append(val_acc)

        # Evaluate on test
        test_acc, t_loss, _, _ = evaluate(model, test_loader)
        test_loss.append(t_loss)
        test_acc_list.append(test_acc)

        scheduler.step(val_l)

        writer.add_scalar('Loss/train_epoch', epoch_loss, epoch)
        writer.add_scalar('Loss/val_epoch', val_l, epoch)
        writer.add_scalar('Loss/test_epoch', t_loss, epoch)

        print(f"Epoch {epoch+1} complete. "
              f"Train Loss: {epoch_loss:.4f}, Val Loss: {val_l:.4f}, Val Acc: {val_acc*100:.2f}%, "
              f"Test Loss: {t_loss:.4f}, Test Acc: {test_acc*100:.2f}%")

        if val_acc > best_accuracy:
            print(f"  [*] New best val_acc = {val_acc:.4f}")
            best_accuracy = val_acc

        if test_acc > best_test_accuracy:
            print(f"  [*] New best test_acc = {test_acc:.4f}, saving model...")
            best_test_accuracy = test_acc
            torch.save(
                model,
                os.path.join(
                    "/home/rgg2706/Multimodal-Sentiment-Analysis/Models/MOASC/SaveFiles/SentimentClassifier",
                    f"model10_{test_acc*100:.0f}.pth"
                )
            )

        print(f"Epoch time: {time.time() - epoch_start_time:.2f}s")

    # final curves
    plot_curves(train_loss, val_loss, test_loss, output_dir)

    # confusion matrix
    _, _, lbls_true, lbls_pred = evaluate(model, test_loader)
    classes = ["0","1","2"]
    plot_confusion_matrix(lbls_true, lbls_pred, classes, output_dir)


################################################################################
# MAIN
################################################################################
if __name__ == "__main__":

    # 1) Setup
    params: Parameters = Parameters()
    params.use_cuda = torch.cuda.is_available()
    params.epochs = 40
    params.print_every = 20

    random.seed(params.SEED)
    np.random.seed(params.SEED)
    torch.manual_seed(params.SEED)
    torch.cuda.manual_seed_all(params.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) Where to store logs & plots
    #    The Slurm script sets NEW_LOGS_DIR. If not set, default to /tmp
    LOG_DIR = os.environ.get("NEW_LOGS_DIR", "/tmp/run_project_logs")
    os.makedirs(LOG_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=LOG_DIR)
    print(f"\n> Logs + images will be saved in: {LOG_DIR}")

    # 3) Load dataset
    dataset: DataSet = DataSet(params.cwd)

    # 4) Build DataLoaders
    multimodal_dataset_train = MultimodalDataset(
        dataset.images_train,
        dataset.text_train,
        dataset.post_train_labels,
        36
    )
    multimodal_dataset_val = MultimodalDataset(
        dataset.images_validation,
        dataset.text_validation,
        dataset.post_val_labels,
        36
    )
    multimodal_dataset_test = MultimodalDataset(
        dataset.images_test,
        dataset.text_test,
        dataset.post_test_labels,
        36
    )

    train_loader = DataLoader(multimodal_dataset_train, batch_size=params.batch_size)
    val_loader   = DataLoader(multimodal_dataset_val,   batch_size=params.batch_size)
    test_loader  = DataLoader(multimodal_dataset_test,  batch_size=params.batch_size)

    # 5) Actually run training
    #    Choose text-only or the full multimodal:
    # run_text_sentiment_classifier(train_loader, val_loader, test_loader, params, device, writer, LOG_DIR)
    run_sentiment_classifier(train_loader, val_loader, test_loader, params, device, writer, LOG_DIR)

    writer.close()
    print(f"\nDone. Logs and plots should be in: {LOG_DIR}")
