from prepare_datasets import *
from Parameters import *
from DataSet import *
from SentimentClassifier import *

<<<<<<< HEAD
def run_text_sentiment_classifier(train_loader, validation_loader, test_loader, params, device):
    model = TextSentimentClassifier(freeze_bert=True).to(device)

=======
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.utils import class_weight
import time
import os
import random
import numpy as np
from transformers import AdamW

def plot_losses(train_loss, val_loss, test_loss):
    plt.plot(train_loss, label="Train Loss", color="green")
    plt.plot(val_loss, label="Validation Loss", color="blue")
    plt.plot(test_loss, label="Test Loss", color="red")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation / Test Loss")
    plt.show()

def run_text_sentiment_classifier(train_loader, validation_loader, test_loader, params, device):
    model = TextSentimentClassifier(freeze_bert=True).to(device)
>>>>>>> 287902a (MOA)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    train_loss, validation_loss, test_loss = [], [], []
    validation_accuracy, test_accuracy = [], []
    average_recall, f1_score_pn = [], []

<<<<<<< HEAD

    def evaluate(model, dataloader):
        mean_acc, mean_loss = 0., 0.
        average_recall, f1_pn = 0., 0.
=======
    def evaluate(model, dataloader):
        mean_acc, mean_loss = 0., 0.
        avg_recall, f1_pn_val = 0., 0.
>>>>>>> 287902a (MOA)
        count = 0

        model.eval()
        with torch.no_grad():
            for seq, attn_masks, labels in dataloader:
                logits = model(seq, attn_masks)
                mean_loss += F.cross_entropy(logits, labels.long()).item()
                mean_acc += get_accuracy_from_logits(logits, labels)
                tp, fn, fp = get_tp_fn_fp_from_logits(logits, labels)
                recall = get_recall(tp, fn)
<<<<<<< HEAD
                average_recall += get_avg_recall(recall)
                f1_pn += get_f1_pn(tp, fp, fn)

                count += 1

        return mean_acc / count, mean_loss / count, average_recall / count, f1_pn / count


    def train(model, optim, train_loader, val_loader, test_loader):
        best_accuracy = 0.

        scheduler = ReduceLROnPlateau(optim, 'min', patience = 1)
        for epoch in range(params.epochs):
            model.train()

            batch_losses = 0.
            count = 0

            for it, (sequences, attn_masks, labels) in enumerate(train_loader):
                # Clear gradients
                optim.zero_grad()
                # Obtaining the logits from the model
                logits = model(sequences, attn_masks)
                # Handling the unbalanced dataset
=======
                avg_recall += get_avg_recall(recall)
                f1_pn_val += get_f1_pn(tp, fp, fn)
                count += 1

        return mean_acc / count, mean_loss / count, avg_recall / count, f1_pn_val / count

    def train(model, optim, train_loader, val_loader, test_loader):
        best_accuracy = 0.0
        scheduler = ReduceLROnPlateau(optim, "min", patience=1)

        for epoch in range(params.epochs):
            model.train()
            batch_losses = 0.0
            count = 0

            for it, (sequences, attn_masks, labels) in enumerate(train_loader):
                optim.zero_grad()
                logits = model(sequences, attn_masks)

>>>>>>> 287902a (MOA)
                class_weights = class_weight.compute_class_weight(
                    class_weight='balanced',
                    classes=np.unique(labels),
                    y=labels.numpy()
                )
<<<<<<< HEAD
                class_weights = torch.tensor(class_weights, dtype = torch.float)
                labels = labels.to(device)
                class_weights = class_weights.to(device)
                
                # Computing loss
                loss = F.cross_entropy(logits, labels.long(), weight=class_weights)
                # Backpropagation
                loss.backward()
                # Optimization step
=======
                class_weights = torch.tensor(class_weights, dtype=torch.float)
                labels = labels.to(device)
                class_weights = class_weights.to(device)

                loss = F.cross_entropy(logits, labels.long(), weight=class_weights)
                loss.backward()
>>>>>>> 287902a (MOA)
                optim.step()

                batch_losses += loss.item()
                count += 1

                if (it + 1) % params.print_every == 0:
                    acc = get_accuracy_from_logits(logits, labels)
<<<<<<< HEAD
                    print("    Iteration {} of epoch {} completed. Loss : {} Accuracy : {:.2f}".format(it+1, epoch+1, loss.item(), acc * 100.))
=======
                    print(f"    Iteration {it+1} of epoch {epoch+1} completed. "
                          f"Loss : {loss.item():.4f} Accuracy : {acc * 100:.2f}")
>>>>>>> 287902a (MOA)

            epoch_loss = batch_losses / count
            train_loss.append(epoch_loss)

<<<<<<< HEAD
            val_acc, val_loss, _, _ = evaluate(model, val_loader)
            validation_loss.append(val_loss)
            validation_accuracy.append(val_acc)
            if val_acc > best_accuracy:
                print("Best validation accuracy improved from {} to {}, saving model...".format(best_accuracy, val_acc))
                best_accuracy = val_acc

            scheduler.step(val_loss)

            acc_test, loss_test, avg_recall, f1_pn = evaluate(model, test_loader)
            print("Epoch {} completed. Evaluation measurements on test dataset:\n\t Loss: {} \n\t Accuracy : {:.2f} \
                \n\t Average recall: {:.4f} \n\t F1_PN: {:.4f}\n".format(epoch, loss_test, acc_test * 100., avg_recall, f1_pn))
            test_accuracy.append(acc_test)
            test_loss.append(loss_test)
            average_recall.append(avg_recall)
            f1_score_pn.append(f1_pn)


    train(model, optimizer, train_loader, validation_loader, test_loader)

=======
            val_acc, val_l, _, _ = evaluate(model, val_loader)
            validation_loss.append(val_l)
            validation_accuracy.append(val_acc)
            if val_acc > best_accuracy:
                print(f"Best validation accuracy improved from {best_accuracy} to {val_acc}, saving model...")
                best_accuracy = val_acc

            scheduler.step(val_l)

            acc_test, loss_test, avg_recall_val, f1_pn_val = evaluate(model, test_loader)
            print(f"Epoch {epoch} completed. Evaluation on test dataset:\n"
                  f"\t Loss: {loss_test:.4f}\n"
                  f"\t Accuracy : {acc_test * 100:.2f}\n"
                  f"\t Average recall: {avg_recall_val:.4f}\n"
                  f"\t F1_PN: {f1_pn_val:.4f}\n")
            test_accuracy.append(acc_test)
            test_loss.append(loss_test)
            average_recall.append(avg_recall_val)
            f1_score_pn.append(f1_pn_val)

    train(model, optimizer, train_loader, validation_loader, test_loader)
>>>>>>> 287902a (MOA)
    plot_losses(train_loss, validation_loss, test_loss)


def run_sentiment_classifier(train_loader, validation_loader, test_loader, params, device):
    model = SentimentClassifier(freeze_bert=False).to(device)
<<<<<<< HEAD

=======
>>>>>>> 287902a (MOA)
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

    train_loss, validation_loss, test_loss = [], [], []
    validation_accuracy, test_accuracy = [], []
    average_recall, f1_score_pn = [], []

    def evaluate(model, dataloader):
        mean_acc, mean_loss = 0., 0.
<<<<<<< HEAD
        average_recall, f1_pn = 0., 0.
=======
        avg_recall, f1_pn_val = 0., 0.
>>>>>>> 287902a (MOA)
        count = 0

        model.eval()
        with torch.no_grad():
            for sequences, attn_masks, images, labels in dataloader:
                sequences = sequences.to(device)
                attn_masks = attn_masks.to(device)
                images = images.to(device)
                labels = labels.to(device)
<<<<<<< HEAD
                
=======

>>>>>>> 287902a (MOA)
                logits = model(sequences, attn_masks, images)
                mean_loss += F.cross_entropy(logits, labels.long()).item()
                mean_acc += get_accuracy_from_logits(logits, labels)
                tp, fn, fp = get_tp_fn_fp_from_logits(logits, labels)
                recall = get_recall(tp, fn)
<<<<<<< HEAD
                average_recall += get_avg_recall(recall)
                f1_pn += get_f1_pn(tp, fp, fn)

                count += 1

        return mean_acc / count, mean_loss / count, average_recall / count, f1_pn / count

    def train(train_loader, validation_loader, test_loader, params, device):
        logits = None
        best_accuracy = 0.
        best_test_accuracy = 0.
        best_model = None

        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=1)
        for epoch in range(params.epochs):
            model.train()

            batch_losses = 0.
=======
                avg_recall += get_avg_recall(recall)
                f1_pn_val += get_f1_pn(tp, fp, fn)
                count += 1

        return mean_acc / count, mean_loss / count, avg_recall / count, f1_pn_val / count

    def train(train_loader, validation_loader, test_loader, params, device):
        best_accuracy = 0.0
        best_test_accuracy = 0.0
        scheduler = ReduceLROnPlateau(optimizer, "min", patience=1)

        for epoch in range(params.epochs):
            model.train()
            batch_losses = 0.0
>>>>>>> 287902a (MOA)
            count_iterations = 0

            for it, (sequences, attn_masks, images, labels) in enumerate(train_loader):
                sequences = sequences.to(device)
                attn_masks = attn_masks.to(device)
                images = images.to(device)
<<<<<<< HEAD

                # Clear gradients
                optimizer.zero_grad()
                # Forward pass: Obtaining the logits from the model
                logits = model(sequences, attn_masks, images)
                
                # Handling the unbalanced dataset
                class_weights = class_weight.compute_class_weight(
                    class_weight='balanced',
                    classes=np.unique(labels),
                    y=labels.numpy()
                )
                class_weights = torch.tensor(class_weights, dtype = torch.float)
                labels = labels.to(device)
                class_weights = class_weights.to(device)
                
                # Computing loss
                loss = None
=======
                labels = labels.to(device)

                optimizer.zero_grad()
                logits = model(sequences, attn_masks, images)

                class_weights = class_weight.compute_class_weight(
                    class_weight='balanced',
                    classes=np.unique(labels.cpu()),
                    y=labels.cpu().numpy()
                )
                class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

>>>>>>> 287902a (MOA)
                if len(class_weights) == 2:
                    loss = F.cross_entropy(logits, labels.long())
                else:
                    loss = F.cross_entropy(logits, labels.long(), weight=class_weights)
<<<<<<< HEAD
                # Backpropagation pass
                loss.backward()
                # Optimization step
=======

                loss.backward()
>>>>>>> 287902a (MOA)
                optimizer.step()

                batch_losses += loss.item()
                count_iterations += 1

                if (it + 1) % params.print_every == 0:
                    acc = get_accuracy_from_logits(logits, labels)
<<<<<<< HEAD
                    print("    Iteration {} of epoch {} completed. Loss : {} Accuracy : {:.2f}".format(it+1, epoch+1, loss.item(), acc * 100.))
                
            epoch_loss = batch_losses / count_iterations
            train_loss.append(epoch_loss)

            val_acc, val_loss, _, _ = evaluate(model, validation_loader)
            validation_loss.append(val_loss)
            validation_accuracy.append(val_acc)
            if val_acc > best_accuracy:
                print("Best validation accuracy improved from {} to {}".format(best_accuracy, val_acc))
                best_accuracy = val_acc

            scheduler.step(val_loss)

            acc_test, loss_test, avg_recall, f1_pn = evaluate(model, test_loader)
            print("Epoch {} completed. Evaluation measurements on test dataset:\n\t Loss: {} \n\t Accuracy : {:.2f} \
                \n\t Average recall: {:.4f} \n\t F1_PN: {:.4f}\n".format(epoch+1, loss_test, acc_test * 100., avg_recall, f1_pn))
            
            if acc_test > best_test_accuracy:
                print("Best test accuracy improved from {} to {}, saving model...".format(best_test_accuracy, acc_test))
                best_test_accuracy = acc_test
                torch.save(model, os.path.join('/home/rgg2706/Multimodal-Sentiment-Analysis/Models/MOASC/SaveFiles/SentimentClassifier', 'model10_{:.0f}_{:.0f}'.format(acc_test * 100, f1_pn * 100 )))
            
            test_accuracy.append(acc_test)
            test_loss.append(loss_test)
            average_recall.append(avg_recall)
            f1_score_pn.append(f1_pn)
                

    train(train_loader, validation_loader, test_loader, params, device)

    plot_losses(train_loss, validation_loss, test_loss)


if __name__ == "__main__":

=======
                    print(f"    Iteration {it+1} of epoch {epoch+1} completed. "
                          f"Loss : {loss.item():.4f} Accuracy : {acc * 100:.2f}")

            epoch_loss = batch_losses / count_iterations
            train_loss.append(epoch_loss)

            val_acc, val_l, _, _ = evaluate(model, validation_loader)
            validation_loss.append(val_l)
            validation_accuracy.append(val_acc)
            if val_acc > best_accuracy:
                print(f"Best validation accuracy improved from {best_accuracy} to {val_acc}")
                best_accuracy = val_acc

            scheduler.step(val_l)

            acc_test, loss_test, avg_recall_val, f1_pn_val = evaluate(model, test_loader)
            print(f"Epoch {epoch+1} completed. Test dataset:\n"
                  f"\t Loss: {loss_test:.4f}\n"
                  f"\t Accuracy : {acc_test * 100:.2f}\n"
                  f"\t Average recall: {avg_recall_val:.4f}\n"
                  f"\t F1_PN: {f1_pn_val:.4f}\n")

            if acc_test > best_test_accuracy:
                print(f"Best test accuracy improved from {best_test_accuracy} to {acc_test}, saving model...")
                best_test_accuracy = acc_test
                torch.save(model,
                           os.path.join(
                               "/home/rgg2706/Multimodal-Sentiment-Analysis/Models/MOASC/SaveFiles/SentimentClassifier",
                               f"model10_{acc_test * 100:.0f}_{f1_pn_val * 100:.0f}"
                           ))

            test_accuracy.append(acc_test)
            test_loss.append(loss_test)
            average_recall.append(avg_recall_val)
            f1_score_pn.append(f1_pn_val)

    train(train_loader, validation_loader, test_loader, params, device)
    plot_losses(train_loss, validation_loss, test_loss)

if __name__ == "__main__":
>>>>>>> 287902a (MOA)
    params: Parameters = Parameters()
    params.use_cuda = torch.cuda.is_available()
    params.epochs = 40
    params.print_every = 20

    random.seed(params.SEED)
    np.random.seed(params.SEED)
    torch.manual_seed(params.SEED)
    torch.cuda.manual_seed_all(params.SEED)

<<<<<<< HEAD
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if params.use_cuda else {}

    dataset: DataSet = DataSet(params.cwd)

    # Creating data loaders for the text and image input from the train, validation and test sets
    multimodal_dataset_train: MultimodalDataset = MultimodalDataset(dataset.images_train, dataset.text_train, dataset.post_train_labels, 36)
    multimodal_dataset_val: MultimodalDataset = MultimodalDataset(dataset.images_validation, dataset.text_validation, dataset.post_val_labels, 36)
    multimodal_dataset_test: MultimodalDataset = MultimodalDataset(dataset.images_test, dataset.text_test, dataset.post_test_labels, 36)


    multimodal_train_loader = DataLoader(multimodal_dataset_train, batch_size = params.batch_size)
    multimodal_val_loader = DataLoader(multimodal_dataset_val, batch_size = params.batch_size)
    multimodal_test_loader = DataLoader(multimodal_dataset_test, batch_size = params.batch_size)

    # run_text_sentiment_classifier(text_train_loader, text_val_loader, text_test_loader, params, device)
    run_sentiment_classifier(multimodal_train_loader, multimodal_val_loader, multimodal_test_loader, params, device)
=======
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if params.use_cuda else {}

    # Create the dataset
    dataset: DataSet = DataSet(params.cwd)

    # =====================[ DEBUG STATEMENTS ]=====================
    print("\nDEBUG: Checking dataset lengths =>")
    print("  images_train:", len(dataset.images_train),
          " text_train:", len(dataset.text_train),
          " post_train_labels:", len(dataset.post_train_labels))

    print("  images_validation:", len(dataset.images_validation),
          " text_validation:", len(dataset.text_validation),
          " post_val_labels:", len(dataset.post_val_labels))

    print("  images_test:", len(dataset.images_test),
          " text_test:", len(dataset.text_test),
          " post_test_labels:", len(dataset.post_test_labels))
    print("=====================================================\n")
    # ============================================================

    # Building the train/val/test sets
    multimodal_dataset_train: MultimodalDataset = MultimodalDataset(
        dataset.images_train,
        dataset.text_train,
        dataset.post_train_labels,
        36
    )
    multimodal_dataset_val: MultimodalDataset = MultimodalDataset(
        dataset.images_validation,
        dataset.text_validation,
        dataset.post_val_labels,
        36
    )
    multimodal_dataset_test: MultimodalDataset = MultimodalDataset(
        dataset.images_test,
        dataset.text_test,
        dataset.post_test_labels,
        36
    )

    print("\nDEBUG: MultimodalDataset train/val/test sizes =>")
    print("  len(multimodal_dataset_train):", len(multimodal_dataset_train))
    print("  len(multimodal_dataset_val):", len(multimodal_dataset_val))
    print("  len(multimodal_dataset_test):", len(multimodal_dataset_test))
    print("=====================================================\n")

    multimodal_train_loader = DataLoader(multimodal_dataset_train, batch_size=params.batch_size)
    multimodal_val_loader = DataLoader(multimodal_dataset_val, batch_size=params.batch_size)
    multimodal_test_loader = DataLoader(multimodal_dataset_test, batch_size=params.batch_size)

    # run_text_sentiment_classifier(...)
    run_sentiment_classifier(multimodal_train_loader,
                             multimodal_val_loader,
                             multimodal_test_loader,
                             params,
                             device)
>>>>>>> 287902a (MOA)
