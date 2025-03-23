# run_project.py

from helpers import *
from prepare_datasets import *
from Parameters import *
from DataSet import *
from SentimentClassifier import *
import argparse

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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

def run_text_sentiment_classifier(train_loader, validation_loader, test_loader, params, device):
    model = TextSentimentClassifier(freeze_bert=True).to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    train_loss, validation_loss, test_loss = [], [], []
    validation_accuracy, test_accuracy = [], []
    average_recall, f1_score_pn = [], []


    def evaluate(model, dataloader):
        mean_acc, mean_loss = 0., 0.
        average_recall, f1_pn, f1_macro, f1_weighted = 0., 0., 0., 0.
        count = 0

        model.eval()
        with torch.no_grad():
            for seq, attn_masks, labels in dataloader:
                logits = model(seq, attn_masks)
                mean_loss += F.cross_entropy(logits, labels.long()).item()
                mean_acc += get_accuracy_from_logits(logits, labels)
                tp, fn, fp = get_tp_fn_fp_from_logits(logits, labels)
                recall = get_recall(tp, fn)
                average_recall += get_avg_recall(recall)
                f1_pn += get_f1_pn(tp, fp, fn)
                f1_macro    += get_f1_macro(tp, fp, fn)
                f1_weighted += get_f1_weighted(tp, fp, fn)

                count += 1

        return mean_acc / count, mean_loss / count, average_recall / count, f1_pn / count, f1_macro / count, f1_weighted / count


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
                # build a 3-length weight vector via bincount
                num_classes = 3
                labels_cpu = labels.cpu().numpy()
                counts = np.bincount(labels_cpu, minlength=num_classes)
                total = counts.sum()
                weights = []
                for c in range(num_classes):
                    if counts[c] == 0:
                        weights.append(0.0)
                    else:
                        weights.append(total / counts[c])
                class_weights = torch.tensor(weights, dtype=torch.float).to(device)
                # Computing loss
                loss = F.cross_entropy(logits, labels.long(), weight=class_weights)
                # Backpropagation
                loss.backward()
                # Optimization step
                optim.step()

                batch_losses += loss.item()
                count += 1

                if (it + 1) % params.print_every == 0:
                    acc = get_accuracy_from_logits(logits, labels)
                    print("    Iteration {} of epoch {} completed. Loss : {} Accuracy : {:.2f}".format(it+1, epoch+1, loss.item(), acc * 100.))

            epoch_loss = batch_losses / count
            train_loss.append(epoch_loss)

            val_acc, val_loss, _, _, _, _ = evaluate(model, val_loader)
            validation_loss.append(val_loss)
            validation_accuracy.append(val_acc)
            if val_acc > best_accuracy:
                print("Best validation accuracy improved from {} to {}, saving model...".format(best_accuracy, val_acc))
                best_accuracy = val_acc

            scheduler.step(val_loss)

            acc_test, loss_test, avg_recall, f1_pn, f1_all, f1_w = evaluate(model, test_loader)
            print("Epoch {} completed. Test dataset:\n"
                "\t Loss: {:.4f}\n"
                "\t Accuracy : {:.2f}\n"
                "\t Average recall: {:.4f}\n"
                "\t F1_PN: {:.4f}\n"
                "\t Macro-F1: {:.4f}\n"
                "\t Weighted-F1: {:.4f}\n"
                .format(epoch+1, loss_test, acc_test * 100., avg_recall, f1_pn, f1_all, f1_w))
            test_accuracy.append(acc_test)
            test_loss.append(loss_test)
            average_recall.append(avg_recall)
            f1_score_pn.append(f1_pn)


    train(model, optimizer, train_loader, validation_loader, test_loader)
    
    LOG_DIR = os.environ.get("NEW_LOGS_DIR", "/tmp/run_project_logs")
    plot_curves(train_loss, validation_loss, test_loss, LOG_DIR)

    plot_losses(train_loss, validation_loss, test_loss)
    
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for seq, attn_masks, labels in test_loader:
            seq = seq.to(device)
            attn_masks = attn_masks.to(device)
            labels = labels.to(device)
            logits = model(seq, attn_masks)
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    plot_confusion_matrix(
        all_labels, 
        all_preds, 
        classes=["Negative", "Positive", "Neutral"], 
        output_dir=LOG_DIR
    )


def run_sentiment_classifier(train_loader, validation_loader, test_loader, params, device):
    model = SentimentClassifier(freeze_bert=False).to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

    train_loss, validation_loss, test_loss = [], [], []
    validation_accuracy, test_accuracy = [], []
    average_recall, f1_score_pn = [], []

    def evaluate(model, dataloader):
        mean_acc, mean_loss = 0., 0.
        average_recall, f1_pn, f1_macro, f1_weighted = 0., 0., 0., 0.
        count = 0

        model.eval()
        with torch.no_grad():
            for sequences, attn_masks, images, labels in dataloader:
                sequences = sequences.to(device)
                attn_masks = attn_masks.to(device)
                images = images.to(device)
                labels = labels.to(device)
                
                logits = model(sequences, attn_masks, images)
                mean_loss += F.cross_entropy(logits, labels.long()).item()
                mean_acc += get_accuracy_from_logits(logits, labels)
                tp, fn, fp = get_tp_fn_fp_from_logits(logits, labels)
                recall = get_recall(tp, fn)
                average_recall += get_avg_recall(recall)
                f1_pn += get_f1_pn(tp, fp, fn)
                f1_macro    += get_f1_macro(tp, fp, fn)
                f1_weighted += get_f1_weighted(tp, fp, fn)

                count += 1

        return mean_acc / count, mean_loss / count, average_recall / count, f1_pn / count, f1_macro / count, f1_weighted / count

    def train(train_loader, validation_loader, test_loader, params, device):
        logits = None
        best_accuracy = 0.
        best_test_accuracy = 0.
        best_model = None

        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=1)
        for epoch in range(params.epochs):
            model.train()

            batch_losses = 0.
            count_iterations = 0

            for it, (sequences, attn_masks, images, labels) in enumerate(train_loader):
                
                sequences = sequences.to(device)
                attn_masks = attn_masks.to(device)
                images = images.to(device)
                labels = labels.to(device)

                # Clear gradients
                optimizer.zero_grad()
                # Forward pass: Obtaining the logits from the model
                logits = model(sequences, attn_masks, images)
                # build a 3-length weight vector via bincount
                num_classes = 3
                labels_cpu = labels.cpu().numpy()
                counts = np.bincount(labels_cpu, minlength=num_classes)
                total = counts.sum()
                weights = []
                for c in range(num_classes):
                    if counts[c] == 0:
                        weights.append(0.0)
                    else:
                        weights.append(total / counts[c])
                class_weights = torch.tensor(weights, dtype=torch.float).to(device)
                labels = labels.to(device)
                class_weights = class_weights.to(device)
                # Computing loss
                loss = None
                if len(class_weights) == 2:
                    loss = F.cross_entropy(logits, labels.long())
                else:
                    loss = F.cross_entropy(logits, labels.long(), weight=class_weights)
                # Backpropagation pass
                loss.backward()
                # Optimization step
                optimizer.step()

                batch_losses += loss.item()
                count_iterations += 1

                if (it + 1) % params.print_every == 0:
                    acc = get_accuracy_from_logits(logits, labels)
                    print("    Iteration {} of epoch {} completed. Loss : {} Accuracy : {:.2f}".format(it+1, epoch+1, loss.item(), acc * 100.))
                
            epoch_loss = batch_losses / count_iterations
            train_loss.append(epoch_loss)

            val_acc, val_loss, _, _, _, _ = evaluate(model, validation_loader)
            validation_loss.append(val_loss)
            validation_accuracy.append(val_acc)
            if val_acc > best_accuracy:
                print("Best validation accuracy improved from {} to {}".format(best_accuracy, val_acc))
                best_accuracy = val_acc

            scheduler.step(val_loss)

            acc_test, loss_test, avg_recall, f1_pn, f1_all, f1_w = evaluate(model, test_loader)
            print("Epoch {} completed. Test dataset:\n"
                "\t Loss: {:.4f}\n"
                "\t Accuracy : {:.2f}\n"
                "\t Average recall: {:.4f}\n"
                "\t F1_PN: {:.4f}\n"
                "\t Macro-F1: {:.4f}\n"
                "\t Weighted-F1: {:.4f}\n"
                .format(epoch+1, loss_test, acc_test * 100., avg_recall, f1_pn, f1_all, f1_w))
            
            if acc_test > best_test_accuracy:
                print("Best test accuracy improved from {} to {}, saving model...".format(best_test_accuracy, acc_test))
                best_test_accuracy = acc_test
                torch.save(model, os.path.join('/home/rgg2706/Multimodal-Sentiment-Analysis/Models/MultimodalOpinionAnalysis2/SaveFiles/SentimentClassifier', 'model10_{:.0f}_{:.0f}'.format(acc_test * 100, f1_pn * 100 )))
            
            test_accuracy.append(acc_test)
            test_loss.append(loss_test)
            average_recall.append(avg_recall)
            f1_score_pn.append(f1_pn)
                

    train(train_loader, validation_loader, test_loader, params, device)
    
    LOG_DIR = os.environ.get("NEW_LOGS_DIR", "/tmp/run_project_logs")
    plot_curves(train_loss, validation_loss, test_loss, LOG_DIR)

    plot_losses(train_loss, validation_loss, test_loss)
    
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for sequences, attn_masks, images, labels in test_loader:
            sequences = sequences.to(device)
            attn_masks = attn_masks.to(device)
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(sequences, attn_masks, images)
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    plot_confusion_matrix(
        all_labels, 
        all_preds, 
        classes=["Negative", "Positive", "Neutral"], 
        output_dir=LOG_DIR
    )

    plot_losses(train_loss, validation_loss, test_loss)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="--model_name variable not set")
    parser.add_argument("--dataset", type=str, default="MOA-MVSA-single")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_model_name(args.model_name)
    load_contractions()

    params: Parameters = Parameters()
    params.use_cuda = torch.cuda.is_available()
    params.epochs = 15
    params.print_every = 20

    print(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if params.use_cuda else {}

    dataset = DataSet(params.dataset_base, args.dataset)

    # Creating data loaders for the text and image input from the train, validation and test sets
    multimodal_dataset_train: MultimodalDataset = MultimodalDataset(dataset.images_train, dataset.text_train, dataset.post_train_labels, 36)
    multimodal_dataset_val: MultimodalDataset = MultimodalDataset(dataset.images_validation, dataset.text_validation, dataset.post_val_labels, 36)
    multimodal_dataset_test: MultimodalDataset = MultimodalDataset(dataset.images_test, dataset.text_test, dataset.post_test_labels, 36)

    print(f"Train samples: {len(multimodal_dataset_train)}")
    print(f"Validation samples: {len(multimodal_dataset_val)}")
    print(f"Test samples: {len(multimodal_dataset_test)}")
    print(f"Total samples: {len(multimodal_dataset_train) + len(multimodal_dataset_val) + len(multimodal_dataset_test)}")
    
    # post_train_labels, post_val_labels, post_test_labels are in [0=negative,1=positive,2=neutral]
    train_counts = np.bincount(dataset.post_train_labels.astype(int))
    val_counts   = np.bincount(dataset.post_val_labels.astype(int))
    test_counts  = np.bincount(dataset.post_test_labels.astype(int))

    print(f"Train label distribution: Negative={train_counts[0]}, Positive={train_counts[1]}, Neutral={train_counts[2]}")
    print(f"Val   label distribution: Negative={val_counts[0]}, Positive={val_counts[1]}, Neutral={val_counts[2]}")
    print(f"Test  label distribution: Negative={test_counts[0]}, Positive={test_counts[1]}, Neutral={test_counts[2]}")

    total_counts = train_counts + val_counts + test_counts
    print(f"Overall label distribution: Negative={total_counts[0]}, Positive={total_counts[1]}, Neutral={total_counts[2]}")

    multimodal_train_loader = DataLoader(multimodal_dataset_train, batch_size = params.batch_size)
    multimodal_val_loader = DataLoader(multimodal_dataset_val, batch_size = params.batch_size)
    multimodal_test_loader = DataLoader(multimodal_dataset_test, batch_size = params.batch_size)

    # run_text_sentiment_classifier(text_train_loader, text_val_loader, text_test_loader, params, device)
    run_sentiment_classifier(multimodal_train_loader, multimodal_val_loader, multimodal_test_loader, params, device)