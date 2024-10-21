import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def plot_tensorboard_loss(log_file):
    # Create output file path for saving the plot
    log_dir = os.path.dirname(log_file)
    output_file = os.path.join(log_dir, 'trainval_loss_curves.png')
    print(f"Output File: {output_file}")
    
    # Load TensorBoard event data
    ea = event_accumulator.EventAccumulator(log_file)
    ea.Reload()
    
    # Extract loss values for training and validation
    train_loss_events = ea.Scalars('Loss/train')
    val_loss_events = ea.Scalars('Loss/val')
    
    # Extract step and loss information
    train_steps = [event.step for event in train_loss_events]
    train_losses = [event.value for event in train_loss_events]
    val_steps = [event.step for event in val_loss_events]
    val_losses = [event.value for event in val_loss_events]
    
    # Plot training and validation loss curves
    plt.figure()
    plt.plot(train_steps, train_losses, label='Training Loss')
    plt.plot(val_steps, val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    
    # Save the plot
    plt.savefig(output_file)
    print(f'Training and validation loss curves saved to {output_file}')

if __name__ == '__main__':
    log_file = 'Multimodal-Sentiment-Analysis/Logs/HHMAFM/2024-10-20/sub-4/011_Oct-20-2024_10:10_PM-grad-clip/events.out.tfevents.1729476688.skl-a-28.rc.rit.edu.810202.0'
    plot_tensorboard_loss(log_file)
