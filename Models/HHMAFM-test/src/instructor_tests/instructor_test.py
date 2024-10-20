# test_instructor.py
import sys
import os
import torch
from argparse import Namespace
from Models.HHMAFM.src.instructor_tests.instructor_test import Instructor  # Assuming your class is saved in a file named 'instructor.py'
import torchvision.transforms as transforms
from torch.optim import Adam
sys.path.append(os.path.abspath('../'))
from Layers.mmfusion import some_function

# Define an options object to pass to the Instructor class
opt = Namespace(
    crop_size=(224, 224),
    batch_size=16,
    dataset="your_dataset_name",
    embed_dim=300,
    max_seq_len=256,
    path_image="/path/to/your/images",
    model_class=MMFUSION',  # Replace with the actual model class you want to use
    fine_tune_cnn=True,
    learning_rate=1e-4,
    num_epoch=10,
    log_step=100,
    clip_grad=1.0,
    initializer=torch.nn.init.xavier_uniform_,
    optimizer=Adam,
    inputs_cols=['text_indices', 'image'],  # Replace with your actual input columns
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Initialize the Instructor
instructor = Instructor(opt)

# Run the Instructor's method (e.g., train the model)
instructor.run()
