from helpers import *

class Parameters():
    def __init__(self):
        self.SEED = 42
        self.use_cuda = False
        self.batch_size = 32
        


        self.cwd = os.path.dirname(os.path.abspath(__file__))
        repo_dir = os.path.dirname(os.path.dirname(self.cwd))
        self.dataset_base = os.path.join(repo_dir, "Datasets")
        self.dir_save_files = os.path.join(self.cwd, 'SaveFiles', 'SentimentClassifier')
        if not os.path.exists(self.dir_save_files):
            os.makedirs(self.dir_save_files)
            print('directory created: {}'.format(self.dir_save_files))
        else:
            print('directory {} exists'.format(self.dir_save_files))
            


        self.epochs = 10
        self.print_every = 15
        self.num_epochs_freeze_bert = 1