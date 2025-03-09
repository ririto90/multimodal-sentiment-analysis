# prepare_datasets.py

from helpers import *

datasets_path = []
for name in datasets_names:
    datasets_path.append(os.path.join('/home/rgg2706/Multimodal-Sentiment-Analysis/Datasets', name + '/'))

dir_save_files = os.path.join(base_dir, 'MultimodalOpinionAnalysis/Datasets')

text_extension_file = ".txt"
labels_filename = 'labels.txt'

if not os.path.exists(dir_save_files):
# create a directory for the processed files of the original MVSA dataset
    try:
        os.makedirs(dir_save_files)
        for dname in datasets_names:
            dataset_path = os.path.join(dir_save_files, dname)
            os.makedirs(dataset_path)
            os.makedirs(os.path.join(dataset_path, "text"))
            os.makedirs(os.path.join(dataset_path, "image"))
        print('directory created: {}'.format(dir_save_files))
    except:
        print('Unexpected error: ', sys.exc_info()[0])

    text_filenames, image_filenames = [], []
    labels_MVSA_Single, labels_MVSA_Multi = None, None

    for idx, dataset in enumerate(datasets_names):
        # change the working directory to the directory where we have all the data files of the current dataset
        os.chdir(datasets_path[idx])
        if dataset == 'MVSA-single':
            labels_MVSA_Single = pd.read_csv(labels_filename, sep = "\t").dropna()
            labels_MVSA_Single[['text', 'image']] = labels_MVSA_Single['text,image'].str.split(',', expand=True)
            labels_MVSA_Single.drop(columns=['text,image'], inplace=True)
            
            labels_MVSA_Single['text']  = labels_MVSA_Single['text'].apply(lambda x: sentiment_label[x])
            labels_MVSA_Single['image'] = labels_MVSA_Single['image'].apply(lambda x: sentiment_label[x])
        else:
            labels_MVSA_Multi = pd.read_csv(labels_filename, sep = "\t").dropna()
            col_list = [c for c in labels_MVSA_Multi.columns if 'text,image' in c]  # e.g. ['text,image', 'text,image.1', 'text,image.2']
            for i, c in enumerate(col_list, start=1):
                # Split into two new columns (textX, imageX):
                labels_MVSA_Multi[[f'text{i}', f'image{i}']] = labels_MVSA_Multi[c].str.split(',', expand=True)# Drop the old columns
                
            labels_MVSA_Multi.drop(columns=col_list, inplace=True)

            # Now you have: "ID", "text1", "image1", "text2", "image2", "text3", "image3"
            for col in ['text1', 'image1', 'text2', 'image2', 'text3', 'image3']:
                labels_MVSA_Multi[col] = labels_MVSA_Multi[col].apply(lambda x: sentiment_label[x])

        # Now read *all* files from "data" directory
        data_dir = os.path.join(datasets_path[idx], "data")  # => "/home/.../Datasets/MVSA-single/data"
        os.chdir(data_dir)

        # Use glob to find the .txt and .jpg files
        text_filenames.append([file for file in glob.glob("*.txt")])
        image_filenames.append([file for file in glob.glob("*.jpg")])


    # the final label of the samples in MVSA_Multi is represented by the majority vote of the annotators
    indexes = labels_MVSA_Multi[~((labels_MVSA_Multi['text1'] == labels_MVSA_Multi['text2']) 
        | (labels_MVSA_Multi['text2'] == labels_MVSA_Multi['text3']) | (labels_MVSA_Multi['text1'] == labels_MVSA_Multi['text3']))
        | ~((labels_MVSA_Multi['image1'] == labels_MVSA_Multi['image2']) | (labels_MVSA_Multi['image2'] == labels_MVSA_Multi['image3']) 
        | (labels_MVSA_Multi['image1'] == labels_MVSA_Multi['image3']))].index

    labels_MVSA_Multi.drop(indexes, inplace=True)

    labels_MVSA_Multi['text'] = labels_MVSA_Multi.filter(like = 'text').mode(axis = 1).iloc[:,0]
    labels_MVSA_Multi['image'] = labels_MVSA_Multi.filter(like = 'image').mode(axis = 1).iloc[:,0]

    labels_MVSA_Multi = labels_MVSA_Multi.drop(columns = ["text1", "text2", "text3", "image1", "image2", "image3"])

    # eliminate all the inconsistent data, i.e the posts where one of the labels is positive and the other is negative
    indexes = labels_MVSA_Single[((labels_MVSA_Single['text'] == 0) & (labels_MVSA_Single['image'] == 1)) \
        | ((labels_MVSA_Single['text'] == 1) & (labels_MVSA_Single['image'] == 0))].index
    labels_MVSA_Single.drop(indexes, inplace=True)
    labels_MVSA_Single.to_csv(os.path.join(dir_save_files, 'MVSA-single', 'labels.csv'), sep='\t', index=False)
    print('Number of valid Tweets in MVSA-Single: ', len(labels_MVSA_Single.index))

    indexes = labels_MVSA_Multi[((labels_MVSA_Multi['text'] == 0) & (labels_MVSA_Multi['image'] == 1)) \
        | ((labels_MVSA_Multi['text'] == 1) & (labels_MVSA_Multi['image'] == 0))].index
    labels_MVSA_Multi.drop(indexes, inplace=True)
    labels_MVSA_Multi.to_csv(os.path.join(dir_save_files, 'MVSA-multiple', 'labels.csv'), sep='\t', index=False)
    print('Number of valid Tweets in MVSA_Multi: ', len(labels_MVSA_Multi.index))

    # save the new datasets
    files_MVSA = [labels_MVSA_Single['ID'].astype(str).to_list() , labels_MVSA_Multi['ID'].astype(str).to_list()]
    for idx, dataset in enumerate(datasets_names):
        cwd = os.path.join(dir_save_files, dataset)
        # move all the text files to the destination directory
        move_files(
            files_MVSA[idx],
            '.txt',
            os.path.join(datasets_path[idx], "data"),         # <-- old dataset path
            os.path.join(dir_save_files, dataset, "text")             # <-- new folder
        )
        # move all the image files to the destination directory
        move_files(
            files_MVSA[idx],
            '.jpg',
            os.path.join(datasets_path[idx], "data"),       # <-- old dataset path
            os.path.join(dir_save_files, dataset, "image")            # <-- new folder
        )

else:
    print('directory {} exists'.format(dir_save_files))
    