# prepare_datasets.py

from helpers import *
import sys

OLD_DATASETS = ["MVSA-single", "MVSA-multiple"] # Old original dataset
NEW_DATASETS = ["MOA-MVSA-single", "MOA-MVSA-multiple"] # New target dataset

old_datasets_paths = [os.path.join(DATASET_BASE, d) for d in OLD_DATASETS]
new_dataset_paths = os.path.join(base_dir, 'Datasets')

moa_single_path = os.path.join(new_dataset_paths, "MOA-MVSA-single")
moa_multi_path  = os.path.join(new_dataset_paths, "MOA-MVSA-multiple")

text_extension_file = ".txt"
labels_filename = "labelResultAll.txt"

if not (os.path.exists(moa_single_path) and os.path.exists(moa_multi_path)):
    os.makedirs(moa_single_path,  exist_ok=True)
    os.makedirs(moa_multi_path,   exist_ok=True)
    os.makedirs(os.path.join(moa_single_path,  "text"), exist_ok=True)
    os.makedirs(os.path.join(moa_single_path,  "image"), exist_ok=True)
    os.makedirs(os.path.join(moa_multi_path,   "text"), exist_ok=True)
    os.makedirs(os.path.join(moa_multi_path,   "image"), exist_ok=True)
    print("Creating dataset in:", moa_single_path)
    print("Creating dataset in:", moa_multi_path)
    print("Files in MOA-MVSA-single text folder:", os.listdir(os.path.join(moa_single_path, "text")))
    print("Files in MOA-MVSA-single image folder:", os.listdir(os.path.join(moa_single_path, "image")))

    text_filenames, image_filenames = [], []
    labels_MVSA_Single, labels_MVSA_Multi = None, None

    for idx, old_dataset_name in enumerate(OLD_DATASETS):
        os.chdir(old_datasets_paths[idx])
        if old_dataset_name == 'MVSA-single':
            labels_MVSA_Single = pd.read_csv(labels_filename, sep = "\t").dropna()
            labels_MVSA_Single[['text', 'image']] = labels_MVSA_Single['text,image'].str.split(',', expand=True)
            labels_MVSA_Single.drop(columns=['text,image'], inplace=True)
            
            labels_MVSA_Single['text']  = labels_MVSA_Single['text'].apply(lambda x: sentiment_label[x])
            labels_MVSA_Single['image'] = labels_MVSA_Single['image'].apply(lambda x: sentiment_label[x])
        else:
            labels_MVSA_Multi = pd.read_csv(labels_filename, sep = "\t").dropna()
            col_list = [c for c in labels_MVSA_Multi.columns if 'text,image' in c]
            for i, c in enumerate(col_list, start=1):
                # Split into two new columns (textX, imageX):
                labels_MVSA_Multi[[f'text{i}', f'image{i}']] = labels_MVSA_Multi[c].str.split(',', expand=True)# Drop the old columns
                
            labels_MVSA_Multi.drop(columns=col_list, inplace=True)

            for col in ['text1', 'image1', 'text2', 'image2', 'text3', 'image3']:
                labels_MVSA_Multi[col] = labels_MVSA_Multi[col].apply(lambda x: sentiment_label[x])

        # read *all* files from "data" directory
        data_dir = os.path.join(old_datasets_paths[idx], "data") 
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

    # eliminate all the inconsistent data. the posts where one of the labels is positive and the other is negative
    indexes = labels_MVSA_Single[((labels_MVSA_Single['text'] == 0) & (labels_MVSA_Single['image'] == 1)) \
        | ((labels_MVSA_Single['text'] == 1) & (labels_MVSA_Single['image'] == 0))].index
    labels_MVSA_Single.drop(indexes, inplace=True)
    labels_MVSA_Single.to_csv(os.path.join(new_dataset_paths, 'MOA-MVSA-single', 'labels.csv'), sep='\t', index=False)
    print('Number of valid Tweets in MOA-MVSA-single: ', len(labels_MVSA_Single.index))

    indexes = labels_MVSA_Multi[((labels_MVSA_Multi['text'] == 0) & (labels_MVSA_Multi['image'] == 1)) \
        | ((labels_MVSA_Multi['text'] == 1) & (labels_MVSA_Multi['image'] == 0))].index
    labels_MVSA_Multi.drop(indexes, inplace=True)
    labels_MVSA_Multi.to_csv(os.path.join(new_dataset_paths, 'MOA-MVSA-multiple', 'labels.csv'), sep='\t', index=False)
    print('Number of valid Tweets in MOA-MVSA-multiple: ', len(labels_MVSA_Multi.index))

    # save the new datasets
    files_MVSA = [labels_MVSA_Single['ID'].astype(str).to_list() , labels_MVSA_Multi['ID'].astype(str).to_list()]
    for idx, dataset in enumerate(datasets_names):
        cwd = os.path.join(new_dataset_paths, dataset)
        # move all the text files to the destination directory
        move_files(
            files_MVSA[idx],
            '.txt',
            os.path.join(old_datasets_paths[idx], "data"), # old dataset path
            os.path.join(new_dataset_paths, dataset, "text") # new folder
        )
        # move all the image files to the destination directory
        move_files(
            files_MVSA[idx],
            '.jpg',
            os.path.join(old_datasets_paths[idx], "data"), # old dataset path
            os.path.join(new_dataset_paths, dataset, "image") # new folder
        )

else:
    print(f'directory {moa_single_path} exists')
    print(f'directory {moa_multi_path} exists')