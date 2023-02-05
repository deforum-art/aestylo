import pathlib
import pandas as pd
from typing import List


def initialize_database(root_folder,database_file):

    # get images in folder
    image_path = pathlib.Path(root_folder)
    extensions = ['.jpg', '.jpeg', '.png', '.gif']
    image_path_list = []
    for path in image_path.rglob('*'):
        if path.is_file() and path.suffix in extensions:
            image_path_list.append(str(path))
    num = len(image_path_list)

    # create dataframe
    df = pd.DataFrame({'name': [pathlib.Path(path).name for path in image_path_list],
                       'path': image_path_list,
                       'flag': [0] * num,
                       'flag_pred': [0] * num,
                       'label': [0] * num,
                       'label_pred': [0] * num,
                       'score': [0] * num,
                       'score_pred': [0] * num,
                       'show': [True] * num,})
    df = df.sort_values(by="name").reset_index(drop=True)

    # database path
    database_path = pathlib.Path(root_folder) / database_file

    # create/load database
    if database_path.exists():
        # load existing database
        old_df = pd.read_csv(database_path)
        # add new images
        df = pd.concat([old_df, df[~df['name'].isin(old_df['name'])]])
        df = df.sort_values(by="name").reset_index(drop=True)
        df.to_csv(database_path, index=False)
    else:
        # create new database
        df.to_csv(database_path, index=False)

    return df

def get_random_image(database, n_samples=4):
    unmarked_images = database[database['show'] == True]
    random_images = unmarked_images.sample(n_samples)
    return list(random_images['name'])