import clip
import torch
import pathlib
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)
    

def prepare_training_data(root_folder,database_file,train_from,clip_model="ViT-L/14"):
    path = pathlib.Path(root_folder)
    database_path = path / database_file
    database = pd.read_csv(database_path)

    if train_from == "label":
        df = database[database.label!=0].reset_index(drop=True)
    elif train_from == "score":
        df = database[database.score!=0].reset_index(drop=True)

    out_path = pathlib.Path(root_folder)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(clip_model, device=device)
    
    x = []
    y = []
    
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):

        if train_from == "label":
            average_rating = float(row.label)
        elif train_from == "score":
            average_rating = float(row.score)
        
        if average_rating < 1:
            continue

        try:
            image = preprocess(Image.open(row.path)).unsqueeze(0).to(device)
        except:
            continue

        with torch.no_grad():
            image_features = model.encode_image(image)

        im_emb_arr = image_features.cpu().detach().numpy() 
        x.append(normalized(im_emb_arr))
        y.append([average_rating])

    x = np.vstack(x)
    y = np.vstack(y)
    x_out = f"x_{clip_model.replace('/', '').lower()}_ebeddings.npy"
    y_out = f"y_{train_from}.npy"
    np.save(out_path / x_out, x)
    np.save(out_path / y_out, y)
    return