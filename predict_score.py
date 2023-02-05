import torch
from PIL import Image
import clip
from tqdm import tqdm
import pandas as pd
import pathlib

from train_predictor import MLP
from prepare_training_data import normalized


def predict_score(root_folder, database_file, train_from, clip_model="ViT-L/14"):
    path = pathlib.Path(root_folder)
    database_path = path / database_file
    database = pd.read_csv(database_path)
    model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
    model_name = f"linear_predictor_{clip_model.replace('/', '').lower()}_{train_from}_mse.pth"
    s = torch.load(path / model_name)   # load the model you trained previously or the model available in this repo
    model.load_state_dict(s)
    model.to("cuda")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model2, preprocess = clip.load("ViT-L/14", device=device)  #RN50x64   
    
    pred = []
    for i, row in tqdm(database.iterrows(), total=database.shape[0]):
        pil_image = Image.open(row["path"])
        image = preprocess(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
           image_features = model2.encode_image(image)
        im_emb_arr = normalized(image_features.cpu().detach().numpy() )
        prediction = model(torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor))
        pred.append(prediction.item())

    if train_from == "label":
        database["label_pred"] = pred
    elif train_from == "score":
        database["score_pred"] = pred

    database.to_csv(database_path,index=False)
    return database