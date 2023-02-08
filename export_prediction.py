import pandas as pd
import shutil
from pathlib import Path

def export_prediction(root_folder, database_file, export_from):
    prefix = database_file.split(".")[0]
    database_path = Path(root_folder) / database_file
    df = pd.read_csv(database_path)

    if export_from == "label":
        df = df[["path", "label"]]
        export_folder = Path(root_folder) / f"{prefix}_export_label"
        subfolder_good = export_folder / "good"
        subfolder_bad = export_folder / "bad"
        
        if not subfolder_good.exists():
            subfolder_good.mkdir(parents=True)
        if not subfolder_bad.exists():
            subfolder_bad.mkdir(parents=True)
        
        for index, row in df.iterrows():
            image_path = Path(root_folder) / row["path"]
            if row["label"] == 2:
                shutil.copy2(image_path, subfolder_good)
            elif row["label"] == 1:
                shutil.copy2(image_path, subfolder_bad)
    
    elif export_from == "label_pred":
        df = df[["path", "label_pred"]]
        export_folder = Path(root_folder) / f"{prefix}_export_label_pred"
        subfolder_good = export_folder / "good"
        subfolder_bad = export_folder / "bad"
        
        if not subfolder_good.exists():
            subfolder_good.mkdir(parents=True)
        if not subfolder_bad.exists():
            subfolder_bad.mkdir(parents=True)
        
        for index, row in df.iterrows():
            image_path = Path(root_folder) / row["path"]
            if 1.5 <= row["label_pred"] <= 2:
                shutil.copy2(image_path, subfolder_good)
            elif 1.0 <= row["label_pred"] < 1.5:
                shutil.copy2(image_path, subfolder_bad)

    elif export_from == "score":
        df = df[["path", "score"]]
        export_folder = Path(root_folder) / f"{prefix}_export_score"
        
        subfolder_1 = export_folder / "1"
        subfolder_2 = export_folder / "2"
        subfolder_3 = export_folder / "3"
        subfolder_4 = export_folder / "4"
        subfolder_5 = export_folder / "5"
        
        if not subfolder_1.exists():
            subfolder_1.mkdir(parents=True)
        if not subfolder_2.exists():
            subfolder_2.mkdir(parents=True)
        if not subfolder_3.exists():
            subfolder_3.mkdir(parents=True)
        if not subfolder_4.exists():
            subfolder_4.mkdir(parents=True)
        if not subfolder_5.exists():
            subfolder_5.mkdir(parents=True)
        
        for index, row in df.iterrows():
            image_path = Path(root_folder) / row["path"]
            if row["score"] == 1:
                shutil.copy2(image_path, subfolder_1)
            elif row["score"] == 2:
                shutil.copy2(image_path, subfolder_2)
            elif row["score"] == 3:
                shutil.copy2(image_path, subfolder_3)
            elif row["score"] == 4:
                shutil.copy2(image_path, subfolder_4)
            elif row["score"] == 5:
                shutil.copy2(image_path, subfolder_5)
    
    elif export_from == "score_pred":
        df = df[["path", "score_pred"]]
        export_folder = Path(root_folder) / f"{prefix}_export_score_pred"
        
        subfolder_1_2 = export_folder / "1_2"
        subfolder_2_3 = export_folder / "2_3"
        subfolder_3_4 = export_folder / "3_4"
        subfolder_4_5 = export_folder / "4_5"
        
        if not subfolder_1_2.exists():
            subfolder_1_2.mkdir(parents=True)
        if not subfolder_2_3.exists():
            subfolder_2_3.mkdir(parents=True)
        if not subfolder_3_4.exists():
            subfolder_3_4.mkdir(parents=True)
        if not subfolder_4_5.exists():
            subfolder_4_5.mkdir(parents=True)
        
        for index, row in df.iterrows():
            image_path = Path(root_folder) / row["path"]
            if row["score_pred"] < 2:
                shutil.copy2(image_path, subfolder_1_2)
            elif 2 <= row["score_pred"] < 3:
                shutil.copy2(image_path, subfolder_2_3)
            elif 3 <= row["score_pred"] < 4:
                shutil.copy2(image_path, subfolder_3_4)
            elif row["score_pred"] >= 4:
                shutil.copy2(image_path, subfolder_4_5)

    if export_from == "flag":
        df = df[["path", "flag"]]
        export_folder = Path(root_folder) / f"{prefix}_export_flag"
        
        if not export_folder.exists():
            export_folder.mkdir(parents=True)
        
        for index, row in df.iterrows():
            image_path = Path(root_folder) / row["path"]
            if row["flag"] == 1:
                shutil.copy2(image_path, export_folder)

    return