import pathlib
import pandas as pd
from flask import Flask, render_template, send_file, redirect, request, url_for
from helpers import get_random_image, initialize_database
from prepare_training_data import prepare_training_data
from train_predictor import train_predictor
from predict_score import predict_score, validate_prediction
from export_prediction import export_prediction

app = Flask(__name__)

@app.route('/')
def images():
    global image_batch, image_batches, width, start_flag
    image_batches += [image_batch]
    if len(image_batches) > 16:
        image_batches = image_batches[-16:]
    #print(image_batches)
    return render_template('index.html', images=image_batch, width=width)

@app.route('/start')
def start():
    return render_template("start.html")

@app.route('/start_session', methods=['POST'])
def start_session():
    global root_folder, database_file, n_samples, image_batches, image_batch, width, database
    root_folder = request.form['image-folder']
    database_file = request.form['database-file']
    n_samples = int(request.form['n_samples'])
    
    # init new session
    image_batches = []
    database = initialize_database(root_folder,database_file)
    assert len(database) > n_samples, "change n_samples"
    image_batch = get_random_image(database, n_samples)
    return redirect('/')

@app.route('/refresh')
def refresh():
    global image_batch, image_batches
    image_batch = get_random_image(database, n_samples)
    return redirect('/')

@app.route('/back')
def back():
    global image_batch, image_batches
    if len(image_batches) > 2:
        image_batch = image_batches[-2]
        image_batches = image_batches[:-2]
    return redirect('/')

@app.route('/train', methods=['POST'])
def train():
    global train_from
    train_from = request.form['train_from']
    return render_template("training.html")

@app.route('/training')
def training():
    global train_from
    prepare_training_data(root_folder,database_file,train_from)
    train_predictor(root_folder,database_file,train_from)
    predict_score(root_folder,database_file,train_from)
    validate_prediction(root_folder,database_file,train_from)
    return redirect('/')

@app.route('/export', methods=['POST'])
def export():
    global export_from
    export_from = request.form['export_from']
    return render_template("exporting.html")

@app.route('/exporting')
def exporting():
    global export_from
    export_prediction(root_folder,database_file,export_from)
    return redirect('/')
    
@app.route('/image/<image_name>')
def image(image_name):
    image_path = database.loc[database["name"]==image_name, "path"].item()
    return send_file(image_path, mimetype='image/jpeg')

@app.route('/assign_metadata/<image_name>', methods=['POST'])
def assign_metadata(image_name):

    global image_batch, image_batches

    # parse request
    col = request.form['col']
    val = request.form['val']
    show = request.form['show']
    if col in ["score"]:
        val = float(val)
    if col in ["mark"]:
        val = int(val)
    if show == "False":
        show = False
    else:
        show = True

    # update dataframe
    database_path = pathlib.Path(root_folder) / database_file
    image_path = database.loc[database["name"]==image_name, "path"].item()
    database.loc[database["path"]==image_path, col] = val
    database.loc[database["path"]==image_path, "show"] = show
    database.to_csv(database_path, index=False)

    # find replacement image
    if show == False:
        i = 0
        index = image_batch.index(image_name)
        new_batch = image_batch.copy()
        while i < len(database):
            new_image = get_random_image(database, 1)
            i += 1
            if new_image[0] not in new_batch:
                new_batch[index] = new_image[0]
                image_batch = new_batch
                break

    return redirect('/')

if __name__ == '__main__':

    # init globals
    root_folder = None
    database_file = None
    n_samples = None
    width = 512
    start_flag = True
    image_batches = []
    image_batch = []
    database = pd.DataFrame({})

    app.run()
