from flask import render_template, request, Response
from flaskApp import app
import pandas as pd
import time
import os
import shutil
import json
from gensim import corpora
import pyLDAvis
from pyLDAvis import gensim as gs
import nltk


from utilities.document_preprocessing import preprocess_documents
from models.topic_model import Topic_Model
from models.embeddings_model import get_embeddings
from utilities.visualizations import get_employee_clusters
from utilities.utils import get_cluster_text, get_employee_text

nltk.download("crubadan")
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")


@app.route("/", methods=["GET"])
@app.route("/home")
def home():

    return render_template("home.html")


@app.route("/analytic")
def analytic():
    # read the details of precomputed configurations from metadata file
    try:
        with open(metadata_path) as json_file:
            data = json.load(json_file)
            n_grams = data["selected_precompute_n_grams"]
            num_clusters = data["selected_precompute_k"]
        return render_template(
            "analytic.html",
            models=models,
            num_clusters=num_clusters,
            n_grams=n_grams,
            visualizations=visualizations,
        )
    except:
        return render_template("results.html", success=False, algorithm={})


@app.route("/upload", methods=["GET", "POST"])
def upload():
    return render_template(
        "upload.html", precompute_k=precompute_k, precompute_ngrams=precompute_ngrams
    )


@app.route("/precompute", methods=["GET", "POST"])
def precompute():
    # global variables to be used by progress function
    global temp_data_folder, destination, selected_precompute_k, selected_precompute_n_grams
    if request.method == "POST":
        try:
            temp_data_folder = os.path.join(base_folder, "temp")
            if not os.path.exists(temp_data_folder):
                os.makedirs(temp_data_folder)
            upload = request.files.get("file")
            if not upload.filename:
                return render_template(
                    "precompute.html",
                    success=False,
                    filename=upload.filename,
                    message="Please Select an input file.",
                )
            filename = upload.filename
            destination = os.path.join(temp_data_folder, filename)
            upload.save(destination)

            selected_precompute_k = [
                int(k) for k in request.form.getlist("precompute_k")
            ]
            selected_precompute_n_grams = [
                int(gram) for gram in request.form.getlist("precompute_ngrams")
            ]

            if len(selected_precompute_k) == 0 or len(selected_precompute_n_grams) == 0:

                return render_template(
                    "precompute.html",
                    success=False,
                    filename=upload.filename,
                    message="Please Select values for k and n-gram.",
                )
            return render_template("progress.html", filename=filename)
        except:
            return render_template(
                "precompute.html",
                success=False,
                filename=filename,
                message="Failed to precompute data. Please Try again or contact Admin",
            )


@app.route("/progress")
def progress():
    def compute(
        temp_data_folder,
        destination,
        selected_precompute_k,
        selected_precompute_n_grams,
    ):

        progress = 0
        try:
            while progress != 100:

                start_time = time.time()
                success = False
                metadata = dict()
                metadata["selected_precompute_k"] = selected_precompute_k
                metadata["selected_precompute_n_grams"] = selected_precompute_n_grams

                # reading and preprocessing the data
                ids, sentences, token_lists = preprocess_documents(
                    destination,
                    verbose=False,
                    dr=temp_data_folder,
                    max_sent_length=1001,
                )

                # generating all possible combinations to compute
                completed_tasks = 0
                # to get the percentage done for progress bar in the front end
                total_tasks = (
                    len(selected_precompute_k) * len(base_models) * len(embeddings)
                )

                for embedding in embeddings:

                    if embedding == None:
                        embedding_path = None
                    else:
                        # computing path to store the selected embedding vector
                        embedding_path = os.path.join(
                            temp_data_folder, embedding + "_embeddings.json"
                        )

                        _ = get_embeddings(
                            model_name=embedding,
                            sentences=sentences,
                            batch_size=100,
                            dr=temp_data_folder,
                        )

                    for k in selected_precompute_k:

                        for base in base_models:
                            yield "data:" + str(progress) + "\n\n"
                            completed_tasks += 1
                            progress = int((completed_tasks / total_tasks) * 100)

                            if embedding == None and base == None:
                                # if embedding and base model, both are not selected
                                continue

                            for visualization in visualizations:

                                vis_id = visualization["i"]
                                # visualization is LDA vis which only works with LDA, skip the iteration when embedding in On
                                if vis_id == 3 and base == None:
                                    continue

                                for gram in selected_precompute_n_grams:

                                    # construct the path to the output directory for the selected combination
                                    output_folder = os.path.join(
                                        temp_data_folder,
                                        str(vis_id),
                                        "base_" + str(base),
                                        "Embedding_" + str(embedding),
                                        "k_" + str(k),
                                        "ngram_" + str(gram),
                                    )

                                    # create the folder
                                    if not os.path.exists(output_folder):
                                        os.makedirs(output_folder)

                                    print(output_folder)
                                    # Create a Topic Model Object
                                    TopicModelObject = Topic_Model(
                                        base=base,
                                        k=k,
                                        verbose=False,
                                        embedding=embedding,
                                        saved_embedding_dir=embedding_path,
                                        embedding_batch_size=100,
                                        use_AE=False,
                                        output_attention=False,
                                    )

                                    # Fit the topic model
                                    TopicModelObject.fit(
                                        sentences, token_lists, vis_clusters=False
                                    )

                                    if vis_id == 2:  # wordcloud
                                        # generating and saving the word clouds for the defined clusters
                                        TopicModelObject.wordcloud(
                                            sentences, gram, output_folder
                                        )

                                    elif vis_id == 0 or vis_id == 1:
                                        # if visualization is Employee Clusters using plotly, get the features and save in a pickle file
                                        averaged_employee_features = TopicModelObject.compress_features(
                                            ids
                                        )

                                        labels = TopicModelObject.get_labels()
                                        # pass it to the function to visualize the data
                                        cluster_titles = get_cluster_text(
                                            sentences, labels, gram
                                        )

                                        # constructing hover text (all objectives belonging to that employee)
                                        employee_titles = get_employee_text(
                                            sentences, ids
                                        )

                                        # calling the visualization function
                                        get_employee_clusters(
                                            averaged_employee_features,
                                            cluster_titles,
                                            employee_titles,
                                            vis=vis_id,
                                            dir=os.path.join(output_folder),
                                        )

                                    elif vis_id == 3:

                                        id2word = corpora.Dictionary(token_lists)
                                        corpus = [
                                            id2word.doc2bow(text)
                                            for text in token_lists
                                        ]

                                        vis = gs.prepare(
                                            TopicModelObject.basemodel, corpus, id2word
                                        )
                                        pyLDAvis.save_html(
                                            vis,
                                            os.path.join(output_folder, "lda_vis.html"),
                                        )

                            print(
                                "completed {} out of {} with progress {} %".format(
                                    completed_tasks, total_tasks, progress
                                )
                            )

                # if the precompute process is a success, delete the previous data and replace it with the new one
                data_folder = os.path.join(base_folder, "flask")
                temp_data_folder = os.path.join(base_folder, "temp")

                # write the metadata related to precomputed stuff into a file
                metadata_path = os.path.join(temp_data_folder, "metadata.json")
                with open(metadata_path, "w") as outfile:
                    json.dump(metadata, outfile)
                if os.path.exists(data_folder):
                    shutil.rmtree(data_folder)
                shutil.move(temp_data_folder, data_folder)

                print("Precompute took {} seconds".format(time.time() - start_time))
                yield "data:" + str(progress) + "\n\n"

        except:
            progress = -1
            yield "data:" + str(progress) + "\n\n"
            if os.path.exists(temp_data_folder):
                shutil.rmtree(temp_data_folder)

    return Response(
        compute(
            temp_data_folder,
            destination,
            selected_precompute_k,
            selected_precompute_n_grams,
        ),
        mimetype="text/event-stream",
    )


@app.route("/result", methods=["GET", "POST"])
def result():

    # define few global variables to be used throughout
    global sentences, algorithm, TopicModelObject, token_lists, images, ids

    # reading the configurations
    algorithm = dict()
    # compute model parameters

    try:
        model = request.form.get("selected_model")

        if model == "LDA Model + Bert Embedding":
            algorithm["base"] = "LDA"
            algorithm["embedding"] = "bert-sent"

        elif model == "LDA Model + Roberta Embedding":
            algorithm["base"] = "LDA"
            algorithm["embedding"] = "roberta-sent"

        elif model == "Bert Embedding":
            algorithm["base"] = None
            algorithm["embedding"] = "bert-sent"

        elif model == "Roberta Embedding":
            algorithm["base"] = None
            algorithm["embedding"] = "roberta-sent"

        elif (
            model == "LDA Model" or model == None
        ):  # it is disabled or LDA model is selected
            algorithm["base"] = "LDA"
            algorithm["embedding"] = None

        # compute number of cluster
        algorithm["k"] = int(request.form.get("selected_k"))

        # compute n-gram
        if request.form.get("selected_ngram") == None:  # if it is disabled
            algorithm["n_gram"] = 1
        else:  # get selected value
            algorithm["n_gram"] = int(request.form.get("selected_ngram"))

        # compute visualization
        algorithm["visualization"] = visualizations[
            int(request.form.get("selected_vis"))
        ]["i"]

        # no embeddings would be used automatically if pyLDAvis is used for visualization as it only works on LDA alone
        # fail-case in case the frontend fails
        if algorithm["visualization"] == 3:

            algorithm["base"] = "LDA"
            algorithm["embedding"] = None

        # construct path to the folder where all the precomputed results are stored,

        output_folder = os.path.join(
            "flask",
            str(algorithm["visualization"]),
            "base_" + str(algorithm["base"]),
            "Embedding_" + str(algorithm["embedding"]),
            "k_" + str(algorithm["k"]),
            "ngram_" + str(algorithm["n_gram"]),
        )

    except:
        return render_template(
            "results.html", algorithm=algorithm, images=[], success=False, viz=[]
        )

    # get list of all word cloud images from the directory
    extensions = [".jpg", ".jpeg", ".png"]
    images = [
        f
        for f in os.listdir(os.path.join(base_folder, output_folder))
        if os.path.splitext(f)[1] in extensions
    ]

    # constructing path to the images inside the static folder
    images = [
        {"id": i + 1, "path": os.path.join(output_folder, image)}
        for i, image in enumerate(images)
    ]

    # this is a hack I am using to split images into 2 per row in html because my development skills suck
    img_count = 0
    temp_images = []
    images_row = []
    num_images_per_row = 2
    for image in images:
        if img_count == num_images_per_row:
            temp_images.append(images_row)
            img_count = 0
            images_row = []
        img_count += 1
        images_row.append(image)
    temp_images.append(images_row)

    images = temp_images
    # get paths to all htmls for 2d/3d cluster visualization and LDA vis
    html = [
        f
        for f in os.listdir(os.path.join(base_folder, output_folder))
        if os.path.splitext(f)[1] in [".html"]
    ]

    # constructing path to the images inside the static folder
    html = [
        {"id": i + 1, "path": os.path.join(output_folder, viz)}
        for i, viz in enumerate(html)
    ]

    return render_template(
        "results.html", algorithm=algorithm, images=images, success=True, html=html
    )


# define the permanent folder for the data
base_folder = "flaskApp/static/"
processed_file = "use_this_data.pkl"

models = [
    "Bert Embedding",
    "LDA Model",
    "LDA Model + Bert Embedding",
    "LDA Model + Roberta Embedding",
    "Roberta Embedding",
]


n_grams = [1, 2]
num_clusters = [2, 3, 4]
embeddings = ["bert-sent", "roberta-sent", None]
base_models = ["LDA", None]
precompute_k = [2, 3, 4, 5, 6, 7, 8]
precompute_ngrams = [1, 2, 3]
visualizations = [
    {"i": 0, "name": "Job Cluster 2D", "path": "static/gui/cluster_2D.png"},
    {"i": 1, "name": "Job Cluster 3D", "path": "static/gui/cluster_3D.png"},
    {"i": 2, "name": "Word Cloud", "path": "static/gui/wordcloud.png"},
    {"i": 3, "name": "LDA Vis", "path": "static/gui/lda_vis.png"},
]

# read the metadata related to the computed combinations of k and ngrams
metadata_path = os.path.join(base_folder, "flask", "metadata.json")
if os.path.exists(metadata_path):
    with open(metadata_path) as json_file:
        data = json.load(json_file)
        n_grams = data["selected_precompute_n_grams"]
        num_clusters = data["selected_precompute_k"]
else:
    print("WARNING: Metadata file not found")
