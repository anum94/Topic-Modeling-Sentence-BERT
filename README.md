## Topic Modeling using BERT Embedding on Job Description Dataset

The goal of this project is to cluster jobs based on their description.This project uses classical NLP techniques as well as
state-of-the-art deep learning approaches.
#### Keywords: LDA, Transformers, K-means, TF-IDF, Word Embedding

#### quick run through docker image
There is a flask project that encapsulates the whole project is available as a docker image.
This image shows result on the whole dataset.

To pull Docker image:
> docker pull anumafzal/topic-model-employee-objective

To run the pulled image:
> docker run --rm -it -p 5000:5000/tcp anumafzal/topic-model-employee-objective:latest


check ports http://127.0.0.1:5000/ or http://0.0.0.0:5000/

If anyone is interested in the models used, there is a jupyter notebook that can be used for experiemtation.

### 1) Datasets:

- job description Dataset. Available on Kaggle
https://www.kaggle.com/airiddha/trainrev1

### 2) Directory Structure:

- ### merck-employee-objectives
    - #### __data (folder):__
        - all the data files are stored here such as the job description dataset and the merck objective dataset are to be stored in this folder
    - #### __flaskApp (folder):__
        - This folder contains all the files needed for the flask App
    - #### __models (folder):__
        This folder contains the models that are used through out the framework.
        - __autoencoder_model.py:__ Contains an autoencoder used to learn the latent representations of the features.
        - __embeddings_models.py:__ All functions related to transformer models are found in this file.
        - __topic_model.py:__ This file contains a topic model class whihc is the backbone of this famework
    - #### __utilities (folder):__
        All utility functions can be found in this folder
    - #### __app.py (file):__
    the starting point for the flask app
    - #### __Dockerfile (file):__
    Docker file for creating a docker image
    - #### __enviroment.yml(file)__ :
    File to create a conda enviroment to run this project. Use the following command to create the enviroment
    > conda env create -f enviroment.yml
    - #### __README.md (file):__
    this file
    - ####__topic-modeling-playground-colab.ipynb (Notebook):__
    Advanced level notebook for doing analytic on the dataset

### 3) Code Entry point:

There are 2 ways to use this framework.
- __Flask App__ which provides a gui to interact with the backend. simply run the command from project directory. This is recommended if you are non-technical user or just want quickly get an insight into the framework.
    > flask run

- __Jupyter Notebook__ (topic-modeling-playground-colab.ipynb) which basically provides the same insight as the flask App but allows you to dive into the analytics and also play around.\
 Just run the notebook using jupyterlab. You can run it also with google colab if you want everything to run faster. But for a small dataset, running on the good computer is also ok.\
  The notebook has comments explaining which extra cells need to be run in case of colab. \
  Make sure you upload the whole directory to your google drive and then run the notebook as it looks for all the python packages.

### 4) Visualizations / Evaluations:

#### Wordcloud
   - Each image(wordcloud) represents a cluster of employees with similiar objective.
   - The words in each wordcloud are the main topics discussed by the employees belonging to that cluster

#### PyLDAvis
This visualization is meant for the LDA model only
    - This visualization gives an indepth analysis of the base model in terms of probabilities.
    - It includes information such as the words and the probability of them belonging to a cluster.
#### Employee clusters in 2D
   - In the visualization, the employees can be visualized in 2D space.
   - Each dot in the graph represents an employee and by hover some basic information of the employee can be seen.
   - The separate colors represent each cluster and on the top right, the topic words for each cluster can be seen
#### Employee clusters in 3D
   - In the visualization, the employees can be visualized in 3D space.
   - Each dot in the graph represents an employee and by hover some basic information of the employee can be seen.
   - The separate colors represent each cluster and on the top right, the topic words for each cluster can be seen.
   - It is also possible to rotate the space by left clicking the mouse and moving it and zooming into the space by using the small
        wheel on the the mouse.

### 5) Good to Know
- The framework operates on english language and hence ignores all text belonging to foreign languages
- It concatenates the all the objectives belonging to a user into one long objective as part of the proprocessing
- It drops all duplicated entries from the data as part of the proprocessing step
