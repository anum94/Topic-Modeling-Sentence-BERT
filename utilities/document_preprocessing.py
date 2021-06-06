import re
import nltk
from stop_words import get_stop_words
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
from langdetect import detect
import os
from utilities.utils import length_analysis
from nltk.stem.wordnet import WordNetLemmatizer

# downloads


# sentence level cleaning
def preprocess_sent(s):
    """
    :param s: string to be processed
    :return: processed string: see comments in the source code for more info
    """

    # normalization 1: xxxThis is a --> xxx. This is a (missing delimiter)
    s = re.sub(r"([a-z])([A-Z])", r"\1\. \2", s)  # before lower case
    # normalization 2: lower case
    s = s.lower()
    # normalization 3: "&gt", "&lt"
    s = re.sub(r"&gt|&lt", " ", s)
    # normalization 4: letter repetition (if more than 2)
    s = re.sub(r"([a-z])\1{2,}", r"\1", s)
    # normalization 5: non-word repetition (if more than 1)
    s = re.sub(r"([\W+])\1{1,}", r"\1", s)
    # normalization 6: string * as delimiter
    s = re.sub(r"\*|\W\*|\*\W", ". ", s)
    # normalization 7: stuff in parenthesis, assumed to be less informal
    s = re.sub(r"\(.*?\)", ". ", s)
    # normalization 8: xxx[?!]. -- > xxx.
    s = re.sub(r"\W+?\.", ".", s)
    # normalization 9: [.?!] --> [.?!] xxx
    s = re.sub(r"(\.|\?|!)(\w)", r"\1 \2", s)
    # normalization 10: ' ing ', noise text
    s = re.sub(r" ing ", " ", s)
    # normalization 11: noise text
    s = re.sub(r"product received for free[.| ]", " ", s)
    # normalization 12: phrase repetition
    s = re.sub(r"(.{2,}?)\1{1,}", r"\1", s)
    # remove all special characters
    s = re.sub("[^A-Za-z0-9]+", " ", s)

    return s.strip()


## word level preprocess


# filtering out punctuations and numbers
def f_punct(w_list):
    """
    :param w_list: word list to be processed
    :return: w_list with punct and number filter out
    """
    return [word for word in w_list if word.isalpha()]


# selecting nouns
def f_noun(w_list):
    """
    :param w_list: word list to be processed
    :return: w_list with only nouns selected
    """
    return [
        word for (word, pos) in nltk.pos_tag(w_list) if (pos[:2] == "NN")
    ]  # or pos[:2] == 'JJ')]


def f_stem(w_list):
    """
    :param w_list: word list to be processed
    :return: w_list with stemming
    """
    # stemming if doing word-wise
    # p_stemmer = PorterStemmer()
    # return [p_stemmer.stem(word) for word in w_list]
    wnl = WordNetLemmatizer()
    return [wnl.lemmatize(word) for word in w_list]


def f_stopw(w_list):
    """
    filtering out stop words
    """
    stop_words = list(set(get_stop_words("en")))
    return [word for word in w_list if word not in stop_words]


# wrapper function around all word level proprocessing utilities
def preprocess_word(s):
    """
    Get word level preprocessed data from preprocessed sentences
    including: remove punctuation, select noun, stem, stop_words
    :param s: sentence to be processed
    :return: word level pre-processed list
    """
    w_list = word_tokenize(s)
    w_list = f_punct(w_list)
    w_list = f_noun(w_list)
    w_list = f_stem(w_list)
    w_list = f_stopw(w_list)

    return w_list


# wrapper function to preprocess all the sentences in a document
def preprocess_documents(
    file=None,
    verbose=False,
    dr="output/preprocessed_data/",
    subset=None,
    max_sent_length=None,
):
    """
        Read the file and preprocess the data

    :param file(str): path to the input file
    :param id_col: column to read for the ids
    :param text_index (list): columns to read for text. concatinate text from all these column
    :param verbose:
    :param dr: path to where the .pkl file should be saved
    :return:
    """

    if file is None:
        print("Please provide a valid file path")
        return

    print("Reading the file")

    _, file_extension = os.path.splitext(file)

    if file_extension == ".csv":
        documents = pd.read_csv(file)
    elif file_extension == ".xlsx":
        documents = pd.read_excel(file)
    else:
        print("File extension not known. Please provide a .csv or .xlsx file")
        return

    # replace nans with empty spaces to avoid trouble in preprocessing
    documents = documents.replace(np.nan, "", regex=True)

    # dropping all duplicates because the data has a lot of redundant rows
    documents.drop_duplicates(inplace=True)

    # merge all records of an employee into one
    documents = documents.groupby("Id", as_index=False).agg(
        {"Title": " ".join, "FullDescription": " ".join}
    )
    id = documents["Id"]
    docs = (
        documents["Title"].astype(str)
        + " \n "
        + documents["FullDescription"].astype(str)
    )

    # type cast to list for easier processing
    docs = list(docs)
    if subset != None:
        id = id[:subset]
        docs = docs[:subset]

    if max_sent_length is None:
        max_sent_length = 400

    # get a quick analysis on the number of words in each objective
    allowed_length = length_analysis(docs)  # max num of words used from each objective

    if (
        allowed_length > max_sent_length
    ):  # because a lot of embeddings don't allow to get an embedding for text with more than 500 words
        print(
            "Allowed length according to data is {} but since it is greater than {}, reducing it to {}".format(
                allowed_length, max_sent_length, max_sent_length
            )
        )
        allowed_length = max_sent_length

    n_docs = len(docs)

    print("Preprocessing raw texts \n")

    sentences_list = []  # sentence level preprocessed
    token_lists = []  # word level preprocessed

    id_drop = (
        []
    )  # place holder to save indexes for entries with non-english langauges to be dropped later.
    for idx in range(n_docs):
        text = docs[idx]

        if idx % 1000 == 0:
            print(idx)

        try:
            detected_language = detect(text)

        except Exception as e:

            print("Couldn't detect the language of the text with id {}".format(id[idx]))
            # donot include undetected languages and drop their ids
            id_drop.append(idx)
            continue

        if detected_language == "en":

            if len(text.split()) > allowed_length:
                text = text[
                    :allowed_length
                ]  # if the objective has more than 500 words, only take the first 500 words

            text = preprocess_sent(text)
            token_list = preprocess_word(text)
            if verbose:
                print("\n")
                print(text)
                print("Candidate Tokens: ", token_list)

            sentences_list.append(text)
            token_lists.append(token_list)
        else:
            id_drop.append(idx)

    # drop the employee id of the non-english text
    if verbose:
        print(
            "Objectives of following employees at the following indices are being dropped: ",
            id_drop,
        )

    ids = id.drop([id.index[i] for i in id_drop])
    # docs_titles = docs_titles.drop([docs_titles.index[i] for i in id_drop])
    print("Dropped {} entries with non english text".format(len(id_drop)))

    print("\n Preprocessing raw texts. Done!")

    # saving the preprocessing data in a file
    temp_df = pd.DataFrame()
    temp_df["id"] = ids
    temp_df["sentences"] = sentences_list
    temp_df["token_list"] = token_lists

    if not os.path.exists(dr):
        os.makedirs(dr)
    temp_df.to_pickle(os.path.join(dr, "use_this_data.pkl"))

    return ids, sentences_list, token_lists


# preprocess_documents("/home/anum/Downloads/25828_32915_compressed_Train_rev1.csv/Train_rev1.csv",  dr = "../output/preprocessed_data/job/", subset = 10000 )
