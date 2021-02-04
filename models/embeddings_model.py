# Word Embeddings

'''
This class has a function called "get_embeddings" in the
end which takes an input file and generates the embeddings
for each sentence and saves it in a file

'''
# 0. Imports

# importing all necessary modules
import pandas as pd
import torch
from transformers import XLNetTokenizer, XLNetModel, BertTokenizer, BertModel, XLMTokenizer, XLMModel, ElectraTokenizer, ElectraModel, AlbertTokenizer, AlbertForMaskedLM
import os
import json
from datetime import datetime
from utilities.document_preprocessing import preprocess_documents
from sentence_transformers import SentenceTransformer
import numpy as np

# needs to be run only once in the beginning

# nltk.download('punkt')

### Function Definations
def preprocess(text_column, model_name=None):
    '''
    Takes in a pandas column containing text and preprocesses it.
    Also creates segment ids if it is required by the model

    - Inputs:
        text_column (pandas series): The column containing all text entries to be processed
        model_name (str): name of the model to be used for generating embeddings

    - Output:
        preprocessed_indexes_column (list): a list of all text tokenized, encoded and padded.

    '''

    # Load pre-trained model tokenizer (vocabulary)
    if model_name == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        pad_id = 0
    elif model_name == 'xlnet':
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        pad_id = tokenizer.pad_token_id
    elif model_name == 'xlm':
        tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-en-2048')
        pad_id = tokenizer.pad_token_id
    elif model_name == 'electra':
        tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
        pad_id = tokenizer.pad_token_id
    elif model_name == 'albert':
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        pad_id = tokenizer.pad_token_id

    else:
        print('Please Specify the appropriate model name.')

    # convert the pandas series to a list for the sake of easier processing
    text_column = list(text_column)

    # define place holders
    preprocessed_indexes_column = []

    max_length = 0
    for text in text_column:  # iterate through each objective

        # tokenize the sentence into words using the model specific tokenizer
        tokenized_text = tokenizer.tokenize(text)
        if len(tokenized_text) > 450:
            tokenized_text = tokenized_text[0:450]

        # Map the token strings to their vocabulary indices.
        indexed_tokens = tokenizer.encode(tokenized_text)
        preprocessed_indexes_column.append(indexed_tokens)

        # keeping track of the maximum sequence length
        max_length = len(indexed_tokens) if len(indexed_tokens) > max_length else max_length
    print("Padding all objectives to be of length ", max_length)
    # pad the sequence to allow batch processing
    preprocessed_indexes_column = pad_sequences(preprocessed_indexes_column, max_length, pad_id)
    return preprocessed_indexes_column

def convert_id_to_token(indexed_tokens,model_name):

    if model_name == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif model_name == 'xlnet':
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    elif model_name == 'xlm':
        tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-en-2048')
    elif model_name == 'electra':
        tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
    elif model_name == 'albert':
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

    word_tokens = [tokenizer.convert_ids_to_tokens(indexed_token) for indexed_token in indexed_tokens]
    return word_tokens

def pad_sequences(data, max_length, pad_id):
    '''
    Pads all the  sequence in 'data' with 'pad_id' to have a constant length of 'max_length'
    - Inputs:
        text_column (pandas series): The column containing all text entries to be processed
        model_name (str): name of the model to be used for generating embeddings

    - Output:
        preprocessed_indexes_column (list): a list of all text tokenized, encoded and padded.

    '''

    data_paded = []
    for d in data:
        if len(d) < max_length:
            temp = d + [pad_id] * (max_length - len(d))
            data_paded.append(temp)
        else:
            data_paded.append(d)
    return data_paded


def generate_embedding(objectives, model_name, batch_size=100,output_attention=False):
    '''
    Takes in a pandas dataframe and generates embeddings for the text column using the hugging face implemented models
    - Inputs:
        pd_dataframe (pandas dataframe): The dataframe containing all text column and their ids
        model_name (str): name of the model to be used for generating embeddings
        batch_size (int): batch size to use when generating embeddings for sentences

    - Output:
        sentence_embedding (tensor): tensor of shape n by 1024 where n is the number of sentence

    '''

    if model_name == 'bert':
        # Load pre-trained bert model (weights)
        model = BertModel.from_pretrained('bert-base-uncased', output_attentions=output_attention)
    elif model_name == 'xlnet':
        # Load pre-trained xlnet model (weights)
        model = XLNetModel.from_pretrained('xlnet-base-cased', output_attentions=output_attention)
    elif model_name == 'xlm':
        # Load pre-trained xlm model (weights)
        model = XLMModel.from_pretrained('xlm-mlm-en-2048', output_attentions=output_attention)
    elif model_name == 'electra':
        # Load pre-trained electra model (weights)
        model = ElectraModel.from_pretrained('google/electra-small-discriminator', output_attentions=output_attention)
    elif model_name == 'albert':
        # Load pre-trained albert model (weights)
        model = AlbertForMaskedLM.from_pretrained('albert-base-v2', output_attentions=output_attention)
    else:
        print("Please select an implemented model name. {} doesn't exist".format(model_name))
        return

    sentences_per_batch = batch_size

    # setting up the device
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    print("using ", device)

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()
    model.to(device)
    num_sentences = len(objectives)
    sentence_embedding = []
    attention_layers = None

    if num_sentences > sentences_per_batch:
        num_batches = num_sentences // sentences_per_batch

        for i in range(num_batches):
            start = i * sentences_per_batch
            end = (i + 1) * sentences_per_batch
            if i == num_batches - 1:
                end = num_sentences
            mini_objective = list(objectives[start:end])

            # Convert inputs to PyTorch tensors
            tokens_tensor = torch.tensor([mini_objective]).squeeze()
            tokens_tensor = tokens_tensor.to(device)

            # Predict hidden states features for each layer
            with torch.no_grad():
                encoded_layers = model(tokens_tensor)

            # taking embeddings of the last layer.
            # token_vecs` is a tensor with shape [n x k x 1024]
            token_vecs = encoded_layers[0]

            # take the vector corresponing to the [CLS] token if it has a cls token.
            if model_name in ['bert', 'albert', 'electra']:
                sentence_embedding += token_vecs[:,0,:].tolist()
            # for those without a cls token, Calculate the average of all k  token vectors and adding to the main list
            else:
                sentence_embedding += torch.mean(token_vecs, dim=1).tolist()
            if output_attention is True:
                attention_layer = [al.tolist() for al in encoded_layers[-1]]
                attention_layer = np.array(attention_layer)
                if len(attention_layers) == 0:
                    attention_layers = attention_layer
                else:
                    attention_layers = np.concatenate((attention_layers, attention_layer), axis = 1)

            print("Embedding for batch {} out of {} batches Completed.".format(i, num_batches))
    else:
        # Convert inputs to PyTorch tensors

        tokens_tensor = torch.tensor([objectives]).squeeze()
        tokens_tensor = tokens_tensor.to(device)

        # Predict hidden states features for each layer
        with torch.no_grad():
            encoded_layers = model(tokens_tensor)

        # taking embeddings of the last layer.
        # token_vecs` is a tensor with shape [n x k x 1024]
        token_vecs = encoded_layers[0]

        # take the vector corresponing to the [CLS] token if it has a cls token.
        if model_name in ['bert', 'albert', 'electra']:
            sentence_embedding = token_vecs[:, 0, :].tolist()
        # for those without a cls token, Calculate the average of all k  token vectors and adding to the main list
        else:
            sentence_embedding = torch.mean(token_vecs, dim=1).tolist()

        if output_attention is True:
            attention_layers = [al.tolist() for al in encoded_layers[-1]]
            attention_layers = np.array(attention_layers)

    print("Our final sentence embedding vector of shape:", len(sentence_embedding), len(sentence_embedding[0]))
    if output_attention:
        print("And the corresponding attention vector of shape:", attention_layers.shape)
    return sentence_embedding, attention_layers


def generate_sentence_embeddings(objectives, model_name, batch_size=100):
    '''
    Takes in a pandas dataframe and generates embeddings for the text column using sentence_transformer
    - Inputs:
        pd_dataframe (pandas dataframe): The dataframe containing all text column and their ids
        model_name (str): name of the model to be used for generating embeddings
        batch_size (int): batch size to use when generating embeddings for sentences

    - Output:
        sentence_embedding (tensor): tensor of shape n by 1024 where n is the number of sentence

    '''

    model2transformer = {'bert-sent': 'bert-base-nli-mean-tokens',
                         'roberta-sent': 'roberta-base-nli-stsb-mean-tokens',
                         # 'roberta-base':'roberta-base-nli-stsb-mean-tokens',
                         'distilbert-sent': 'distilbert-base-nli-stsb-mean-tokens'}

    # get the specfic model name for sentence transformer
    print (model_name)
    model_name = model2transformer[model_name]
    embedder = SentenceTransformer(model_name)
    sentences_per_batch = batch_size

    # setting up the device
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    print(device)

    num_sentences = len(objectives)
    sentence_embedding = []

    if num_sentences > sentences_per_batch:
        num_batches = num_sentences // sentences_per_batch

        for i in range(num_batches):
            start = i * sentences_per_batch
            end = (i + 1) * sentences_per_batch
            if i == num_batches - 1:
                end = num_sentences
            mini_objective = list(objectives[start:end])

            encoded_layers = embedder.encode(mini_objective, convert_to_tensor=True)

            # Calculate the average of all k  token vectors and adding to the main list
            sentence_embedding += encoded_layers.cpu().tolist()
            print("Embedding for batch {} out of {} batches Completed.".format(i, num_batches))
    else:

        encoded_layers = embedder.encode(objectives, convert_to_tensor=True)

        sentence_embedding = encoded_layers.cpu().tolist()

    print("Our final sentence embedding vector of shape:", len(sentence_embedding), len(sentence_embedding[0]))
    return (sentence_embedding)


def get_embeddings(model_name=None, file_name=None, sentences=None, saved_embedding_dir=None, batch_size=None, dr=None,
                   saved_attention_dir = None, output_attention = False):
    '''
    wrapper functions which generates the the embeddings for the text in input file and saves in a json file
    - Inputs:
        file_name (str): string pointing to the file to be read
        model_name (str): name of the model to be used for generating embeddings
        saved_embedding_dir (str): path to json file, if the precomputed embedding needs to be used.
        saved_attention_dir (str): path to json file, if the precomputed attention needs to be used.
        sentences (list of list): sentences, if data has been read previously
        bath_size (int): batch size to use when generating the embeddings
        dr (str): folder path where the embedding should be stored. if not given, the default path is used

    - Output:
        sentence_embedding (tensor): tensor of shape n by 1024 where n is the number of sentence

    '''

    #load presaved embedding
    if saved_embedding_dir is not None:
        with open(saved_embedding_dir) as json_file:
            sentence_embedding = json.load(json_file)
            return sentence_embedding

    # load presaved attention
    if saved_attention_dir is not None:
        with open(saved_attention_dir) as json_file:
            sentence_attention = json.load(json_file)
            return sentence_attention

    if file_name is None and sentences is None:
        print('Either a file name or a list of sentences must be provided')
        return

    pd_objectives = pd.DataFrame()
    # Read file if file path is provided, read the data from there and preprocess it using the pre-defined utility function
    if file_name is not None:
        _, sentences, _ = preprocess_documents(file=file_name, verbose=False,subset=20)

    pd_objectives['objectives'] = sentences

    # generate embeddings

    sentence_transformers_model = [ 'bert-sent','roberta-sent', 'roberta-base-sent', 'distilbert-sent']
    hugging_face_models = ['xlnet', 'electra', 'xlm', 'bert','albert']
    attention = None # used for hugging face models only
    embedding_tokens = [] # used for hugging face models only

    if model_name in sentence_transformers_model:  # use sentence transformer library for it
        if batch_size == None:
            batch_size = 500
        sentence_embedding = generate_sentence_embeddings(pd_objectives['objectives'], model_name, batch_size)

    else:  # for xlnet, use the hugging face library
        # do model specific preprocessing for embedding generation
        pd_objectives['objectives'] = preprocess(pd_objectives['objectives'], model_name)
        if batch_size == None:
            batch_size = 100
        sentence_embedding, attention = generate_embedding(pd_objectives['objectives'], model_name, batch_size,
                                                           output_attention=output_attention)
        embedding_tokens = convert_id_to_token(pd_objectives['objectives'], model_name)

    # save embeddings

    if dr == None:
        output_folder = 'output/embeddings/' + str(
            datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + '_' + model_name + '/'
    else:
        output_folder = dr

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # save the embedding
    file_name = os.path.join(output_folder, model_name + '_embeddings.json')
    with open(file_name, 'w') as fp:
        json.dump(sentence_embedding, fp)

    # save the attention, if it is there
    if attention is not None:
        file_name = os.path.join(output_folder, model_name + '_attention.json')
        with open(file_name, 'w') as fp:
            json.dump(sentence_embedding, fp)

    #todo: also write embedding_tokens somewhere

    return sentence_embedding, attention, embedding_tokens

#get_embeddings(model_name="xlnet", file_name="../data/job_small.xlsx", batch_size = 10,output_attention=False)