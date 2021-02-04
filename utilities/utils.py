import nltk
from stop_words import get_stop_words
from nltk.stem.wordnet import WordNetLemmatizer
from utilities.visualizations import create_hist
from collections import Counter
import numpy as np


def get_main_topics(topics, top = 50):
    '''
    Given a list of dictionaries where each dictionary has contains the topic in the cluster as key and count of the cluster as value
    this function picks the top words for each cluster and returns it
    :param topics: list of dictionaries


    :return: list of list
    '''
    main_topics = []
    for cluster in topics:
        cluster_topics = []

        # sort a dictionary in descending order
        cluster = {k: v for k, v in sorted(cluster.items(), key=lambda item: item[1])}
        # to keep track of
        counter = 0
        for key, value in cluster.items():
            if counter < top:
                cluster_topics.append(key)
                counter += 1
            else:
                break
        main_topics.append(cluster_topics)
    return main_topics


def clean_topics_from_clusters(topics, delete = False):
    '''
    clean the topic words in each cluster by eliminating the common words
    :param topics: list of dictionary. Each dictionary contains the words as keys and their frequency (score) as values
    :param delete: if True, delete the common words from all clusters
    :return: list of dictionary with cleaned topic words for each cluster
    '''
    K = len(topics)

    # place holder for saving the words to be deleted from each cluster
    topics_to_delete = []
    for _ in range(K):
        topics_to_delete.append([])

    # interate over the topics of all clusters
    for k, topic_k in enumerate(topics):
        for word_k, count_k in topic_k.items():
            #for each word in the topic, check if it exists in the other clusters
            for ind_j, topic_j in enumerate(topics[k+1:K]):
                if word_k in topic_j.keys(): # if the word exists in other clusters too

                    # if the policy is to delete common words from all clusters
                    if delete:
                        topics_to_delete[ind_j + k + 1].append(word_k)
                        topics_to_delete[k].append(word_k)
                    # otherwise keep common words in the cluster with maximum frequency
                    else:
                        if topic_k[word_k] >= topic_j[word_k]: #the word has more frequency in initial cluster
                            topics_to_delete[ind_j+k+1].append(word_k)
                        elif topic_k[word_k] < topic_j[word_k]: # topic has less frequency in initial cluster
                            topics_to_delete[k].append(word_k) # so we delete it from initial cluster


    topics_clean = []
    for i , topic in enumerate(topics):
        for word in topics_to_delete[i]:
            if word in topic.keys():
                del topic[word]
        topics_clean.append(topic)

    return topics_clean

def tokenize_sentence(sent, keep_stopwords = True, stem=False, keep_nouns_only = False):
    stop_words = (list(
        set(get_stop_words('en'))
    ))
    tokenized_sent = nltk.word_tokenize(sent)

    if keep_stopwords == False:
        tokenized_sent = [word for word in tokenized_sent if word not in stop_words]

    if stem == True:
        wnl = WordNetLemmatizer()
        tokenized_sent = [wnl.lemmatize(word) for word in tokenized_sent]

    if keep_nouns_only == True:
        tokenized_sent = [word for (word, pos) in nltk.pos_tag(tokenized_sent) if pos[:2] == 'NN']

    tokenized_sent = [word for word in tokenized_sent if word.isalpha()]
    return tokenized_sent

def get_cluster_text(sentences, labels,ngram):

    # Getting indexes of sentences belonging to each cluster
    sentence_label_dict = dict()
    k  = len(set(labels))
    for label in set(labels):
        sentence_label_dict[label] = [idx for idx, l in enumerate(labels) if l == label]
    # constructing cluster titles
    cluster_titles = []
    topic_words = []

    # first get the clean topic words
    for i in range(k):
        k_sentences = np.array(sentences)[sentence_label_dict[i]]
        ngram_token_with_frequency, _ = generate_n_gram(k_sentences, n=ngram, separator='-',
                                                            keep_stopwords=False, stem=True,
                                                            keep_nouns_only=True)
        topic_words.append(ngram_token_with_frequency)

    # remove same words from clusters

    topic_words = clean_topics_from_clusters(topics=topic_words, delete=False)

    # pick the top ones and create label string out of it

    for i in range(k):
        cluster_token = Counter(topic_words[i])
        cluster_title = ', '.join([key for key, value in cluster_token.most_common(10)])
        cluster_titles.append(cluster_title)

    return cluster_titles

def get_employee_text(sentences, ids):

    np_sentences = np.array(sentences)
    ids = np.array(ids)
    # get the indices of all objectives/documents/sentence for each employee
    employee_objective_indices = dict()
    for employee in ids:
        employee_indices = [i for i, e in enumerate(ids) if (ids[i] == employee)]
        employee_objective_indices[employee] = employee_indices[0]

    employee_titles = dict()

    # constructing hover text (all objectives belonging to that employee)
    for emp in ids:
        emp_index = employee_objective_indices[emp]
        employee_title = np_sentences[emp_index]

        employee_titles[emp] = str(emp) + ': ' + employee_title[0:150]
    return employee_titles


def generate_n_gram(sentences, n=2,separator = '-',  keep_stopwords = True, stem=False, keep_nouns_only = False, top = 30):
    '''
    personal thoughts: A very brain consuming function I had written in a while.

    Given a list of sentences, it generates a list of all n-grams
    :param sentences: a list of sentences to generate n-grams of
    :param n: (int) gram value
    :param top (int): top number of words to be used
    :return: n_gram_per_sentence: (list) a list of list (which corresponds to each sentence) and contains n-gram token
    features of the respective sentence
    '''

    # n for grams cannot be negative
    if n < 1:
        print ("n must be greater than or equal to 1")
        return None

    n_grams_all_sentences = []
    token_list = [] # keep track all features/tokens belonging to each sentences to keep calculate the coherence model
    insufficient_count_for_feature = n-1

    for sent in sentences:

        tokenized_sent = tokenize_sentence(sent, keep_stopwords, stem, keep_nouns_only)
        sent_features = []

        #only if there are enough tokens in the sentence to create at least one feature
        if len(tokenized_sent) >= n:
            for token_number in range(len(tokenized_sent)-insufficient_count_for_feature):

                feature = "" #empty place holder for building n-gram
                for i in range(n):
                    feature += tokenized_sent[token_number + i]
                    # append a "-" only if more tokens need to be added to the feature
                    if i < n-1:
                        feature += separator

                n_grams_all_sentences.append(feature)
                sent_features.append(feature)

        token_list.append(sent_features)
    ngram_token_with_frequency = Counter(n_grams_all_sentences).most_common()
    return dict(ngram_token_with_frequency), token_list

def length_analysis(objectives):
    '''
    given a list of all objective(text -> str), it returns a list with their respective lengths and also plots it ina histogram
    :param objectives: list of all objective
    :return: length (list) list of integers representing the number of words in each objective
    '''
    objectives = list(objectives)
    length = [len(obj.split()) for obj in objectives]
    length.sort()

    print ("Shortest Objective is of length ", min(length))
    print("Largest Objective is of length ", max(length))

    # how many objectives have a length of x
    count_frequency_dict = {x: length.count(x) for x in length}

    # find the maximum allowed length of the objective (in terms of word cound)
    # by finding 'allowed_len' which is the upper cap of the 98% of the objectives lens.
    #since we have to pad the sentences when getting embeddings, some outliers have a very long length which is noise for
    # all the remaining 98% of the objectives
    allowed_percentage = 0.98
    total_length = len(objectives) # count of total objectives
    allowed_count = total_length*allowed_percentage
    allowed_length = 400 # objectives word count would be truncated at this length
    count = 0
    for key, value in count_frequency_dict.items():
        if count >= allowed_count: # if the count has reached the allowed percentage of total length
            allowed_length = key
            break
        count += value


    # use the function from utilities.visualizations
    #create_hist(length,'Number of words per objective', 'Length of Objective', 'Number of Objectives' , save=True )

    return allowed_length

