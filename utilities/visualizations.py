
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import os
import plotly.graph_objects as go


#todo: move this function inside the topic model class
def get_wordcloud(token_lists=None, topic=None, word_distribution=None, word_count_dict=None, dir = os.getcwd()):
    """
    Get word cloud of each topic from fitted model
    :param model: Topic_Model object
    :param sentences: preprocessed sentences from docs
    """

    print('Getting wordcloud for topic {} ...'.format(topic))

    # LDA + embeddings
    if word_count_dict is not None:
        wordcloud = WordCloud(width=800, height=560,
                              background_color='white', collocations=False,
                              min_font_size=10).generate_from_frequencies(word_count_dict)

    #LDA
    elif word_distribution is not None:
        tokens = str()
        for word, dist in word_distribution.items():

            # from probability to actually numbers the word appears in the document
            word_occurance = int (dist * 100 )
            tokens += ' '.join([''.join(word) for _ in range(word_occurance)])
            tokens += ' '

        wordcloud = WordCloud(width=800, height=560,
                              background_color='white', collocations=False,
                              min_font_size=10).generate(tokens)

    else:
        return

    # plot the WordCloud image
    plt.figure(figsize=(8, 5.6), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    dir = os.path.join(dir,'Topic' + str(topic) + '_wordcloud' )
    plt.savefig(dir)
    print('Getting wordcloud for topic {}. Done!'.format(topic))


def get_employee_clusters(averaged_employee_features, cluster_titles ,employee_titles, vis = 'Employee Cluster 2D', dir = os.getcwd()):


    fig = go.Figure()
    k = len(cluster_titles)
    if vis == 0:
        for i, cl in enumerate(range(k)):
            cluster_df = averaged_employee_features[averaged_employee_features.cluster == cl]

            # plot all the employees belonging to a cluster using a new trace.
            features = np.stack(cluster_df['feature_vector_2d'].to_numpy())

            # normalize the features between 0 and 1
            #features = preprocessing.normalize(features)
            min_ = features.min(axis=0)
            max_ = features.max(axis=0)
            features = (features - min_) / (max_ - min_)
            # constructing hover text for all employees belonging to that cluster
            employee_titles_this_cluster = [employee_titles[emp] for emp in cluster_df.index.values]

            fig.add_trace(go.Scattergl(
                x=features[:, 0],
                y=features[:, 1],
                mode='markers',
                showlegend=True,
                text=employee_titles_this_cluster,
                marker=dict(size=8, color=i),
                name="cluster_" + str(cl + 1) + ": " + cluster_titles[cl]
            ))

        fig.update_layout(title='Distribution of employees over clusters in 2d space', legend=dict(
            x=0.6,
            y=1.4,
            traceorder='normal',
            font=dict(
                size=10)))
        path = os.path.join(dir , '2D_clusters.html')

    else:
        #vis is 3d (Employee Cluster 3D)
        fig = go.Figure()

        for i, cl in enumerate(range(k)):
            cluster_df = averaged_employee_features[averaged_employee_features.cluster == cl]

            # plot all the employees belonging to a cluster using a new trace.
            features = np.stack(cluster_df['feature_vector_3d'].to_numpy())

            # normalize the features between 0 and 1
            #features = preprocessing.normalize(features)
            min_ = features.min(axis=0)
            max_ = features.max(axis=0)
            features = (features - min_) / (max_ - min_)

            # constructing hover text for all employees belonging to that cluster
            employee_titles_this_cluster = [employee_titles[emp] for emp in cluster_df.index.values]


            fig.add_trace(go.Scatter3d(
                x=features[:, 0],
                y=features[:, 1],
                z=features[:, 2],
                mode='markers',
                showlegend=True,
                text=employee_titles_this_cluster,
                marker=dict(size=3, color=i),
                name="cluster_" + str(cl + 1) + ": " + cluster_titles[cl]
            ))

        fig.update_layout(title='Distribution of employees over clusters in 3d Space', legend=dict(
            x=0.6,
            y=1.4,
            traceorder='normal',
            font=dict(
                size=10)))
        path = os.path.join(dir, '2D_clusters.html')

    #fig.show()
    fig.write_html(path)
    print ("Saved Clusters to " , path)
    #ok = fig.to_html(full_html=False)

def create_hist(x, title=None, x_title = None, y_title = None, save= False, dir = None):


    fig = go.Figure(data=[go.Histogram(x=x)])
    fig.update_layout(
        title_text=title,  # title of plot
        xaxis_title_text=x_title,  # xaxis label
        yaxis_title_text=y_title,  # yaxis label
        bargap=0.2,  # gap between bars of adjacent location coordinates
        bargroupgap=0.1  # gap between bars of the same location coordinates

    )
    fig.show()
    if save == True:

        if dir:
            hist_path = dir + title + '.jpeg'
        else:
            hist_path = title + '.jpeg'

        fig.write_image(hist_path)


def vis_3d(vec_3d,lbs):
    fig = go.Figure()

    for i, cl in enumerate(set(lbs)):
        cluster_samples = np.argwhere(lbs==cl)
        features = vec_3d[cluster_samples]
        features = features.squeeze()

        # normalize the features between 0 and 1

        min_ = features.min(axis=0)
        max_ = features.max(axis=0)

        features = (features - min_) / (max_ - min_)

        fig.add_trace(go.Scatter3d(
            x=features[:, 0],
            y=features[:, 1],
            z=features[:, 2],
            mode='markers',
            showlegend=True,
            marker=dict(size=3, color=i),
            name="cluster_" + str(cl + 1)
        ))

    fig.update_layout(title='Distribution of jobs over clusters in 3d Space', legend=dict(
            x=0.6,
            y=1.4,
            traceorder='normal',
            font=dict(
                size=10)))
    fig.show()

def vis_2d(vec_2d,lbs):
    fig = go.Figure()

    for i, cl in enumerate(set(lbs)):
        cluster_samples = np.argwhere(lbs==cl)
        features = vec_2d[cluster_samples]
        features = features.squeeze()

        # normalize the features between 0 and 1

        min_ = features.min(axis=0)
        max_ = features.max(axis=0)

        features = (features - min_) / (max_ - min_)

        fig.add_trace(go.Scattergl(
            x=features[:, 0],
            y=features[:, 1],
            mode='markers',
            showlegend=True,
            marker=dict(size=3, color=i),
            name="cluster_" + str(cl + 1)
        ))

    fig.update_layout(title='Distribution of jobs over clusters in 2d Space', legend=dict(
            x=0.6,
            y=1.4,
            traceorder='normal',
            font=dict(
                size=10)))
    fig.show()



