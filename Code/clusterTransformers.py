# DATA PROCCESSING
import pandas as pd # Matrix Operations
from pandas import ExcelWriter
import numpy as np # Linear Algebra
import os # OS Functions
from pathlib import Path # File Pathing

relative_path = Path("..") # Setting relative path that is os agnostic

# GRAPHING
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.patches as mpatches
import seaborn as sns
import seaborn.objects as so
from scipy.ndimage.filters import gaussian_filter
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# MACHINE LEARNING
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, SpectralClustering, OPTICS
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn import manifold


# NATURAL LANGUAGE PROCCESSING (NLP)
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# MISC
import distinctipy # Generate n colors
import re # Regex
import random # Random Operations
import string # String operators
from itertools import chain
import warnings # Control warnings output
warnings.filterwarnings("ignore")
from copy import deepcopy # Making copies of classes
import pickle # Saving and loading models
from typing import * # Type hinting
from tqdm import tqdm # Progress Bar

# LOCATION-BASED
from sklearn.metrics.pairwise import haversine_distances # Distance around Earth's curvature
from math import radians, isnan
import requests # HTTP requests
from requests.adapters import HTTPAdapter, Retry # Retry requests
from bs4 import BeautifulSoup # HTML Parser
import pgeocode # Zipcode to Coordinates
import pycountry_convert as pc # Country to Continent
import time # Time functions
import logging # Logging

class COLOR:
    """Class for displaying colored text in console outputs
    
    Use 'COLOR.<color>' in print statement before text.
    """
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    LIGHT_PURPLE = '\033[35m'
    LIGHT_CYAN = '\033[36m'
    LIGHT_BLUE = '\033[34m'
    LIGHT_GREEN = '\033[32m'
    LIGHT_YELLOW = '\033[33m'
    LIGHT_RED = '\033[31m'
    BOLD = '\033[1m'
    BOLD_OFF = '\033[21m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def preprocess_data(raw_data: str = None, dict_data: str = None, echo: bool = True, reset_seeds: bool = True) -> pd.DataFrame:
    raw_data_path = relative_path / "Data" / raw_data
    raw_data = pd.read_excel(raw_data_path)
    raw_data = clean_data(raw_data)


def make_table(values: pd.Series, dropna: bool = False, dec: int = 3) -> pd.DataFrame:
    """Output a frequency table for a set of values.

    Parameters:
    values (DataFrame Column): values 
    dropna (boolean)(default: False): Drop NAN values before frequency calculation.
    dec (int)(default: 3): Number of decimal places to round 'pct' column to.

    Returns:
    (pd.DataFrame): Frequency Table.
    """
    df = pd.value_counts(values, dropna=dropna).to_frame().reset_index()
    df['pct'] = round(df['count'] / sum(df['count']), dec)
    return(df)

def clean_data(data: pd.DataFrame = None) -> pd.DataFrame:
    """Clean SEM data from rows and columns that are un-needed for analysis.

    Parameters:
    data (DataFrame): Data to be cleaned.

    Returns:
    (DataFrame): Cleaned DataFrame.
    """
    if (data is None): # Ensure paramters are specified
        print("ERROR: Specify DataFrame.")

    # Filtering Data to ensure they accepted the survey and they are eligible
    data = data[(data['s_eligible'] == "1") & (data['s_intro'] == "1")]
    # data = data[(data['s_eligible'] == 1) & (data['s_intro'] == 1)] # NEW DATA
    # Drop columns: 's_eligible', 's_intro' and 4 other columns (That are all NRB Related)
    data = data.drop(columns=['s_eligible', 's_intro', 's_nrb', 's_nrb_residence', 's_nrb_us', 's_nrb_overnight', 's_datayear', 'o_eligible'])
    data = data.replace({'#NULL!':np.nan, None:np.nan}) # Converting #NULL!'s/None's to "NaN" datatype
    data = data.apply(pd.to_numeric, errors='ignore') # Convering column to numeric if possible
    data = data.reset_index(drop=True) # Resetting and droping the index column.
    return(data)


def set_all_seeds(seed: int = 42, echo: bool = True) -> None:
    """Function to set all seeds for reproducibility.

    Parameters:
    seed (int)(default: 42): Integer to set the seeds to.
    echo (bool)(default: True): Boolean to declare if function can output to console.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    if echo:
        print(COLOR.GREEN, "Seeds have been reset to: ", COLOR.BOLD, seed, COLOR.END, sep='')

def get_text_columns(data: pd.DataFrame) -> list:
    """Get list of all string columns from SEM dataset

    Parameters:
    data (DataFrame): Dataframe to check columns

    Returns:
    list: list of all column headers that start with 'o' (i.e text column)
    """
    return([x for x in data if x.startswith('o')]) #Columns that start with 'o'

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def sort_responses(data: pd.DataFrame = None, length: int = 10) -> None:
    # remove nan values
    data = data[~data.isnull()].to_list()
    sorted_responses = sorted(data, key=len, reverse=True)
    sorted_responses = sorted_responses[:length]
    print("Top - Character Length - Token Count - Response")
    for i, response in enumerate(sorted_responses):
        print(i+1, len(response), len(response.split()), response)

class NLP():
    """Class to hold information, data and function for Natural Language Proccessing(NLP) clusters.
    

    """
    def __init__(self, data: pd.Series, echo: bool = True, reset_seeds: bool = True, model_name: string = None, variable: string = None, question: string = None) -> None:
        """ NLP class to hold cluster information/data

        Parameters:
        data (DataFrame Column): data to be analyzed.
        echo (bool)(default: True): Boolean to declare if function can output to console.
        reset_seeds (bool)(default: True): Boolean to declare if set_all_seeds() function is called.

        Returns:
        object to hold information/data on the NLP cluster.

        Available Functions:

        dimension_reduce()
        
        clusterize()

        get_sil_values()

        get_cluster_labels()

        get_cluster_info()

        get_clusters()

        generate_cluster_graph_mpl()

        generate_cluster_graph()
        """
        self.question = question
        self.variable = variable
        self.model_name = model_name
        self.data = data # Storing data into class
        # Booleans to keep track of what operations have been done for helper functions
        self.embedded = False
        self.dim_reduced = False
        self.clustered = False

        if reset_seeds: # Reset seeds for reproducibility
            set_all_seeds(echo=echo)
            
        # Cleaning Data
        responses = pd.DataFrame(np.array(data.copy()), columns=['variable']) # Convert column to array, drop NaN's
        responses = responses[~responses['variable'].isnull()] # Remove null rows
        
        
        # Load model from HuggingFace Hub
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        responses = responses['variable']

        # omit responses in respones that have more than 60 tokens
        token_limit = 100
        old_len = len(responses)
        # filter responses dataframe to only include responses with less than 60 tokens
        responses = responses[responses.apply(lambda x: len(x.split()) <= token_limit)]
        new_len = len(responses)
        print("Omitted", old_len - new_len, "responses with more than", token_limit, "tokens")
        self.indexi = responses.index # Keep Track of index for future analysis
        if echo:
            print(len(responses), 'total samples') # Log samples
        self.docs = responses.values

        # Tokenize sentences
        encoded_input = tokenizer(responses.to_list(), padding=True, truncation=True, return_tensors='pt')
        print("Tokenized")
        # print information about the encoded input
        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)
        print("Computed Embeddings")
        # Perform pooling
        self.sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        print("Performed Pooling")
        # Normalize embeddings
        self.sentence_embeddings = F.normalize(self.sentence_embeddings, p=2, dim=1)
        print("Normalized Embeddings")
        if echo:
            print('Successfully Embedded')
        self.embedded = True

    def dimension_reduce(self, dim_redu_algorithm: str = 'pca', echo: bool = True, reset_seeds: bool = True, perplexity: float = 30, early_exaggeration: float = 12, learning_rate: any = 'auto', n_iter: int = 1000) -> None:
        """Function to reduce n-dimensional vectors to 2-dimensions.

        Parameters:
        dim_redu_algorithm (str)(default: 'pca'): Algorithm to reduce dimensions (see list below).
        echo (bool)(default: True): Boolean to declare if function can output to console.
        reset_seeds (bool)(default: True): Boolean to declare if set_all_seeds() function is called.

        Returns:
        None

        Available dimension reduction algorithms:

        'pca': Principal Component Analysis

        'isomap': Non-Linear - Isometric Mapping

        'tsne': T-distributed Stochastic Neighbor Embedding

        'mds': Multidimensional Scaling

        'lle': Locally Linear Embedding

        'spectral_emb': Spectral Embedding
        """
        if not self.embedded: # Ensure data has been vectorized before reduction
            print(COLOR.RED, 'ERROR: No tokens found.', COLOR.END, sep='')
            return
        if reset_seeds: # Reset Seeds
            set_all_seeds(echo=echo)
        self.dim_redu_algorithm = dim_redu_algorithm # Setting the algorithm
        # Dimensionality Reducution to reduce to 2-dimensions
        if self.dim_redu_algorithm == 'pca': # Principal Component Analysis (PCA)
            self.pca = PCA(n_components=2).fit(self.sentence_embeddings)
            self.datapoints = self.pca.transform(self.sentence_embeddings)
            self.datapoints = pd.DataFrame(self.datapoints, columns=['component1', 'component2'])
        elif self.dim_redu_algorithm == 'isomap': # Non-Linear - Isometric Mapping (ISOMAP)
            self.iso = manifold.Isomap(n_neighbors=3, n_components=2).fit(self.sentence_embeddings)
            self.datapoints = self.iso.transform(self.sentence_embeddings)
            self.datapoints = pd.DataFrame(self.datapoints, columns=['component1', 'component2'])
        elif self.dim_redu_algorithm == 'tsne': # T-distributed Stochastic Neighbor Embedding (TSNE)
            self.tsn = manifold.TSNE(n_components=2, perplexity=perplexity, early_exaggeration=early_exaggeration, learning_rate=learning_rate, n_iter=n_iter)
            self.datapoints = self.tsn.fit_transform(pd.DataFrame(self.sentence_embeddings))
            self.datapoints = pd.DataFrame(self.datapoints, columns=['component1', 'component2'])
        elif self.dim_redu_algorithm == 'mds': # Multidimensional Scaling (MDS)
            self.mds = manifold.MDS(n_components=2, max_iter=10, eps=1)
            self.datapoints = self.mds.fit_transform(pd.DataFrame(self.sentence_embeddings))
            self.datapoints = pd.DataFrame(self.datapoints, columns=['component1', 'component2'])
        elif self.dim_redu_algorithm == 'lle': # Locally Linear Embedding
            self.lle = manifold.LocallyLinearEmbedding(n_neighbors=5, n_components=2)
            self.datapoints = self.lle.fit_transform(pd.DataFrame(self.sentence_embeddings))
            self.datapoints = pd.DataFrame(self.datapoints, columns=['component1', 'component2'])
        elif self.dim_redu_algorithm == 'spectral_emb': # Spectral Embedding
            self.lle = manifold.SpectralEmbedding(n_components=2)
            self.datapoints = self.lle.fit_transform(pd.DataFrame(self.sentence_embeddings))
            self.datapoints = pd.DataFrame(self.datapoints, columns=['component1', 'component2'])
        else:
            print(COLOR.RED, COLOR.BOLD, 'ERROR: ', self.dim_redu_algorithm, 'is not an available algorithm', COLOR.END)
            return
        self.datapoints['sentance'] = self.docs # Add the english sentances to the dataframe
        self.datapoints['orig_index'] = self.indexi # Add the original indexi to the dataframe for future analysis
        self.X = self.datapoints[['component1', 'component2']].to_numpy() # Convert X/Y components to self.X
        self.dim_reduced = True
        if echo:
            print(COLOR.LIGHT_BLUE, 'Reduced dimensions to 2 using ', COLOR.BLUE, COLOR.BOLD, self.dim_redu_algorithm.upper(), COLOR.END, sep='')
    
    def clusterize(self, cluster_algorithm: str = 'kmeans', num_clusters: int = 15, n_neighbors: int = 10, n_init: int = 10, gamma: float = 1.0, degree: float = 3, coef0: float = 1, echo: bool = True, reset_seeds: bool = True) -> None:
        """Apply Cluster Algorithms to group data.

        Parameters:
        cluster_algorithm (str)(default: 'kmeans'): Cluster Algorithm to use (see list below).
        num_clusters (int)(default: 15): Number of clusters to generate (does not apply to DBSCAN and OPTICS)
        echo (bool)(default: True): Boolean to declare if function can output to console.
        reset_seeds (bool)(default: True): Boolean to declare if set_all_seeds() function is called.

        Returns:
        None

        Available Cluster Algorithms to use:

        'kmeans': K-Means

        'kmeans_mb': MiniBatchK-Means

        'dbscan': Density-Based Spatial Clustering of Applications with Noise (DBSCAN)

        'spectral_clustering': Spectral Clustering

        'optics': Ordering Points To Identify the Clustering Structure (OPTICS)

        """
        if not self.dim_reduced: # Ensure data has been reduced to 2-dimensions
            print(COLOR.RED, 'ERROR: Dimensions have not been reduced.', COLOR.END, sep='')
            return
        if reset_seeds: # Reset Seeds
            set_all_seeds(echo=echo)
        self.num_clusters = num_clusters # Setting number of clusters
        self.cluster_algorithm = cluster_algorithm # Setting cluster algorithm
        # KMEANS
        if (self.cluster_algorithm == 'kmeans'):
            self.km = KMeans(n_clusters=self.num_clusters).fit(self.X)
            self.ca = self.km
            self.datapoints['cluster'] = self.km.labels_
        # MINIBATCH_KMEANS
        elif (self.cluster_algorithm == 'kmeans_mb'):
            self.km = MiniBatchKMeans(n_clusters=self.num_clusters, batch_size=500, n_init='auto').fit(self.X)
            self.ca = self.km
            self.datapoints['cluster'] = self.km.labels_
        # DBSCAN
        elif (self.cluster_algorithm == 'dbscan'):
            self.db = DBSCAN(eps=0.01, min_samples=20).fit(self.X)
            self.ca = self.db
            self.num_clusters = len(set(self.db.labels_)) - (1 if -1 in self.db.labels_ else 0) # Number of clusters (not including -1, i.e samples that do not fit in a cluster)
            self.datapoints['cluster'] = self.db.labels_
        # SPECTRAL CLUSTERING
        elif (self.cluster_algorithm == 'spectral_clustering'):
            self.sc = SpectralClustering(n_clusters=self.num_clusters, n_neighbors=n_neighbors, n_init=n_init, gamma=gamma, degree=degree, coef0=coef0).fit(self.X)
            self.ca = self.sc
            self.datapoints['cluster'] = self.sc.labels_
        # OPTICS
        elif (self.cluster_algorithm == 'optics'):
            self.op = OPTICS(min_samples=5).fit(self.X)
            self.ca = self.op
            self.datapoints['cluster'] = self.op.labels_
        else:
            print(COLOR.RED, COLOR.BOLD, 'ERROR: ', self.cluster_algorithm, ' is not an available algorithm', COLOR.END, sep='')
            return
        self.clustered = True
        if echo:
            print(COLOR.LIGHT_BLUE, 'Clustered using ', COLOR.BLUE, COLOR.BOLD, self.cluster_algorithm.upper(), COLOR.END, sep='')
            print(COLOR.YELLOW, "For n_clusters = ", str(self.num_clusters), COLOR.END, sep='')
            print(COLOR.YELLOW, COLOR.BOLD, f"Silhouette coefficient: {silhouette_score(self.X, self.ca.labels_):0.2f}", COLOR.END, sep='')
            try:
                print(COLOR.YELLOW, COLOR.BOLD, f"Inertia: {self.ca.inertia_:0.3f}", COLOR.END, sep='')
                # pass
            except:
                pass
    def get_sil_values(self) -> None:
        """Print Silhouette Values for each cluster to console.

        Parameters:
        None

        Returns:
        None
        """
        if not self.clustered: # Ensure data has been clustered first
            print(COLOR.RED, 'ERROR: No Clusters found.', COLOR.END, sep='')
            return
        sample_silhouette_values = silhouette_samples(self.X, self.ca.labels_)
        print(f"Silhouette values:")
        silhouette_values = []
        for i in range(self.num_clusters):
            cluster_silhouette_values = sample_silhouette_values[self.ca.labels_ == i]
            silhouette_values.append(
                (
                    i,
                    cluster_silhouette_values.shape[0],
                    cluster_silhouette_values.mean(),
                    cluster_silhouette_values.min(),
                    cluster_silhouette_values.max(),
                )
            )
        silhouette_values = sorted(
            silhouette_values, key=lambda tup: tup[2], reverse=True
        )
        for s in silhouette_values:
            print(
                f"    Cluster {s[0]}: Size:{s[1]} | Avg:{s[2]:.2f} | Min:{s[3]:.2f} | Max: {s[4]:.2f}"
            )

    def get_cluster_labels(self) -> list:
        """Get Cluster Labels
        
        Parameters:
        None

        Returns:
        list: List of clusters.
        """
        if not self.clustered: # Ensure data has been clustered first
            print(COLOR.RED, 'ERROR: No Clusters found.', COLOR.END, sep='')
            return
        return(list(np.unique(self.datapoints['cluster'])))

    def get_cluster_info(self, cluster: int = None) -> pd.DataFrame:
        """Get Cluster Information (components 1 and 2, sentence, orig_index, cluster, colorCode, alpha)
        
        Parameters:
        cluster (int): Number of cluster to get information for.

        Returns:
        DataFrame: Dataframe with cluster information
        """
        if not self.clustered:
            print(COLOR.RED, 'ERROR: No Clusters found.', COLOR.END, sep='')
            return
        if cluster is None:
            print(COLOR.RED, 'ERROR: No cluster number given.', COLOR.END, sep='')
            return
        return(self.datapoints[self.datapoints['cluster'] == cluster])

    def get_clusters(self, cluster_filter: list = None) -> pd.DataFrame:
        """Get Dataframe of cluster sentences split by cluster

        Parameters:
        cluster_filter (list)(default: all clusters): list of clusters to retrieve

        Returns:
        DataFrame: Dataframe of clusters(cols) with sentences for each cluster(rows)
        """
        if not self.clustered: # Ensure data has been clustered first
            print(COLOR.RED, 'ERROR: No Clusters found.', COLOR.END, sep='')
            return
        # Filter by Cluster
        if cluster_filter is None:
            cluster_filter = self.get_cluster_labels()

        df = pd.DataFrame()
        for clust in self.get_cluster_labels():
            if clust in cluster_filter:
                df = pd.concat([df, pd.DataFrame(self.get_cluster_info(clust)['sentance'].to_numpy(), columns=['cluster_' + str(clust)])], axis=1)
        return(df)

    def generate_cluster_graph(self, cluster_filter: list = None, figsize: tuple = (10, 10), num_annotations: int = 100, max_char_length: int = 20, hide_labels: bool = False, hide_legend: bool = False, echo = True, reset_seeds: bool = True, jitter: bool = True, jitter_amount: float = 0.01, save=False) -> None:
        """Visualize Clusters using seaborne
        Parameters:
        cluster_filter (list)(default: all clusters): List of clusters to display.
        figsize (tuple)(default: (10, 10)): X and Y size for plot.
        num_annotations (int)(default: 100): Number of random text sentence annotations to display.
        max_char_length (int)(default: 20): Limit of characters for a random sentence to be annotated.
        hide_labels: (bool)(default: False): Boolean to declare whether the annotations are hidden or not.
        hide_legend: (bool)(default: False): Boolean to declare whether the legend is hidden or not.
        echo (bool)(default: True): Boolean to declare if function can output to console.
        reset_seeds (bool)(default: True): Boolean to declare if set_all_seeds() function is called.
        jitter (bool)(default: True): Boolean to declare if the points are jittered for better readability.

        Returns:
        None (Except for Plot Display)
        """
        if not self.clustered: # Ensure data has been clustered first
            print(COLOR.RED, COLOR.BOLD, 'ERROR: No Clusters found.', COLOR.END, sep='')
            return
        if reset_seeds: # Reset Seeds
            set_all_seeds(echo=echo)
        plt.figure(figsize=figsize, dpi=100)

        # Declaring Colors based on Clusters
        color_list = distinctipy.get_colors(self.num_clusters)
        # self.datapoints['colorCode'] = [color_list[x] for x in self.datapoints['cluster']]
        self.datapoints['colorCode'] = [(0,0,0) if x == -1 else color_list[x] for x in self.datapoints['cluster']] # -1 only gets used for DBSCAN models (Makes unused points the Black cluster)
        # self.datapoints['alpha'] = [0.05 if x == -1 else 0.3 for x in self.datapoints['cluster']] # -1 only gets used for DBSCAN models (Makes Black points more transparent)
        chart_data = self.datapoints.copy()
        if jitter:
            chart_data.component1 = chart_data.component1 + np.random.normal(jitter_amount, jitter_amount, chart_data.component1.shape)
            chart_data.component2 = chart_data.component2 + np.random.normal(jitter_amount, jitter_amount, chart_data.component1.shape)
        # Filter by Cluster
        if cluster_filter is not None:
            chart_data = chart_data[chart_data['cluster'].isin(cluster_filter)]
        else:
            cluster_filter = self.get_cluster_labels()
        
        f, ax = plt.subplots(figsize=figsize)
        sns.set_style('white')
        sns.scatterplot(data=chart_data, x=chart_data.component1, y=chart_data.component2, hue='cluster', palette=color_list, alpha=0.3, edgecolor = None)
        # sns.scatterplot(data=chart_data, x=jitter(chart_data.component1, 0.01), y=jitter(chart_data.component2, 0.01), hue='cluster', palette=color_list, alpha=self.datapoints['alpha'], edgecolor = None)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend(title='Cluster')
        plt.title('MODEL_NAME: ' + self.model_name.lower() + '\nDIM_REDU_ALG: ' + self.dim_redu_algorithm.lower() + '\nCLUSTER_ALG: ' + self.cluster_algorithm.lower() + ' ' + str(self.num_clusters) + ' clusters' + '\nVARIABLE: ' + self.variable + '\nQUESTION: ' + self.question + '\nNumber of responses: ' + str(len(self.docs)), loc='left')
        # Display rabdom labels
        if not hide_labels:
            num_annotated = 0
            sent_nums = np.random.choice(range(len(chart_data)), len(chart_data), replace=False)
            for ind in sent_nums:
                # Ensures we only annotate x amount of labels
                if (num_annotated == num_annotations):
                    break
                # Grab Random Point
                # Ensuring Sentance is less than specified length for readability
                if (len(chart_data.iloc[ind]['sentance']) <= max_char_length):
                    plt.text(chart_data.iloc[ind]['component1'], chart_data.iloc[ind]['component2'], chart_data.iloc[ind]['sentance'], size = 7, weight='bold')
                    num_annotated += 1
            print("Annotated", num_annotated, "Sentences")
            if (num_annotated < num_annotations):
                print("WARNING: Only", num_annotated, "Sentences from the data had character lengths under a max_char_length of", max_char_length)
        if hide_legend:
            plt.legend([],[], frameon=False)
        # make legend on the outside of the plot
        plt.legend(bbox_to_anchor=(-0.2, 1), loc=2, borderaxespad=0.)
        plt.show()
        if save:
            filepath = relative_path / 'Writeup Files' / 'temp_plot.png'
            f.savefig(filepath, transparent=True)
            print("Saved plot as temp_plot.png")

    """Output a frequency table for a set of values.

    Parameters:
    values (DataFrame Column): values 
    dropna (boolean)(default: False): Drop NAN values before frequency calculation.
    dec (int)(default: 3): Number of decimal places to round 'pct' column to.

    Returns:
    (pd.DataFrame): Frequency Table.
    """

def build_frequency_table(values: pd.Series, dropna: bool = False, dec: int = 3) -> pd.DataFrame:
    """AI is creating summary for build_frequency_table

    Args:
        values (pd.Series): [description]
        dropna (bool, optional): [description]. Defaults to False.
        dec (int, optional): [description]. Defaults to 3.

    Returns:
        pd.DataFrame: [description]
    """
    df = pd.value_counts(values, dropna=dropna).to_frame().reset_index()
    df['pct'] = round(df['count'] / sum(df['count']), dec)
    return(df)

def translate_response_codes(values: pd.Series = None, data_ref = None) -> pd.Series:
    if (values is None): # Ensure paramters are specified
        print("ERROR: Specify Series.")
    # Return unchanged values if the variable is not in the dictionary.
    if values.name not in data_ref['variable'].to_list():
        return(values)
    ref_table = data_ref[data_ref['variable'] == values.name] # Grab the variable information that matches
    var_refs = ref_table[['value', 'value_label']].set_index('value')['value_label'].to_dict() # Convert to dict
    new_values = values.map(var_refs) # Replace values with dict values
    return(new_values)

def merge_model_data(models: list = None) -> pd.DataFrame:
    """ Merge information regarding multiple models suitable for Power BI import.

    Parameters:
    models (list): list containing NLP models to be merged.

    Returns:
    (pd.DataFrame): Merged data
    """
    merged_data = pd.DataFrame() # Initalizing merged DataFrame
    for model in models: # For each model
        data = model.datapoints.copy() # Copy the datapoint information
        data['model'] = model.model_name # Extract the name and place in a new column
        data['question'] = model.question # Extract the question and place in a new column
        data['variable'] = model.variable # Extract the variable and place in a new column
        data['dim_redu_alg'] = model.dim_redu_algorithm
        data['clust_alg'] = model.cluster_algorithm
        merged_data = pd.concat([merged_data, data]).reset_index(drop=True) # Concat the data to the frame
    return(merged_data)

def update_geodesic_park_data(n: bool = None):
    filepath = relative_path / 'Data' / 'parkInformation.xlsx'
    parkData = pd.read_excel(filepath, 
                  names=['park_id', 'unit_code', 'name', 'designation', 'population_center',
                         'region', 'stats_reporting', 'sem_eligible', 'use_type', 'size',
                         'sem_draw'])
    parkData = add_park_geodesic_info(parkData, n = n)
    save_data(parkData, 'parkInformationGeodesic', parquet=False)

def is_mainland_us(state):
    return 'Show only Mainland US States' if state not in ['Hawaii', 'Alaska'] else None

def code_data(data_clean: pd.DataFrame = None, data_ref: pd.DataFrame = None, save: bool = False, geodesic_park_information: pd.DataFrame = None) -> pd.DataFrame:
    if (data_clean is None) or (data_ref is None): # Ensure paramters are specified
        print("ERROR: Insufficient Information.")
        return
    readable_columns = list(set(data_clean.columns) & set(data_ref['variable'])) # Readable columns are in the intersection with the data dictionary
    coded_data = data_clean.copy() # Drop un-needed column and copy data
    coded_data[readable_columns] = coded_data[readable_columns].apply(translate_response_codes, args=(data_ref,), axis=0).copy() # Convert each column to the readable values from the dictionary
    coded_data = coded_data.join(coded_data['s_npssite'].str.split(' - ', n=1, expand=True).rename(columns={0:'x_unitcode', 1:'x_parkname'})).drop(columns=['x_parkname', 's_npssite']) # Extract the park unit code and park name
    coded_data = coded_data.reset_index().rename(columns={'index': 'orig_index'})
    coded_data['x_unitcode'] = coded_data['x_unitcode'].str.replace(' ', '')
    coded_data = coded_data.replace('THJE', 'JEFM')
    geodesic_user_information = get_geodesic_info(data_clean['n_zip_int'].astype("Int64"))
    geodesic_user_information.columns = ["x_user_" + str(col) for col in geodesic_user_information.columns]
    coded_data = coded_data.join(geodesic_user_information)
    print(COLOR.BOLD, 'Added the users geodesic data.', COLOR.END, sep='')

    coded_data['x_distance_traveled'] = coded_data[['x_user_latitude', 'x_user_longitude', 'x_unitcode']].apply(lambda x: compute_haversine_distance(*x, geodesic_park_information), axis=1)
    coded_data['x_distance_traveled'] = coded_data['x_distance_traveled'].astype("Float64")

    coded_data['x_race'] = coded_data[['m_race_native', 
                                        'm_race_asian', 
                                        'm_race_black', 
                                        'm_race_hawaiian', 
                                        'm_race_white', 
                                        'm_race_other']].rename(columns={'m_race_native': 'Native', 
                                        'm_race_asian': 'Asian', 
                                        'm_race_black': 'Black', 
                                        'm_race_hawaiian': 'Hawaiian', 
                                        'm_race_white': 'White', 
                                        'm_race_other': 'Other'}).replace('Not Selected', 0).replace('Selected', 1).idxmax(axis=1)
    coded_data['n_yearvisit'] = coded_data['n_yearvisit'].replace({np.nan: 1})
    coded_data['x_user_continent'] = coded_data['s_country_int'].apply(lambda x: country_to_continent(x))
    coded_data['x_user_mainland'] = coded_data['x_user_state_name'].apply(is_mainland_us)
    # Change Weights for parks that have a weight of 0. This will allow their specific stats to work, but they do not affect the overall numbers.
    coded_data['weight_peak'] = coded_data['weight_peak'].replace({0: 0.00001}) # Replace 0 weights with 0.00001
    print("Data Coded.")
    if save:
        save_data(coded_data, 'codedData')
    else:
        return(coded_data)

age_groups = [
    (0, 1, 'Up to 1 year old'),
    (2, 4, '2-4'),
    (5, 9, '5-9'),
    (10, 14, '10-14'),
    (15, 17, '15-17'),
    (18, 29, '18-29'),
    (30, 49, '30-49'),
    (50, 64, '50-64'),
    (65, float('inf'), '65+')
]
age_group_positions = [
    ('Up to 1 year old', 0),
    ('2-4', 1),
    ('5-9', 2),
    ('10-14', 3),
    ('15-17', 4),
    ('18-29', 5),
    ('30-49', 6),
    ('50-64', 7),
    ('65+', 9)
]
visitor_age_group_positions = [
    ('18-24', 0),
    ('25-34', 1),
    ('35-44', 2),
    ('45-54', 3),
    ('55-64', 4),
    ('65-74', 5),
    ('75 or older', 6)
]

def add_group_positions(value: string = None, positions: list = None):
    for label, position in positions:
        if value == label:
            return position

def slice_age_groups(age: int = None):
    for lower, upper, label in age_groups:
        if lower <= age <= upper:
            return label

def country_to_continent(country_name):
    if country_name is np.nan:
        return(None)
    if country_name is None:
        return(None)
    if country_name == "Hong Kong (S.A.R.)":
        country_name = "China"
    country_alpha2 = pc.country_name_to_country_alpha2(country_name)
    country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
    country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
    return country_continent_name

def get_geodesic_info(zips: list = None) -> int:
    nomi = pgeocode.Nominatim('us') # Use United States Zip Codes
    zips = [str(zip) for zip in zips]
    # Extract Information
    return(pd.DataFrame(nomi.query_postal_code(zips)).drop(columns=['county_code', 'community_name', 'community_code', 'accuracy']))

def extract_postal_code(string: str = None):
    string = ' '.join(string.split()) # Replace trailing whitespace
    code = re.search(r'\s?[A-Z]{2}\s(\d{5})(-\d{4})?', string) # Search for Zipcode
    if code is None:
        return(np.nan)
    return(code.group(1))

def add_park_geodesic_info(data: pd.DataFrame = None, n: int = None):
    if n is not None:
        geodesic_data = lookup_park_address_info(data['unit_code'][:n].to_list())
    else:
        geodesic_data = lookup_park_address_info(data['unit_code'][:].to_list())
    newData = data.merge(geodesic_data, left_on='unit_code', right_on='unit_code')
    return(newData)

def lookup_park_address_info(unitCodes: list = None) -> pd.DataFrame():
    logging.basicConfig(level=logging.DEBUG)
    session = requests.Session() # Start Session
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[ 502, 503, 504 ]) # Setup retry succesion
    session.mount('http://', HTTPAdapter(max_retries=retries))
    logging.getLogger("urllib3").setLevel(logging.WARNING) # Disable console logs
    manualParks = pd.DataFrame({'unitCode': ['WWIM', 'TUSK', 'YOSE', 'SAND', 'INDE'], 
                               'address': ['1750 Independence Ave SW, Washington, DC 20024',
                                           '4400 Horse Dr, North Las Vegas, NV 89085',
                                           'Po Box 577, Yosemite National Park, CA 95389',
                                           '2995 Lincoln Farm Road, Hodgenville, Kentucky 42748',
                                           '143 South Third Street, Philadelphia, PA 19106']})
    address_book = [] # List to store address information
    pbar = tqdm(unitCodes, unit="Parks", desc="Finding Addresses... ")
    for code in pbar:
        code = code.replace(' ', '') # Remove any whitespace
        pbar.set_description(f'Proccessing %s' % code)
        if code in manualParks['unitCode'].to_list():
            address = manualParks[manualParks['unitCode'] == code]['address'].values[0]
        else:
            if code == "NCPC":
                url = "https://www.nps.gov/" + "NAMA" + "/contacts.htm"
            elif code == "KICA" or code == "SEQU":
                url = "https://www.nps.gov/" + "SEKI" + "/contacts.htm"
            elif code == "OBRI":
                url = "https://www.nps.gov/" + "OBED" + "/contacts.htm"
            elif code == "MABE":
                url = "https://www.nps.gov/" + "MAMC" + "/contacts.htm"
            elif code == "PRPA":
                url = "https://www.nps.gov/" + "WHHO" + "/contacts.htm"
            elif code == "JEFM":
                url = "https://www.nps.gov/" + "THJE" + "/contacts.htm"
            elif code == "NCPE":
                url = "https://www.nps.gov/" + "NACE" + "/contacts.htm"
            elif code == "JOFK":
                url = "https://www.nps.gov/" + "JOFI" + "/contacts.htm"
            elif code == "NACA":
                address_book.append([code, np.nan, np.nan]) # Add Null Address
                continue
            else:
                url = "https://www.nps.gov/" + code + "/contacts.htm"
            headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
            response = session.get(url,headers=headers)
            soup = BeautifulSoup(response.content, 'html') # Extract HTML
            soup = soup.find('div', attrs = {'class':'mailing-address'}) # Find Mailing Address Information
            address = soup.find('p', attrs = {'class':'adr'})
            if address is None:
                address = soup.text
            else:
                address = address.text
        address = ' '.join(address.split()) # Remove unnessesary whitespace
        zipcode = extract_postal_code(address) # Grab the ZipCode
        address_book.append([code, address, zipcode]) # Add to Addressbook
    address_book = pd.DataFrame(address_book, columns=['unit_code', 'address', 'zip_code']) # Convert to DataFrame
    address_book = address_book.join(get_geodesic_info(address_book['zip_code'])).drop(columns=['postal_code']) # Grab Geodesic Information
    print("Addresses Collected.")
    return(address_book)

def compute_haversine_distance(u_lat: float = None, u_lon: float = None, p_unitcode: string = None, geodesic_park_information: pd.DataFrame = None) -> float:
    """ Return dista
    
    Parameters:
    
    Returns:
    (float): Distance in miles between the two coordinates
    """
    if isnan(u_lat) or isnan(u_lon) or (p_unitcode is None):
        return(np.nan)
    geodesic_info = pd.DataFrame([geodesic_park_information[geodesic_park_information['unit_code'] == p_unitcode][['latitude', 'longitude']].values[0], [u_lat, u_lon]])
    geodesic_info = geodesic_info.applymap(radians) # Convert to Radians
    # Calculate and Extract Haversine Distance
    distance = haversine_distances([geodesic_info.iloc[0,:].values, geodesic_info.iloc[1,:].values])[0][1]
    distance = distance * 3958.75587 # Multiply by Earth radius to convert to miles
    return(distance)

def save_model(model: NLP = None, filename: string = None, echo: bool = True) -> None:
    """ Function to save NLP models for future use
    
    Parameters:
    model (.NLP): NLP model to be saved.
    filename (string): Name of the model filename.
    """
    if (filename is None) or (model is None): # Make sure paramters are specified
        print("ERROR: Specify Filename/Model")
    else:
        # Save Model
        filepath = relative_path / 'Models' / (filename + '.pickle') # this filepath has not been tested to work on all systems
        with open(filepath, 'wb') as file_:
            pickle.dump(model, file_, -1)
            if echo:
                print('Model Saved.')

def load_model(filename: string = None, echo: bool = True) -> NLP:
    """ Load in a saved model.pickle file
    
    Paramters: 
    filename (string): name of the model file to be loaded.
    """
    if (filename is None): # Ensure paramters are specified
        print("ERROR: Specify Filename")
    else:
        # Open Model
        filepath = relative_path / 'Models' / (filename + '.pickle') # this filepath has not been tested to work on all systems
        model = pickle.load(open(filepath, 'rb', -1))
        if echo:
            print('Model Loaded.')
        return(model)

def save_data(df: pd.DataFrame = None, filename: string = None, echo: bool = True, parquet: bool = True, excel: bool = True) -> None:
    """ Save a dataframe to .xlsx and .parquet for future use
    
    Parameters:
    df (pd.DataFrame): Dataframe to be saved
    filename (string): Name of the file to be saved
    """
    if (filename is None) or (df is None): # Ensure paramters are specified
        print("ERROR: Specify Filename/DataFrame")
    else:
        # Save Data
        if parquet:
            if echo:
                print("Saving Data(.parquet)...")
            filepath = relative_path / 'Data' / (filename + '.parquet')
            df.to_parquet(filepath)
        if excel:
            if echo:
                print("Saving Data(.xlsx)...")
            filepath = relative_path / 'Data' / (filename + '.xlsx')
            df.to_excel(filepath, index=False)
        if echo:
            print(COLOR.GREEN, COLOR.BOLD, 'SUCCESS: ', filename, ' Data Saved.', COLOR.END, sep='')
def save_pivot_data(df_list: list = None, names: list = None, echo: bool = True):
    if df_list is None or names is None:
        print("ERROR: Specify Names/DataFrame")
    else:
        filepath = relative_path / 'Data' / 'pivot_data.xlsx'
        with ExcelWriter(filepath) as writer:
            for n, df in enumerate(df_list):
                df.to_excel(writer, sheet_name = names[n], index=False)