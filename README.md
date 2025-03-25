# import librairies
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math as math
import time 
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = [14,14]

df=pd.read_csv("C:/Users/nehar/Downloads/data_file.csv")

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import MiniBatchKMeans

vector=TfidfVectorizer(max_df=0.4,
                       min_df=1,
                       stop_words=('english'),
                       lowercase=True,
                       use_idf=True,
                       norm=(u'12'),
                       smooth_idf=True)

def find_similar(tfidf_matrix,index,top_n=5):
    cosine_similarities=linear_kernel(tfidf_matrix[index:index+1],tfidf_matrix).flatten()
    related_docs_indices=[i for i in cosine_similarities.argsort()[::-1] if i != index]
    return [index for index in related_docs_indices][0:top_n]

import networkx as nx

import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import MiniBatchKMeans

# Build the tfidf matrix with the descriptions
start_time = time.time()
text_content = df['description']
vector = TfidfVectorizer(max_df=0.4,         # drop words that occur in more than X percent of documents
                             min_df=1,      # only use words that appear at least X times
                             stop_words='english', # remove stop words
                             lowercase=True, # Convert everything to lower case 
                             use_idf=True,   # Use idf
                             norm=u'l2',     # Normalization
                             smooth_idf=True # Prevents divide-by-zero errors
                            )
tfidf = vector.fit_transform(text_content)

# Clustering  Kmeans
k = 50
kmeans = MiniBatchKMeans(n_clusters = k)
kmeans.fit(tfidf)
centers = kmeans.cluster_centers_.argsort()[:,::-1]
terms = vector.get_feature_names()

# print the centers of the clusters
# for i in range(0,k):
#     word_list=[]
#     print("cluster%d:"% i)
#     for j in centers[i,:10]:
#         word_list.append(terms[j])
#     print(word_list) 
    
request_transform = vector.transform(df['description'])
# new column cluster based on the description
df['cluster'] = kmeans.predict(request_transform) 

df['cluster'].value_counts().head()

def find_similar(tfidf_matrix, index, top_n = 5):
    cosine_similarities = linear_kernel(tfidf_matrix[index:index+1], tfidf_matrix).flatten()
    related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != index]
    return [index for index in related_docs_indices][0:top_n]

G = nx.Graph(label="Unnamed Women")
start_time = time.time()
for i, rowi in df.iterrows():
    if (i%1000==0):
        print(" iter {} -- {} seconds --".format(i,time.time() - start_time))
    G.add_node(rowi['description'],key=rowi['uuid'],label="Unnamed Woman")
    for element in rowi['description']:
        G.add_node(element,label="Woman")
        G.add_edge(rowi['description'], element, label="in")
   # for element in rowi['categ']:
#G.add_node(element,label="CAT")
        #G.add_edge(rowi['title'], element, label="CAT_IN")
    #for element in rowi['directors']:
        #G.add_node(element,label="PERSON")
        #G.add_edge(rowi['title'], element, label="DIRECTED")
    #for element in rowi['countries']:
        #G.add_node(element,label="COU")
        #G.add_edge(rowi['title'], element, label="COU_IN")
    
    indices = find_similar(tfidf, i, top_n = 5)
    snode="Sim("+rowi['description'][:15].strip()+")"        
    G.add_node(snode,label="SIMILAR")
    G.add_edge(rowi['description'], snode, label="SIMILARITY")
    for element in indices:
        G.add_edge(snode, df['description'].loc[element], label="SIMILARITY")
print(" finish -- {} seconds --".format(time.time() - start_time))   

def get_all_adj_nodes(list_in):
    sub_graph=set()
    for m  in list_in:
        sub_graph.add(m)
        for e in G.neighbors(m):
            sub_graph.add(e)
    return list(sub_graph)
def draw_sub_graph(sub_graph):
    subgraph = G.subgraph(sub_graph)
    colors=[]
    for e in subgraph.nodes():
        if G.nodes[e]['label']=="Unnamed Women":
            colors.append('blue')
        elif G.nodes[e]['label']=="description":
            colors.append('yellow')
        elif G.nodes[e]['label']=="location":
            colors.append('red')
    nx.draw(subgraph,with_labels=True,font_weight='bold',node_color=colors)
    plt.show()

list_in=['Woman smiles while pushing a wheelbarrow of wood','Army Nurses stand at atttention for inspection']
sub_graph= get_all_adj_nodes(list_in)
draw_sub_graph(sub_graph)

def find_similar(tfidf_matrix, index, top_n = 5):
    cosine_similarities = linear_kernel(tfidf_matrix[index:index+1], tfidf_matrix).flatten()
    related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != index]
    return [index for index in related_docs_indices][0:top_n]

def concat_similar(tfidf_matrix, df, index, top_n = 5):
    indices = find_similar(tfidf_matrix, index, top_n)
    similar_df = df.iloc[indices][['uuid', 'description']]
    similar_descriptions = similar_df['description'].tolist()
    return ' | '.join(similar_descriptions)


index = 0
similar_descriptions = concat_similar(tfidf, df, index, top_n = 5)
print(similar_descriptions)


index = 0
similar_descriptions = concat_similar(tfidf, df, index, top_n=5)
concatenated_output = ''.join(similar_descriptions)
print(concatenated_output)


concatenated_output = '\\n'.join(similar_descriptions)
print(concatenated_output)

import markovify

# Define the text corpus using the concatenated string
text_model = markovify.Text(concatenated_output)

# Generate a new sentence using the Markov chain model
new_sentence = text_model.make_sentence()

# Print the new sentence
print(new_sentence)


import pandas as pd
from nltk.tokenize import word_tokenize
import random

# Load the Excel file
df = pd.read_csv("C:/Users/nehar/Downloads/data_file.csv")

# Define the function to check for similarity
def check_similarity(description, df):
    for index, row in df.iterrows():
        similarity = row['description'].lower() == description.lower()
        if similarity:
            return row['description']
    return None

# Concatenate the sentences
sentences = df['description'].tolist()
concatenated_string = ' '.join(sentences)

# Tokenize the words
words = word_tokenize(concatenated_string)

# Shuffle the words randomly
random.shuffle(words)

# Construct a new sentence
new_sentence = ' '.join(words)

# Check for similarity in the Excel file
category = check_similarity(new_sentence, df)

if category:
    print(f"The new sentence '{new_sentence}' belongs to the category '{category}'.")
else:
    print(f"No similar sentence found in the Excel file.")


import nltk
from nltk.tokenize import word_tokenize
import random

# Sample concatenated string
concatenated_string = "Industrial workers building SS George Washington Carver,Women await Rosa Lee's parole hearing in Georgia,Congregation of Store Front Baptist Church, Chicago,Women working as insurance clerks,Women in the Repair Depot repair military clothing"

# Tokenize the words
words = word_tokenize(concatenated_string)

# Shuffle the words randomly
random.shuffle(words)

# Combine the shuffled words into a new sentence
new_sentence = ' '.join(words)

# Print the new sentence
print(new_sentence)


import random

# define a list of words
words = ['working',' Church', 'Congregation', 'military', 'insurance' , 'Washington', 'as' , 'Women', 'hearing','of','Carver', 'Rosa', 'await', 'Depot' , 'Lee', 'repair','Store', 'workers', 'George', 'Industrial' , 'Women','Georgia', 'Front','parole', 'in', 'Women' , 'SS', 'the', 'clothing', 'clerks', 'in', 'building','Baptist', 'Repair', 'Chicago']

# shuffle the words randomly
random.shuffle(words)

# join the shuffled words into a sentence
sentence = ' '.join(words)

# print the sentence
print(sentence)


import random

# Concatenated words
concatenated_words = "Industrial workers building SS George Washington Carver | Women await Rosa Lee's parole hearing in Georgia | Congregation of Store Front Baptist Church, Chicago | Women working as insurance clerks | Women in the Repair Depot repair military clothing"

# Split the words by the separator |
words = concatenated_words.split(' | ')

# Shuffle the words randomly
random.shuffle(words)

# Combine the shuffled words into a new title
new_title = ' '.join(words)

# Print the new title
print(new_title)


import pandas as pd
import random

# read the Excel file containing the sentences
df = pd.read_csv('C:/Users/nehar/Downloads/data_file.csv')

# choose a random sentence from the Excel file
row = random.choice(df.index)
sentence = df.loc[row, 'description']

# tokenize the words
words = sentence.split()

# shuffle the words randomly
random.shuffle(words)

# combine the shuffled words into a new sentence
new_sentence = ' '.join(words)

# print the new sentence
print(new_sentence)


import pandas as pd

# Read the Excel file containing the sentences
df = pd.read_csv('C:/Users/nehar/Downloads/data_file.csv')

# Get the concatenated string
concatenated_string = "Industrial workers building SS George Washington Carver,Women await Rosa Lee's parole hearing in Georgia,Congregation of Store Front Baptist Church, Chicago,Women working as insurance clerks,Women in the Repair Depot repair military clothing"

# Split the string into words
words = concatenated_string.split()

# Create a set of unique words
unique_words = set(words)

# Loop through each sentence in the Excel file
for index, row in df.iterrows():
    # Get the sentence from the current row
    sentence = row['description']
    
    # Split the sentence into words
    sentence_words = sentence.split()
    
    # Calculate the number of words in common between the sentence and the unique words set
    common_words = len(set(sentence_words).intersection(unique_words))
    
    # If there are at least 2 common words, print the sentence
    if common_words >= 2:
        print(sentence)


import networkx as nx
import matplotlib.pyplot as plt

# Define the list of sentences
sentences = [
    'Women sitting in front of an empty store in Harlem',
    'Woman dressed in men\'s clothing',
    'Portrait of women employed by the Navy',
    'Woman reading bible in Store Front Baptist church, Chicago',
    'Industrial worker building SS George Washington Carver',
    'Woman working as a typesetter in Atlanta',
    'Industrial workers building SS George Washington Carver',
    'Women of the National Youth Adminstration meet in Chicago',
    'Industrial workers building SS George Washington Carver',
    'Women working machinery in apron factory',
    'Woman working as nanny feeds child',
    'Women and girls working in a milliner factory',
    'Women and girls working in lampshade factory',
    'Women working in a garment factory during WWI',
    'Women in the Repair Depot repair military clothing',
    'Women working as insurance clerks',
    'Congregation of Store Front Baptist Church, Chicago',
    'Women await Rosa Lee\'s parole hearing in Georgia',
    'Portrait of woman in black lace dress',
    'Portrait of woman in floral costume',
    'Portrait of a girl with long hair in a white blouse',
    'Portrait of a woman in a striped dress',
    'Portrait of woman in white blouse with flowers in her hair',
    'Portrait of a woman in a lace dress with large beaded necklace',
    'Portrait of a woman in a dotted dress with large ribbon in her hair',
    'Portrait of a woman in a lace trimmed blouse',
    'Portrait of a woman born in 1791 wearing a shawl',
    'Portrait of student of Industrial School in South Carolina',
    'Congregation of Abyssinian Baptist Church, Harlem',
    'Congregation of Abyssinian Baptist Church, Harlem',
    'Portrait of dressing making students in South Carolina',
    'Portrait of a woman in a cotton field',
    'Portrait of a woman in overgrown field',
    'Portrait of former slave in Alabama',
    'Congregation sitting in pews in Heard County, George',
    'Portrait of a woman in a buttoned bodice',
    'Portrait of a woman in velvet',
    'Portrait of a woman in a spinning factory',
    'Young girl poses in front of spinning machine',
    '14 year old girl working in paper box factory',
    'Girls and instructor in ballet class at the NYC School of Ballet',
    'Women dressed as gypsies pose for production of Chauve-Souris',
    'Portrait of a seated older woman in a striped dress',
    'Young women working at a table in artificial flower factory'
]


# Create an empty graph
G = nx.Graph()

# Add nodes to the graph
for sentence in sentences:
    G.add_node(sentence)

# Add edges to the graph
for i, node1 in enumerate(G.nodes()):
    for j, node2 in enumerate(G.nodes()):
        if i < j:
            sentence1_words = node1.split()
            sentence2_words = node2.split()
            common_words = set(sentence1_words).intersection(set(sentence2_words))
            if len(common_words) >= 2:
                G.add_edge(node1, node2)

# Draw the graph
k = 3.0

# Draw the graph
pos = nx.spring_layout(G, k=k)
nx.draw(G, pos, node_size=1000, with_labels=True, font_size=10)
nx.draw_networkx_edge_labels(G, pos)
plt.show()

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Read the Excel file containing the sentences
df = pd.read_csv('C:/Users/nehar/Downloads/data_file.csv')

# Get the concatenated string
concatenated_string = "Industrial workers building SS George Washington Carver,Women await Rosa Lee's parole hearing in Georgia,Congregation of Store Front Baptist Church, Chicago,Women working as insurance clerks,Women in the Repair Depot repair military clothing"

# Split the string into words
words = concatenated_string.split()

# Create a set of unique words
unique_words = set(words)

# Create an empty graph
G = nx.Graph()

# Add nodes to the graph
for index, row in df.iterrows():
    sentence = row['description']
    if sentence not in G.nodes:
        G.add_node(sentence)

# Add edges to the graph
for i, node1 in enumerate(G.nodes()):
    for j, node2 in enumerate(G.nodes()):
        if i < j:
            sentence1_words = node1.split()
            sentence2_words = node2.split()
            common_words = set(sentence1_words).intersection(set(sentence2_words))
            if len(common_words) >= 2:
                G.add_edge(node1, node2)

# Draw the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, node_size=1000, with_labels=True, font_size=10)
nx.draw_networkx_edge_labels(G, pos)
plt.show()


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Read the Excel file containing the sentences
df = pd.read_csv('C:/Users/nehar/Downloads/data_file.csv')

# Get the concatenated string
concatenated_string = "Industrial workers building SS George Washington Carver,Women await Rosa Lee's parole hearing in Georgia,Congregation of Store Front Baptist Church, Chicago,Women working as insurance clerks,Women in the Repair Depot repair military clothing"

# Split the string into words
words = concatenated_string.split()

# Create a set of unique words
unique_words = set(words)

# Create an empty graph
G = nx.Graph()

# Add nodes to the graph
counter = 0
for index, row in df.iterrows():
    if counter < 30:
        sentence = row['description']
        if sentence not in G.nodes:
            G.add_node(sentence)
        counter += 1
    else:
        break

# Add edges to the graph
for i, node1 in enumerate(G.nodes()):
    for j, node2 in enumerate(G.nodes()):
        if i < j:
            sentence1_words = node1.split()
            sentence2_words = node2.split()
            common_words = set(sentence1_words).intersection(set(sentence2_words))
            if len(common_words) >= 2:
                G.add_edge(node1, node2)

# Draw the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, node_size=1000, with_labels=True, font_size=12)
nx.draw_networkx_edge_labels(G, pos)
plt.show()


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Read the Excel file containing the sentences
df = pd.read_csv('C:/Users/nehar/Downloads/data_file.csv')

# Get the concatenated string
concatenated_string = "Industrial workers building SS George Washington Carver,Women await Rosa Lee's parole hearing in Georgia,Congregation of Store Front Baptist Church, Chicago,Women working as insurance clerks,Women in the Repair Depot repair military clothing"

# Split the string into words
words = concatenated_string.split()

# Create a set of unique words
unique_words = set(words)

# Initialize the graph
G = nx.Graph()

# Loop through each sentence in the Excel file
for index, row in df.iterrows():
    # Get the sentence from the current row
    sentence = row['description']
    
    # Split the sentence into words
    sentence_words = sentence.split()
    
    # Calculate the number of words in common between the sentence and the unique words set
    common_words = list(set(sentence_words).intersection(unique_words))
    
    # Add nodes and edges to the graph
    for i in range(len(common_words)):
        for j in range(i+1, len(common_words)):
            G.add_edge(common_words[i], common_words[j], description=sentence)
            
# Set node positions using the spring layout algorithm
pos = nx.spring_layout(G)

# Draw nodes and edges
nx.draw_networkx_nodes(G, pos)
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_edge_labels(G, pos, edge_labels={(u,v):d['description'] for u,v,d in G.edges(data=True)})

# Set axis labels and show the plot
plt.xlabel('Words')
plt.ylabel('Sentences')
plt.show()


import pandas as pd
import plotly.graph_objs as go

# Read the Excel file containing the sentences
df = pd.read_csv('C:/Users/nehar/Downloads/data_file.csv')

# Get the concatenated string
concatenated_string = "Industrial workers building SS George Washington Carver,Women await Rosa Lee's parole hearing in Georgia,Congregation of Store Front Baptist Church, Chicago,Women working as insurance clerks,Women in the Repair Depot repair military clothing"

# Split the string into words
words = concatenated_string.split()

# Create a set of unique words
unique_words = set(words)

# Count the frequency of each word in the output
word_freq = {}
for word in words:
    if word in word_freq:
        word_freq[word] += 1
    else:
        word_freq[word] = 1

# Create a list of word-frequency pairs sorted by frequency in descending order
sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

# Create a bar graph showing the frequency of each word
data = [go.Bar(
            x=[pair[0] for pair in sorted_word_freq],
            y=[pair[1] for pair in sorted_word_freq]
    )]

# Set the layout of the graph
layout = go.Layout(
    title='Frequency of Words in Output',
    xaxis=dict(title='Words'),
    yaxis=dict(title='Frequency')
)

# Create the figure object and display the graph
fig = go.Figure(data=data, layout=layout)
fig.show()


pip install plotly

import pandas as pd
import difflib

# read the Excel file containing the descriptions
df = pd.read_csv('C:/Users/nehar/Downloads/data_file.csv')

# define the given description
given_desc = "in Lincoln lounge students Nursing"

# compare the given description with descriptions in the Excel file
similar_descs = difflib.get_close_matches(given_desc, df['description'], n=3, cutoff=0.6)

# print the similar descriptions
print("Similar descriptions:")
for desc in similar_descs:
    print("- " + desc)

# get recommendations based on the similar descriptions
recommendations = []
for desc in similar_descs:
    row = df[df['description'] == desc].iloc[0]
    recommendations += row['Recommendations'].split(";")

# remove duplicates from recommendations
recommendations = list(set(recommendations))

# print the recommendations
print("\nRecommendations:")
for rec in recommendations:
    print("- " + rec)


import nltk
from nltk.tokenize import word_tokenize
import random

#Sample concatenated string
concatenated_string = "Industrial workers building SS George Washington Carver,Women await Rosa Lee's parole hearing in Georgia,Congregation of Store Front Baptist Church, Chicago,Women working as insurance clerks,Women in the Repair Depot repair military clothing"

#Tokenize the words
words = word_tokenize(concatenated_string)

#Shuffle the words randomly
random.shuffle(words)

#Combine the shuffled words into a new sentence
new_sentence = ' '.join(words)

#Use NLTK's Part-of-Speech Tagger to get the part-of-speech tags for each word
pos_tags = nltk.pos_tag(words)

#Initialize an empty list to store the noun phrases
noun_phrases = []

#Iterate over the part-of-speech tags and extract the noun phrases
for i in range(len(pos_tags)):
    if pos_tags[i][1].startswith('NN'): # check if the tag starts with 'NN' (i.e. a noun)
        phrase = pos_tags[i][0] # add the current noun to the phrase
        j = i + 1 # move to the next word
        while j < len(pos_tags) and pos_tags[j][1].startswith('NN'): # check if the next word is also a noun
            phrase += ' ' + pos_tags[j][0] # add the next noun to the phrase
            j += 1 # move to the next word
noun_phrases.append(phrase) # add the completed phrase to the list

#Combine the noun phrases into a new description
new_description = ', '.join(noun_phrases)

#Print the new sentence and the new description
print('New sentence:', new_sentence)
print('New description:', new_description)
