```python
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
    #for element in rowi['directors']:\
        #G.add_node(element,label="PERSON")
        #G.add_edge(rowi['title'], element, label="DIRECTED")
    #for element in rowi['countries']:\
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
        if G.nodes[e]['label']=="Unnamed Woman":
            colors.append('blue')
        elif G.nodes[e]['label']=="Woman":
            colors.append('yellow')
        elif G.nodes[e]['label']=="SIMILAR":
            colors.append('red')
    nx.draw(subgraph,with_labels=True,font_weight='bold',node_color=colors)
    plt.show()

list_in=['Woman smiles while pushing a wheelbarrow of wood','Army Nurses stand at atttention for inspection']
sub_graph= get_all_adj_nodes(list_in)
draw_sub_graph(sub_graph)
```


