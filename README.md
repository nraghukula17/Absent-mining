# Absent-mining


The provided code snippets are part of a Python script designed to build a Netflix movie recommendation system. Let's break down the code and its functionality:

**1. Importing Libraries:**

- The first code block imports necessary libraries:
    - `networkx as nx`: For creating and manipulating graphs.
    - `matplotlib.pyplot as plt`: For plotting and visualization.
    - `pandas as pd`: For data manipulation and analysis, especially using DataFrames.
    - `numpy as np`: For numerical operations.
    - `math as math`: For mathematical functions.
    - `time`: To measure the execution time of different parts of the code.
- It also sets the style for plots using `seaborn` and defines the default figure size.

**2. Loading the Dataset:**

- The second code block reads a CSV file named "data_file.csv" located in the specified directory and loads it into a pandas DataFrame named `df`. **Note:** The file path `C:/Users/nehar/Downloads/data_file.csv` is specific to the user "nehar" and their local machine. You would need to replace this with the correct path to your data file.

**3. Importing Machine Learning Modules:**

- The third code block imports various modules from the `sklearn` (scikit-learn) library, which are commonly used for machine learning tasks:
    - `CountVectorizer`: To convert a collection of text documents to a matrix of token counts.
    - `TfidfVectorizer`: To convert a collection of raw documents to a matrix of TF-IDF features.
    - `linear_kernel`: To compute the linear kernel between two sets of feature vectors, often used for cosine similarity.
    - `MiniBatchKMeans`: An efficient version of the K-Means algorithm for large datasets.

**4. Initializing TfidfVectorizer:**

- The fourth code block initializes the `TfidfVectorizer` with specific parameters:
    - `max_df=0.4`: Ignores terms that appear in more than 40% of the documents.
    - `min_df=1`: Ignores terms that appear in less than 1 document.
    - `stop_words=('english')`: Removes common English stop words.
    - `lowercase=True`: Converts all text to lowercase.
    - `use_idf=True`: Enables Inverse Document Frequency weighting.
    - `norm=(u'l2')`: Applies L2 normalization to the feature vectors.
    - `smooth_idf=True`: Adds 1 to document frequencies to prevent division by zero.

**5. Defining the `find_similar` Function:**

- The fifth code block defines a function `find_similar` that takes a TF-IDF matrix, an index of a document, and the number of top similar documents to return (`top_n`) as input.
    - It calculates the cosine similarity between the document at the given `index` and all other documents in the `tfidf_matrix` using `linear_kernel`.
    - It flattens the resulting similarity matrix.
    - It gets the indices of the documents sorted in descending order of similarity, excluding the document itself.
    - It returns the indices of the top `top_n` most similar documents.

**6. Importing Networkx:**

- The sixth code block re-imports the `networkx` library, which was already imported in the first block.

**7. Importing Time:**

- The seventh code block re-imports the `time` library, which was also imported earlier.

**8. Building the TF-IDF Matrix and Clustering:**

- The eighth code block performs the following steps:
    - It again imports the necessary modules from `sklearn`.
    - It measures the start time using `time.time()`.
    - It extracts the 'description' column from the DataFrame `df`.
    - It initializes `TfidfVectorizer` with the same parameters as in the fourth block.
    - It fits the `TfidfVectorizer` to the text content and transforms the text into a TF-IDF matrix.
    - It initializes `MiniBatchKMeans` with 50 clusters (`k=50`).
    - It fits the K-Means model to the TF-IDF matrix.
    - It gets the indices of the words in the top 10 most important words for each cluster (though this part is commented out).
    - It transforms the 'description' column using the fitted `TfidfVectorizer`.
    - It predicts the cluster for each description based on the fitted K-Means model and adds a new 'cluster' column to the DataFrame.
    - It prints the value counts of the 'cluster' column (how many descriptions belong to each cluster).

**9. Re-defining the `find_similar` Function:**

- The ninth code block re-defines the `find_similar` function, which is identical to the one defined in the fifth block.

**10. Creating a Graph and Finding Similar Items:**

- The tenth code block is the core of the recommendation system's graph-based approach:
    - It initializes a directed graph `G` using `nx.Graph()`.
    - It iterates through each row of the DataFrame `df`.
    - It adds each movie/TV show description as a node to the graph with the label "Unnamed Woman".
    - For each description, it iterates through the words in the description and adds each word as a node with the label "Woman", creating an edge between the description and the word labeled "in".
    - It then calls the `find_similar` function to get the indices of the top 5 most similar items based on their descriptions (using TF-IDF and cosine similarity).
    - For each similar item found, it adds a new node labeled "SIMILAR" with the name "Sim(first 15 chars of description)".
    - It adds edges between the original description node and the "SIMILAR" node with the label "SIMILARITY".
    - It adds edges between the "SIMILAR" node and the descriptions of the similar items with the label "SIMILARITY".
    - It prints the time taken to build the graph.

**11. Helper Functions for Graph Visualization:**

- The eleventh code block defines two helper functions:
    - `get_all_adj_nodes(list_in)`: Takes a list of node names as input and returns a list of all the nodes in the graph that are directly connected to any of the nodes in the input list, along with the input nodes themselves.
    - `draw_sub_graph(sub_graph)`: Takes a list of node names (a subgraph) as input. It creates a subgraph from the main graph `G` containing only these nodes and their connections. It then draws the subgraph using `nx.draw()`, coloring the nodes based on their labels ("Unnamed Woman" in blue, "Woman" in yellow, "SIMILAR" in red) and displays the plot using `plt.show()`.

**12. Visualizing a Subgraph:**

- The twelfth code block defines a list `list_in` containing two example movie/TV show descriptions.
- It calls `get_all_adj_nodes` to get all the nodes connected to these two descriptions in the graph.
- It then calls `draw_sub_graph` to visualize the subgraph containing these descriptions and their neighbors.

**In summary, the code implements a content-based movie recommendation system using the following techniques:**

- **TF-IDF:** To represent the movie descriptions as numerical vectors, capturing the importance of words in each description.
- **K-Means Clustering:** To group similar movies/TV shows based on their descriptions.
- **Cosine Similarity:** To measure the similarity between the TF-IDF vectors of different descriptions.
- **Graph Representation:** To build a graph where movies/TV shows and the words in their descriptions are nodes, and edges represent the relationship between them. Similarity is also represented by edges to a "SIMILAR" node.
- **Visualization (using Networkx and Matplotlib):** To visualize the relationships between the items in a subgraph based on the descriptions.

**To run this code successfully, you would need to:**

1. **Install the necessary libraries:** `pandas`, `scikit-learn`, `networkx`, `matplotlib`, `numpy`. You can install them using pip:
   ```bash
   pip install pandas scikit-learn networkx matplotlib numpy seaborn
   ```
2. **Have a CSV file named "data_file.csv"** in the specified path (or update the path in the code to point to your data file). This file should contain at least a column with movie/TV show descriptions (named 'description' in the code).
3. **Ensure the data file has a 'uuid' column** if you want to use that as a key for the graph nodes.

The final part of the code will then generate a graph visualization showing the relationships between the two example descriptions provided and the items most similar to them based on their textual descriptions.
