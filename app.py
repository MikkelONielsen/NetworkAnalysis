# **Network Analysis Application

import community as community_louvain

# Packaging
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
import streamlit as st

sns.set()

# Import the libraries and link to the bokeh backend
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')
from bokeh.plotting import show
from holoviews.operation.datashader import datashade, bundle_graph

# Function to load the dataset
@st.cache  # Cache the function to enhance performance
def load_data():
    # Define the file path
    file_path = 'https://raw.githubusercontent.com/gsprint23/CongressionalTwitterNetwork/master/congress_network_data.json'
    
    # Load the json file into a pandas dataframe
    df = pd.read_json(file_path)
    # Extract the necessary data from the dataframe
    inList = df['inList'][0]
    inWeight = df['inWeight'][0]
    outList = df['outList'][0]
    outWeight = df['outWeight'][0]
    usernameList = df['usernameList'][0]

    # Creating empty lists to hold the source, target, and weight values
    sources = []
    targets = []
    weights = []

    # Populating the lists with values from inList and outList
    for i, (in_edges, out_edges) in enumerate(zip(inList, outList)):
        source = usernameList[i]

        # Handling inbound connections
        for target_index, weight in zip(in_edges, inWeight[i]):
            target = usernameList[target_index]
            sources.append(target)
            targets.append(source)
            weights.append(weight)

        # Handling outbound connections
        for target_index, weight in zip(out_edges, outWeight[i]):
            target = usernameList[target_index]
            sources.append(source)
            targets.append(target)
            weights.append(weight)

    # Creating a DataFrame from the populated lists
    network_df = pd.DataFrame({
        'source': sources,
        'target': targets,
        'weight': weights
    })

    return network_df

df = load_data()
# Set the app title and sidebar header
st.title("Network analysis for Twitter Interaction Network for the US Congress")

    # Display dataset overview
st.header("Dataset Overview")
st.dataframe(df.head(10))

# Determine the weight threshold for the top 10%
top_10_percent_weight_threshold = df['weight'].quantile(0.9)

# Filter the edges above this new weight threshold
edges_top_10_percent_weight = df[df['weight'] > top_10_percent_weight_threshold]

# Now select the top 10% nodes based on degree
# Calculate the degree for each node
degree = edges_top_10_percent_weight['source']._append(edges_top_10_percent_weight['target']).value_counts()

# Determine the cutoff for the top 10% of nodes by degree
top_10_percent_nodes_cutoff = degree.quantile(0.9)

# Select the top 10% nodes
top_nodes_by_degree = degree[degree > top_10_percent_nodes_cutoff].index.tolist()

# Filter the DataFrame to include only edges that have both source and target in the top 10% nodes
filtered_network_df = edges_top_10_percent_weight[
    edges_top_10_percent_weight['source'].isin(top_nodes_by_degree) &
    edges_top_10_percent_weight['target'].isin(top_nodes_by_degree)
]
# Create a graph from the filtered network dataframe
G = nx.from_pandas_edgelist(filtered_network_df, 'source', 'target', ['weight'])

assortativity_coefficient = nx.degree_assortativity_coefficient(G)

# Create a graph from the entire network dataframe
G_full = nx.from_pandas_edgelist(df, 'source', 'target', ['weight'])

# Calculate Degree Centrality
degree_centrality = nx.degree_centrality(G_full)

# Calculate Betweenness Centrality
betweenness_centrality = nx.betweenness_centrality(G_full)

# Calculate Eigenvector Centrality
eigenvector_centrality = nx.eigenvector_centrality_numpy(G_full)

# Combine all centrality measures into a single DataFrame for analysis
centrality_measures_df = pd.DataFrame({
    'Degree Centrality': degree_centrality,
    'Betweenness Centrality': betweenness_centrality,
    'Eigenvector Centrality': eigenvector_centrality
})

centrality_measures_df.sort_values(by=['Degree Centrality'], ascending=False).head()

centrality_measures_df.sort_values(by=['Betweenness Centrality'], ascending=False).head()

centrality_measures_df.sort_values(by=['Eigenvector Centrality'], ascending=False).head()

partition = community_louvain.best_partition(G_full)

# Create a new attribute in the graph for community membership
for node, community in partition.items():
    G_full.nodes[node]['community'] = community

# Function to characterize communities
def characterize_community(G, community_nodes):
    subgraph = G_full.subgraph(community_nodes)
    density = nx.density(subgraph)
    clustering_coefficient = nx.average_clustering(subgraph)
    avg_degree = sum(dict(subgraph.degree()).values()) / subgraph.number_of_nodes()
    return density, clustering_coefficient, avg_degree

# Apply the function to each community
community_characteristics = {}
unique_communities = set(partition.values())
for community in unique_communities:
    community_nodes = [node for node, comm in partition.items() if comm == community]
    community_characteristics[community] = characterize_community(G, community_nodes)

for node, community in partition.items():
    G_full.nodes[node]['community'] = community

# Sorting nodes by degree centrality
sorted_nodes = sorted(nx.degree_centrality(G_full).items(), key=lambda x: x[1], reverse=True)

# Keeping the top 200 nodes
top_200_nodes = [node for node, centrality in sorted_nodes[:200]]

# Create a subgraph with the top 100 nodes
G_sub = G_full.subgraph(top_200_nodes)
layout = nx.layout.spring_layout(G_sub)
graph = hv.Graph.from_networkx(G_sub, layout).opts(tools=['hover'], node_color='community')
labels = hv.Labels(graph.nodes, ['x', 'y'], 'index')
bundled = bundle_graph(graph)

st.sidebar.title("Options for data analysis")
choose = st.sidebar.selectbox(
    "Select an option for data analysis", 
    ["Visualisations",
     "Data Calculations"]
)
if choose == "Visualisations":
    # Displaying the Visualisations header
    st.header("Visualisations")
    
    # Dropdown to select the type of visualization
    visualization_option = st.selectbox(
        "Select Visualization", 
        ["Distributions of weights",
         "Nodes with the most edges in a network",
         "Network with communities",
         "Filtered network to top 200 nodes and their communities"]  # This should match the if condition below
    )
    # Visualizations based on user selection
    if visualization_option == "Distributions of weights":  # This should match the option above
        # Making a histogram on the weights
        plt.figure(figsize=(12, 6))
        plt.hist(df['weight'], bins=50, color='skyblue', edgecolor='black')
        plt.title('Distribution of Weights')
        plt.xlabel('Weight')
        plt.ylabel('Frequency')
        plt.grid(True)
        st.pyplot(plt)  # Correctl2y display the plot in Streamlit
    elif visualization_option == "Nodes with the most edges in a network":
        plt.figure(figsize=(10, 10))
        pos = nx.spring_layout(G, k=0.15, iterations=20)
        nx.draw_networkx(G, pos, with_labels=True, node_size=50, font_size=10, edge_color='grey', linewidths=1)
        plt.title('Top 10% Weighted 1-Mode Graph')
        plt.axis('off')
        st.pyplot(plt)
    elif visualization_option == "Network with communities":
        plt.figure(figsize=(10, 10))
        pos = nx.spring_layout(G_full)  # you can use other layout algorithms as well
        cmap = plt.get_cmap('viridis')  # choosing a colormap
        colors = [partition[node] for node in G_full.nodes]  # assigning colors based on community
        nx.draw_networkx_nodes(G_full, pos, node_color=colors, cmap=cmap)
        nx.draw_networkx_edges(G_full, pos, alpha=0.2)
        plt.title('Network with Communities')
        plt.axis('off')
        st.pyplot(plt)
    elif visualization_option == "Filtered network to top 200 nodes and their communities":
        for node, community in partition.items():
            G_full.nodes[node]['community'] = community
        # Sorting nodes by degree centrality
        sorted_nodes = sorted(nx.degree_centrality(G_full).items(), key=lambda x: x[1], reverse=True)
        # Keeping the top 200 nodes
        top_200_nodes = [node for node, centrality in sorted_nodes[:200]]
        # Create a subgraph with the top 200 nodes
        G_sub = G_full.subgraph(top_200_nodes)
        # Create a layout for our nodes
        pos = nx.spring_layout(G_sub)
        # Prepare a colormap that has as many colors as there are communities
        unique_communities = list(set(partition.values()))
        community_colors = plt.cm.viridis(np.linspace(0, 1, len(unique_communities)))
        # Map communities to colors
        community_to_color = {community: color for community, color in zip(unique_communities, community_colors)}
        # Draw the network using Matplotlib
        plt.figure(figsize=(10, 10))
        # Draw nodes with colors based on their community
        for community in unique_communities:
            nodes_of_community = [node for node in G_sub.nodes if G_sub.nodes[node].get('community') == community]
            nx.draw_networkx_nodes(G_sub, pos, nodelist=nodes_of_community, node_color=[community_to_color[community]], label=f'Community {community}')
        # Draw edges
        nx.draw_networkx_edges(G_sub, pos, alpha=0.3, edge_color='gray')
        # Draw node labels
        nx.draw_networkx_labels(G_sub, pos, font_size=7, font_color='black')
        # Set the title and turn off the axis
        plt.title('Top 200 Nodes with Communities')
        plt.axis('off')
        # Optionally, create a legend for the communities
        plt.legend(scatterpoints=1, frameon=False, labelspacing=1, loc='best')
        # Show the plot
        st.pyplot(plt)

if choose == "Data Calculations":
    # Displaying the Visualisations header
    st.header("Data Calculations")
    
    # Dropdown to select the type of visualization
    data_option = st.selectbox(
        "Select information you want to see", 
        ["Basic information about the data",
         "Top centrality sorted"]  # This should match the if condition below
    )
    # Visualizations based on user selection
    if data_option == "Basic information about the data":
        # Calculate basic statistics to understand the distribution
        weight_stats = df['weight'].describe()
        tab1, tab2 = st.tabs(["First 10 ten rows of the data", "All the weights summarized"])
        with tab1:
            st.dataframe(df.head(10))
        with tab2:
            st.dataframe(weight_stats)
    elif data_option == "Top centrality sorted":
        tab1, tab2, tab3 = st.tabs(["Storted by Degree Centrality", "Sorted by Betweenness Centrality", "Sorted by Eigenvector Centrality"])
        with tab1:
            st.dataframe(centrality_measures_df.sort_values(by=['Degree Centrality'], ascending=False).head(10))
        with tab2:
            st.dataframe(centrality_measures_df.sort_values(by=['Betweenness Centrality'], ascending=False).head(10))
        with tab3:
            st.dataframe(centrality_measures_df.sort_values(by=['Eigenvector Centrality'], ascending=False).head(10))

st.header("Main take aways from the network analysis")

st.subheader("Top 5 eigenvector centrality twitter profiles")

st.markdown("""Nancy Pelosi (@SpeakerPelosi): Democratic congresswoman representing California's 11th district, and the 52nd Speaker of the House known for her legislative influence and leadership​1​​2​. 

Kevin McCarthy (@GOPLeader): Republican congressman serving as the House Minority Leader, representing California's 23rd district, and a prominent Republican voice in the House​3​.

Steny Hoyer (@LeaderHoyer): A Democratic politician and attorney, long-serving U.S. Representative for Maryland's 5th congressional district, and former House Majority Leader​4​​5​.

Bobby Rush (@RepBobbyRush): Democratic Party. A former U.S. Representative for Illinois's 1st congressional district, a civil rights activist, and co-founder of the Illinois Black Panther Party. He was first elected to Congress in 1992.

Scott Franklin (@RepFranklin): Republican congressman representing Florida's 18th district, a businessman, and has served in Congress since 2021​6​.""")

st.markdown("""References:

Nancy Pelosi @SpeakerPelosi, Twitter Profile - twstalker.com

Kevin McCarthy - Wikipedia

Steny Hoyer - Wikipedia

Congressman Scott Franklin - official website

Scott Franklin - Ballotpedia""")

st.subheader("Conclusion")

st.markdown(""" 
Eigenvector centrality is like a popularity score that looks at not just how many followers a Twitter profile has, but also who those followers are. 
These top 5 profiles have the highest eigenvector centrality, it means they're not just followed by many people, but by the right kind of people—other influential accounts. 
This creates a ripple effect, where their tweets are seen and shared by users who have a big audience themselves. So, a high eigenvector centrality on Twitter suggests that these profiles are super influential and can make their tweets travel far and wide.
""")
