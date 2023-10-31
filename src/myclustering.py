import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

def run_hierarchical_clustering(data, row_labels=None):
    linkage_matrix = linkage(data, method='ward')
    print(row_labels)
    dendrogram(linkage_matrix, labels = row_labels)
    # Visualize the dendrogram
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Data Point Index')
    plt.ylabel('Distance')
    plt.show()
    # Assign data points to clusters
    hierarchical_clusters = AgglomerativeClustering().fit_predict(data)
    cluster_indices = {}
    for i, cluster in enumerate(hierarchical_clusters):
        if cluster not in cluster_indices:
            cluster_indices[cluster] = []
        cluster_indices[cluster].append(i)
    for cluster, indices in cluster_indices.items():
        print(f"Cluster {cluster} ({len(indices)} data points): Indices {indices}")
    return

def run_kmeans(data, num_clusters, labels=None):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=20)
    clusters = kmeans.fit_predict(data)
    cluster_indices = {}
    for i, cluster in enumerate(clusters):
        if cluster not in cluster_indices:
            cluster_indices[cluster] = []
        cluster_indices[cluster].append(i)

    for cluster, indices in cluster_indices.items():
        print(f"Cluster {cluster} ({len(indices)} data points): Indices {indices}")

    # Define a colormap with a fixed number of colors
    colormap = plt.cm.get_cmap('Dark2_r', num_clusters)
    # Reduce dimensionality for plot
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    
    # Visualize the clusters with labels
    plt.figure(figsize=(8, 6))
    
    symbols = ['o', '*', 's', '^', 'D', 'v', 'p', 'H', '+']
    unique_clusters = np.unique(clusters)
    unique_labels = np.unique(labels)
    print('labels:',unique_labels)
    for i, cluster in enumerate(unique_clusters):
        cluster_indices = np.where(clusters == cluster)
        label = f'Cluster {cluster}'
        symbol = symbols[i % len(symbols)]
        
        for j, label_value in enumerate(unique_labels):
            label_indices = np.where(labels == label_value)
            intersect_indices = np.intersect1d(cluster_indices, label_indices)
            
            if len(intersect_indices) > 0:
                label = f'Cluster {cluster}, Label {label_value}'
                symbol = symbols[j % len(symbols)]
                scatter =plt.scatter(reduced_data[intersect_indices, 0], reduced_data[intersect_indices, 1],
                            label=label, c=colormap(i % num_clusters), marker=symbol)
    
    plt.title('K-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    # Create a legend with cluster and label combinations
    legend_labels = [f'Cluster {i}' for i in unique_clusters]
    for j, label_value in enumerate(unique_labels):
        legend_labels.extend([f'Cluster {i}, Label {label_value}' for i in unique_clusters])
    legend = plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels)
    plt.colorbar(label='Cluster')
    plt.show()
    return