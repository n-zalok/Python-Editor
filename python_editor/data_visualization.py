import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
import umap


def plot_embeddings(embeddings, values, cmap="coolwarm"):
    """
    Plot embeddings with color intensity based on a continuous value
    :param embeddings: (n, 2) array of embeddings
    :param values: continuous values (n,)
    """
    
    sc = plt.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        c=values,              # continuous values
        cmap=cmap,        # colormap (try plasma, inferno, magma, coolwarm)
        s=40,
        alpha=0.6
    )

    plt.colorbar(sc, label="Value")  # heatmap scale

    plt.gca().set_aspect("equal", "box")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()


def cluster(to_cluster, range_of_clusters: int, random_state: int):
    # dimensionality reduction
    umap_embedder = umap.UMAP()
    umap_emb = umap_embedder.fit_transform(to_cluster)

    # color set
    cmap = plt.get_cmap("Set2")

    for n_clusters in range(2, range_of_clusters+1):
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        ax1.set_xlim([-1, 1])
        # points will be layed accross y axis
        ax1.set_ylim([0, len(to_cluster)])

        # Initialize the clusterer with n_clusters value and a random generator
        clusterer = KMeans(n_clusters=n_clusters, random_state=random_state)
        cluster_labels = clusterer.fit_predict(to_cluster)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed clusters
        silhouette_avg = silhouette_score(to_cluster, cluster_labels, metric='cosine')
        print(f"For n_clusters = {n_clusters}, The average silhouette_score is : {silhouette_avg}")

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(to_cluster, cluster_labels, metric='cosine')

        y_lower = 0
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()

            # each cluster will occupy space proportional to number of its points
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cmap(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                            0, ith_cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cmap(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(umap_emb[:, 0], umap_emb[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centerss = clusterer.cluster_centers_
        print(centerss.shape)
        centers = umap_embedder.transform(centerss)
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                    "with n_clusters = %d" % n_clusters),
                    fontsize=14, fontweight='bold')