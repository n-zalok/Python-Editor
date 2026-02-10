import matplotlib.pyplot as plt

def plot_embeddings(embeddings, values, cmap="coolwarm"):
    """
    Plot embeddings with color intensity based on a continuous value
    :param embeddings: (n, 2) array of embeddings
    :param values: continuous values (n,)
    """
    fig = plt.figure(figsize=(16, 9))

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