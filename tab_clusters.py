from os.path import basename
from os import walk
from pathlib import Path
from pprint import pprint

import click
from click import argument, command, option
import matplotlib.pyplot as plt
from numpy import array
from pandas import DataFrame
from pyquery import PyQuery as pq
import seaborn
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer


def samples_from_dir(in_dir):
    """Return an iterable of Paths to samples found in ``in_dir``,
    recursively."""
    for dir_path, dirs, files in walk(in_dir):
        try:
            # Skip resources/ folders. Sometimes they contain .html files, and
            # those aren't samples.
            dirs.remove('resources')
        except ValueError:
            pass
        yield from (Path(dir_path) / file for file in files
                    if file.endswith('.html'))


@command()
@argument('folder', type=click.Path(exists=True, file_okay=False))
@option('--lsa/--no-lsa', default=True)
def tab_clusters(folder, lsa):
    # Read samples:
    paths_and_docs = []
    failures = 0
    for path in samples_from_dir(folder):
        try:
            paths_and_docs.append((basename(path), text_from_sample(path)))
        except (UnicodeDecodeError, ValueError):  # lxml throws ValueErrors when it has internal unicode trouble.
            failures += 1
            pass
    paths, docs = zip(*paths_and_docs)
    print(f'{failures} had a unicode decode error.')

    # Build TF/IDF matrix:
    vectorizer = TfidfVectorizer(max_df=.6)  # Decrease max_df to be more aggressive about declaring things stopwords.
    tfidf_docs = vectorizer.fit_transform(docs)
    print(f'Stopwords: {vectorizer.stop_words_}')

    if lsa:
        # Do SVD to reduce matrix size and (tend to) merge synonyms and split
        # polysemes:
        decompose_and_normalize = make_pipeline(TruncatedSVD(100),
                                                Normalizer(copy=False))
        # McCormick normalizes the vectors after SVD, I guess so he can use
        # cosine distance for classifying. Is there a point to our doing it?
        lsa_docs = decompose_and_normalize.fit_transform(tfidf_docs)
        vectors_to_cluster = lsa_docs
    else:
        vectors_to_cluster = tfidf_docs.toarray()

    # Cluster:
    clustering = AgglomerativeClustering(n_clusters=None, linkage='ward', compute_full_tree=True, distance_threshold=1.45)
    clustering.fit(vectors_to_cluster)
    print(f'Found {clustering.n_clusters_} clusters:')
    path_clusters = [[] for _ in range(clustering.n_clusters_)]
    vector_clusters = [[] for _ in range(clustering.n_clusters_)]

    for path_index, label in enumerate(clustering.labels_):
        vector_clusters[label].append(vectors_to_cluster[path_index])

    # Viz:
    frames = []
    for cluster_num, cluster in enumerate(vector_clusters):
        lower_dimensional = TSNE().fit_transform(vector_clusters[0])  # Should I fit this once and then use it to transform them all?
        for x, y in lower_dimensional:
            frames.append((x, y, cluster_num))
    seaborn.scatterplot(
        x='x',
        y='y',
        hue='cluster',
        data=DataFrame(frames, columns=['x', 'y', 'cluster']),
        palette='bright'
    )
    plt.show()

    for path_index, label in enumerate(clustering.labels_):
        path_clusters[label].append(paths[path_index])
    pprint(path_clusters)


def text_from_sample(filename):
    """Return the innerText (or thereabouts) from an HTML file."""
    with open(filename, encoding='utf-8', errors='ignore') as file:
        return pq(file.read()).remove('script').remove('style').text()


if __name__ == '__main__':
    tab_clusters()


# NEXT: HTML signal, throw the URL or domain in as signal. Measure that first cluster, which seems to be pretty misc, so see if it has high spread/variance or something that I can ignore it based on.
# TODO: Try switching to cosine distance rather than Euclidean. That should keep documents from differing just by dint of being different lengths. Though does TFIDF inherently normalize the vectors itself? Yes. TfidfVectorizer normalizes in a way such that cosine similarity is dot product when using norm='l2', which is the default. So never mind; cosine similarity SHOULD not change things, if cos similarity is always the same as Euclidean on normalized vectors. It's proportional to (1 - cos sim).
