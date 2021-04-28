from os.path import basename
from sys import argv
from os import walk
from pathlib import Path

from pprint import pprint
from pyquery import PyQuery as pq
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer


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


def tab_clusters(folder):
    paths_and_docs = []
    failures = 0
    for path in samples_from_dir(folder):
        try:
            paths_and_docs.append((basename(path), text_from_sample(path)))
        except (UnicodeDecodeError, ValueError):  # lxml throws ValueErrors when it has internal unicode trouble.
            failures += 1
            pass
    print(f'{failures} had a unicode decode error.')
    paths, docs = zip(*paths_and_docs)
    vectorizer = TfidfVectorizer(max_df=.6)  # Decrease max_df to be more aggressive about declaring things stopwords.
    tfidfs = vectorizer.fit_transform(docs)
    print(f'Stopwords: {vectorizer.stop_words_}')
    clustering = AgglomerativeClustering(n_clusters=None, linkage='ward', compute_full_tree=True, distance_threshold=1.45)
    clustering.fit(tfidfs.toarray())
    print(f'Found {clustering.n_clusters_} clusters:')
    clusters = [[] for _ in range(clustering.n_clusters_)]
    for path_index, label in enumerate(clustering.labels_):
        clusters[label].append(paths[path_index])
    pprint(clusters)


def text_from_sample(filename):
    """Return the innerText (or thereabouts) from an HTML file."""
    with open(filename, encoding='utf-8', errors='ignore') as file:
        return pq(file.read()).remove('script').remove('style').text()


if __name__ == '__main__':
    tab_clusters(argv[1])


# NEXT: HTML signal, throw the URL or domain in as signal. Measure that first cluster, which seems to be pretty misc, so see if it has high spread/variance or something that I can ignore it based on.
# TODO: Try switching to cosine distance rather than Euclidean. That should keep documents from differing just by dint of being different lengths. Though does TFIDF inherently normalize the vectors itself? Yes. TfidfVectorizer normalizes in a way such that cosine similarity is dot product when using norm='l2', which is the default. So never mind; cosine similarity SHOULD not change things, if cos similarity is always the same as Euclidean on normalized vectors. It's proportional to (1 - cos sim).
# TODO: Am I accidentally doing latent semantic analysis, or is that something I should consider separately? It's separate: it does some things I don't.
