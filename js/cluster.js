// Port of clustering to JS

//import {matrix} from "mathjs";
import {clusters as clustersModule} from "fathom-web";
const clusters = clustersModule.clusters;

/**
 * Return dot product of 2 vectors.
 */
function dotProduct(a, b) {
  return a.map((x, i) => x * b[i]).reduce((m, n) => m + n);
}

function cosineSimilarity(a, b) {
  return dotProduct(a, b) / (norm(A) * norm(B))
}

/**
 * Do vector subtraction: a - b.
 */
function difference(a, b) {
  return a.map((e, i) => e - b[i]);
}

/**
 * Return the L2-normalized Euclidean distance, which gives the same comparison
 * ordering as cosine distance.
 * https://en.wikipedia.org/wiki/Cosine_similarity#%7F'%22%60UNIQ--
 * postMath-00000011-QINU%60%22'%7F-normalised_Euclidean_distance
 *
 * Never mind; I need honest distance, not comparison. I'm not sorting; I'm clustering.
 */
function distance(a, b) {
  return magnitude(difference(l2Norm(a), l2Norm(b)));
}

function sum(v) {
  return v.reduce((acc, cur) => acc + cur);
}

/**
 * Return the length of a vector.
 */
function magnitude(v) {
  return Math.sqrt(sum(v.map(e => e * e)));
}

/**
 * Turn a vector into a unit vector.
 */
function l2Norm(v) {
  return v.map(e => e / magnitude(v));
}

//console.log(dotProduct([1, 2], [3, 4]));
console.log(clusters([[0, 1], [1, 0], [9, 3], [8, 2]],
                     1,
                     distance));
//console.log(l2Norm([3, 4]));

// NEXT: Construct some term vectors. Remember to add 1 to everything (Laplace smoothing) so we don't have divide-by-zeros. Do TFIDF if you like. Or maybe just scale to unit vectors.