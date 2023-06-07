"""
This script is just for validating that my translation from tensorflow in pytorch was correct
"""

"""
Tensorflow code from the paper:

import numpy as np
import tensorflow.compat.v2 as tf

def _calculate_action_cost_matrix(ac1, ac2):
  diff = tf.expand_dims(ac1, axis=1) - tf.expand_dims(ac2, axis=0)
  return tf.cast(tf.reduce_mean(tf.abs(diff), axis=-1), dtype=tf.float32)


def metric_fixed_point_fast(cost_matrix, gamma=0.99, eps=1e-7):
  # Dynamic programming for calculating PSM.
  d = np.zeros_like(cost_matrix)
  def operator(d_cur):
    d_new = 1 * cost_matrix
    discounted_d_cur = gamma * d_cur
    d_new[:-1, :-1] += discounted_d_cur[1:, 1:]
    d_new[:-1, -1] += discounted_d_cur[1:, -1]
    d_new[-1, :-1] += discounted_d_cur[-1, 1:]
    return d_new

  while True:
    d_new = operator(d)
    if np.sum(np.abs(d - d_new)) < eps:
      break
    else:
      d = d_new[:]
  return d


def compute_metric(actions1, actions2, gamma):
  action_cost = _calculate_action_cost_matrix(actions1, actions2)
  return tf_metric_fixed_point(action_cost, gamma=gamma)


@tf.function
def tf_metric_fixed_point(action_cost_matrix, gamma):
  return tf.numpy_function(metric_fixed_point_fast, [action_cost_matrix, gamma], Tout=tf.float32)


def cosine_similarity(x, y):
  # Computes cosine similarity between all pairs of vectors in x and y.
  x_expanded, y_expanded = x[:, tf.newaxis], y[tf.newaxis, :]
  similarity_matrix = tf.reduce_sum(x_expanded * y_expanded, axis=-1)
  similarity_matrix /= (tf.norm(x_expanded, axis=-1) * tf.norm(y_expanded, axis=-1) + 1e-8)
  return similarity_matrix


def contrastive_loss(similarity_matrix,
    metric_values,
    temperature,
    beta=1.0):
    #Contrative Loss with embedding similarity.
    metric_shape = tf.shape(metric_values)
    ## z_\theta(X): embedding_1 = nn_model.representation(X)
    ## z_\theta(Y): embedding_2 = nn_model.representation(Y)
    ## similarity_matrix = cosine_similarity(embedding_1, embedding_2
    ## metric_values = PSM(X, Y)
    similarity_matrix /= temperature
    neg_logits1 = similarity_matrix

    col_indices = tf.cast(tf.argmin(metric_values, axis=1), dtype=tf.int32)
    pos_indices1 = tf.stack(
    (tf.range(metric_shape[0], dtype=tf.int32), col_indices), axis=1)
    pos_logits1 = tf.gather_nd(similarity_matrix, pos_indices1)

    metric_values /= beta
    similarity_measure = tf.exp(-metric_values)
    pos_weights1 = -tf.gather_nd(metric_values, pos_indices1)
    pos_logits1 += pos_weights1
    negative_weights = tf.math.log((1.0 - similarity_measure) + 1e-8)
    neg_logits1 += tf.tensor_scatter_nd_update(
    negative_weights, pos_indices1, pos_weights1)

    neg_logits1 = tf.math.reduce_logsumexp(neg_logits1, axis=1)
    return tf.reduce_mean(neg_logits1 - pos_logits1) # Equation 4



e1 = np.array([[.7, .54, .345], [.6, .4625, .423], [.9876, .2, .3]], np.float64)
e2 = np.array([[.123, .56, .3], [.3, 5.2, .1], [.2763, .928, .786]], np.float64)

a1 = np.array([[0], [1], [0]])
a2 = np.array([[1], [0], [0]])

e1 = tf.convert_to_tensor(e1)
e2 = tf.convert_to_tensor(e2)
a1 = tf.convert_to_tensor(a1)
a2 = tf.convert_to_tensor(a2)

sim_matrix = cosine_similarity(e1, e2)
psm = compute_metric(a1, a2, gamma=0.99)
psm = tf.cast(psm, tf.float64)

print(contrastive_loss(
    sim_matrix,
    psm,
    1.
))
"""

import torch
import numpy as np
from psm import psm_paper


def cosine_similarity(a, b, eps=1e-8):
    """
    Computes cosine similarity between all pairs of vectors in x and y
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def contrastive_loss(similarity_matrix, metric_values, temperature=1.0, beta=1.0):
    """Contrative Loss with embedding similarity."""
    # z_\theta(X): embedding_1 = nn_model.representation(X)
    # z_\theta(Y): embedding_2 = nn_model.representation(Y)
    # similarity_matrix = cosine_similarity(embedding_1, embedding_2
    # metric_values = PSM(X, Y)
    metric_shape = metric_values.size()
    similarity_matrix /= temperature
    neg_logits1 = similarity_matrix

    col_indices = torch.argmin(metric_values, dim=1)
    pos_indices1 = torch.stack(
        (torch.arange(metric_shape[0], dtype=torch.int32), col_indices), dim=1)
    pos_logits1 = similarity_matrix[pos_indices1[:, 0], pos_indices1[:, 1]]

    metric_values /= beta
    similarity_measure = torch.exp(-metric_values)
    pos_weights1 = -metric_values[pos_indices1[:, 0], pos_indices1[:, 1]]
    pos_logits1 += pos_weights1
    negative_weights = torch.log((1.0 - similarity_measure) + 1e-8)
    negative_weights[pos_indices1[:, 0], pos_indices1[:, 1]] = pos_weights1

    neg_logits1 += negative_weights

    neg_logits1 = torch.logsumexp(neg_logits1, dim=1)
    return torch.mean(neg_logits1 - pos_logits1)  # Equation 4


e1 = np.array([.7, .54, .345, .6, .4625, .423, .9876, .2, .3, .34, .134, 2], np.float64)
e2 = np.array([.123, .56, .3, .3, 5.2, .1, .2763, .928, .786, .23, .436, 43], np.float64)

a1 = np.array([0, 1, 0, 0])
a2 = np.array([1, 0, 0, 0])

e1 = torch.tensor(e1)
e2 = torch.tensor(e2)

sim_matrix = cosine_similarity(e1, e2)
psm = psm_paper(a1, a2)

print(contrastive_loss(
    sim_matrix,
    torch.tensor(psm),
    1.
))
