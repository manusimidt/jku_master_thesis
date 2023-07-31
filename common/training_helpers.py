import torch


def contrastive_loss_paper(similarity_matrix, metric_values, temperature=1.0, beta=1.0):
    """
    Contrastive Loss with embedding similarity.
    Taken from Agarwal.et.al. rewritten in pytorch
    """
    # z_\theta(X): embedding_1 = nn_model.representation(X)
    # z_\theta(Y): embedding_2 = nn_model.representation(Y)
    # similarity_matrix = cosine_similarity(embedding_1, embedding_2
    # metric_values = PSM(X, Y)
    metric_shape = metric_values.size()
    similarity_matrix /= temperature
    neg_logits1 = similarity_matrix

    col_indices = torch.argmin(metric_values, dim=1)
    pos_indices1 = torch.stack(
        (torch.arange(metric_shape[0], dtype=torch.int32, device=col_indices.device), col_indices), dim=1)
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


def contrastive_loss_repository(similarity_matrix, metric_values, temperature=1.0):
    """
    Contrastive Loss with embedding similarity.
    Taken from Agarwal.et.al. rewritten in pytorch
    """
    """
    Contrastive Loss with embedding similarity.
    Taken from Agarwal.et.al. rewritten in pytorch
    """
    metric_shape = metric_values.size()
    similarity_matrix /= temperature
    neg_logits1, neg_logits2 = similarity_matrix, similarity_matrix

    col_indices = torch.argmin(metric_values, dim=1)
    pos_indices1 = torch.stack(
        (torch.arange(metric_shape[0], dtype=torch.int32, device=col_indices.device), col_indices), dim=1)
    # pos_logits1 = torch.gather(similarity_matrix, dim=-1, index=pos_indices1)
    pos_logits1 = similarity_matrix[tuple(pos_indices1.t())]

    row_indices = torch.argmin(metric_values, dim=0)
    pos_indices2 = torch.stack(
        (row_indices, torch.arange(metric_shape[1], dtype=torch.int32, device=col_indices.device)), dim=1)
    # pos_logits2 = torch.gather(similarity_matrix, dim=0, index=pos_indices2)
    pos_logits2 = similarity_matrix[tuple(pos_indices2.t())]

    neg_logits1 = torch.logsumexp(neg_logits1, dim=1)
    neg_logits2 = torch.logsumexp(neg_logits2, dim=0)

    loss1 = torch.mean(neg_logits1 - pos_logits1)
    loss2 = torch.mean(neg_logits2 - pos_logits2)
    return loss1 + loss2


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
