import torch.nn

from .lovacz_losses import lovasz_hinge


def convex(value1, value2, value1_weight=1.0):
    """

    :param value1: First loss
    :param value2: Second loss
    :param value1_weight: weight of first loss(weight for second loss = 1. - value1_weight)
    :return: weighted sum of two losses
    """

    return value1_weight * value1 + (1.0 - value1_weight) * value2


def pseudoLap1_weight_loop(l1_loss):

    # l1_loss[l1_loss > 0.01] *=2
    # l1_loss[l1_loss > 0.02] *= 2
    # l1_loss[l1_loss > 0.04] *= 2
    # l1_loss[l1_loss > 0.06] *= 2
    # l1_loss[l1_loss > 0.1] *= 2
    l1_loss = l1_loss.where(l1_loss <= 0.01, l1_loss * 2)
    for i in range(1, 6):
        l1_loss = l1_loss.where(l1_loss <= i * 2e-2, l1_loss * 2)
    return l1_loss


def pseudoLap1_weight_no_loop(l1_loss):

    l1_loss = l1_loss.where(l1_loss <= 0.01, l1_loss * 2)
    l1_loss = l1_loss.where(l1_loss <= 0.02, l1_loss * 2)
    l1_loss = l1_loss.where(l1_loss <= 0.04, l1_loss * 2)
    l1_loss = l1_loss.where(l1_loss <= 0.06, l1_loss * 2)
    l1_loss = l1_loss.where(l1_loss <= 0.1, l1_loss * 2)

    # l1_loss = l1_loss.where(l1_loss <= 0.01, l1_loss * 2)
    # for i in range(1,6):
    #     l1_loss = l1_loss.where(l1_loss <= i*2e-2, l1_loss * 2)
    return l1_loss


def mapping_l2(y_pred, y_true):
    """
    :param y_pred:
    :param y_true:
    :return:

    Mapping L2 loss:
     .. math::
        mappingloss = MSE_x + (x_{true}^{2}-x_{pred}^{2})*(x_{true}^{1}-x_{pred}^{1}) + ...
    """

    ## x_1 = [..., 0], x_2 = [..., 2], y_1 = [..., 1], y_2 = [..., 3]

    first_x = y_true[..., 0] - y_pred[..., 0]
    second_x = y_true[..., 2] - y_pred[..., 2]
    loss_x = first_x * first_x + first_x * second_x + second_x * second_x

    first_y = y_true[..., 1] - y_pred[..., 1]
    second_y = y_true[..., 3] - y_pred[..., 3]
    loss_y = first_y * first_y + first_y * second_y + second_y * second_y

    map_loss = loss_x + loss_y

    return map_loss


def mapping_l1(y_pred, y_true):
    """

    Mapping l1 loss:
      .. math:: mappingloss = \int_{0}^{1} \left( ( x \\ y)_1 (t)- (x \\ y)_2 (t) )_p d(t)
    :param y_pred: pred y lines with [...,[x0,y0,x1,y1,...]]
    :type y_pred: torch tensor size bx10x6
    :param y_true: ground truth of y lines with [...,[x0,y0,x1,y1,...]]
    :type y_pred: torch tensor size bx10x6
    :return:
    """
    first_x = y_true[..., 0] - y_pred[..., 0]
    second_x = y_true[..., 2] - y_pred[..., 2]
    # first_x[first_x == second_x] += 1e-7
    first_x = first_x.where(first_x != second_x, first_x + 1e-7)

    loss_x = (
        -(first_x + second_x) * ((second_x <= 0) * (first_x < 0)).type(first_x.dtype)
        + (first_x + second_x) * ((second_x >= 0) * (first_x >= 0)).type(first_x.dtype)
        - (first_x**2 + second_x**2) / (first_x - second_x) * ((second_x > 0) * (first_x < 0)).type(first_x.dtype)
        + (first_x**2 + second_x**2) / (first_x - second_x) * ((second_x <= 0) * (first_x > 0)).type(first_x.dtype)
    )

    first_y = y_true[..., 1] - y_pred[..., 1]
    second_y = y_true[..., 3] - y_pred[..., 3]
    first_y = first_y.where(first_y != second_y, first_y + 1e-7)
    # first_y[first_y == second_y] += 1e-7
    loss_y = (
        -(first_y + second_y) * ((second_y <= 0) * (first_y < 0)).type(first_y.dtype)
        + (first_y + second_y) * ((second_y >= 0) * (first_y >= 0)).type(first_y.dtype)
        - (first_y**2 + second_y**2) / (first_y - second_y) * ((second_y > 0) * (first_y < 0)).type(first_y.dtype)
        + (first_y**2 + second_y**2) / (first_y - second_y) * ((second_y <= 0) * (first_y > 0)).type(first_y.dtype)
    )

    map_loss = loss_x + loss_y

    return map_loss


def vectran_loss(y_pred, y_true, l2_weight=0.5, bce_weight=0.5, reduction="mean", **kwargs):
    """

    :param y_pred: pred y lines with [...,[x0,y0,x1,y1,...]]
    :param y_true: ground truth y lines with [...,[x0,y0,x1,y1,...]]
    :param l2_weight: weight of l2 loss in convex function
    :param bce_weight: weight of bce loss in convex function
    :param reduction: default mean, if none saved  mean loss by sample
    :return: loss function of mapping
    """

    l1 = torch.nn.L1Loss(reduction=reduction)
    mse = torch.nn.MSELoss(reduction=reduction)
    bce = torch.nn.BCELoss(reduction=reduction)

    cpoints_pred, logits_pred = y_pred[..., :-1], y_pred[..., -1]
    cpoints_true, logits_true = y_true[..., :-1], y_true[..., -1]
    l1_loss = l1(cpoints_pred, cpoints_true)
    l2_loss = mse(cpoints_pred, cpoints_true)
    bce_loss = bce(logits_pred, logits_true)

    if "none" == reduction:
        l1_loss = l1_loss.mean((1, 2))
        l2_loss = l2_loss.mean((1, 2))
        bce_loss = bce_loss.mean((1))
    endpoint_loss = convex(l2_loss, l1_loss, l2_weight)
    loss = convex(bce_loss, endpoint_loss, bce_weight)
    return loss


def vectran_mapping_L2(y_pred, y_true, l2_weight=0.5, bce_weight=0.5, reduction="mean", width_weight=0.2, **kwargs):
    """


    :param y_pred: pred y lines with [...,[x0,y0,x1,y1,...]]
    :param y_true: ground truth y lines with [...,[x0,y0,x1,y1,...]]
    :param bce_weight: weight of bce loss in convex function
    :param reduction: default mean, if none saved  mean loss by sample
    :param width_weight: wight of l2 loss of width
    :param l2_weight:
    :param kwargs:
    :return:
    """
    l1 = torch.nn.L1Loss(reduction=reduction)
    bce = torch.nn.BCELoss(reduction=reduction)

    ### For width
    if width_weight:
        mse = torch.nn.MSELoss(reduction=reduction)
        width_l2_loss = mse(y_pred[..., -2], y_true[..., -2])

    cpoints_pred, logits_pred = y_pred[..., :-1], y_pred[..., -1]
    cpoints_true, logits_true = y_true[..., :-1], y_true[..., -1]
    l1_loss = l1(cpoints_pred, cpoints_true)
    map_loss = mapping_l2(cpoints_pred, cpoints_true)
    bce_loss = bce(logits_pred, logits_true)

    if "none" == reduction:
        l1_loss = l1_loss.mean((1, 2))
        map_loss = map_loss.mean((1))
        bce_loss = bce_loss.mean((1))
        if width_weight:
            width_l2_loss = width_l2_loss.mean((1))
    else:
        map_loss = map_loss.mean()

    if width_weight:
        map_loss = convex(width_l2_loss, map_loss, width_weight)

    endpoint_loss = convex(map_loss, l1_loss, l2_weight)
    loss = convex(bce_loss, endpoint_loss, bce_weight)
    return loss


def vectran_mapping_L1(y_pred, y_true, l2_weight=0.5, bce_weight=0.5, reduction="mean", width_weight=0.2, **kwargs):
    """

    :param y_pred: pred y lines with [...,[x0,y0,x1,y1,...]]
    :param y_true: ground truth y lines with [...,[x0,y0,x1,y1,...]]
    :param bce_weight: weight of bce loss in convex function
    :param reduction: default mean, if none saved  mean loss by sample
    :param width_weight: wight of l2 loss of width
    :param l2_weight:
    :param kwargs:
    :return:
    """
    l1 = torch.nn.L1Loss(reduction=reduction)
    bce = torch.nn.BCELoss(reduction=reduction)

    ### For width
    if width_weight:
        l1_w = torch.nn.MSELoss(reduction=reduction)
        width_l1_loss = l1_w(y_pred[..., -2], y_true[..., -2])

    cpoints_pred, logits_pred = y_pred[..., :-1], y_pred[..., -1]
    cpoints_true, logits_true = y_true[..., :-1], y_true[..., -1]
    l1_loss = l1(cpoints_pred, cpoints_true)
    map_loss = mapping_l1(cpoints_pred, cpoints_true)
    bce_loss = bce(logits_pred, logits_true)

    if "none" == reduction:
        l1_loss = l1_loss.mean((1, 2))
        map_loss = map_loss.mean((1))
        bce_loss = bce_loss.mean((1))
        if width_weight:
            width_l1_loss = width_l1_loss.mean((1))
    else:
        map_loss = map_loss.mean()

    if width_weight:
        map_loss = convex(width_l1_loss, map_loss, width_weight)

    endpoint_loss = convex(map_loss, l1_loss, l2_weight)
    loss = convex(bce_loss, endpoint_loss, bce_weight)
    return loss


def pseudoLap1_func(y_pred, y_true, bce_weight=0.5, reduction="mean", func=pseudoLap1_weight_loop, **kwargs):
    """

    :param y_pred: pred y lines with [...,[x0,y0,x1,y1,...]]
    :param y_true: ground truth y lines with [...,[x0,y0,x1,y1,...]]
    :param l2_weight: weight of l2 loss in convex function
    :param bce_weight: weight of bce loss in convex function
    :param reduction: default mean, if none saved  mean loss by sample
    :return: loss function of mapping
    """
    l1 = torch.nn.L1Loss(reduction="none")
    bce = torch.nn.BCELoss(reduction=reduction)

    cpoints_pred, logits_pred = y_pred[..., :-1], y_pred[..., -1]
    cpoints_true, logits_true = y_true[..., :-1], y_true[..., -1]
    l1_loss = l1(cpoints_pred, cpoints_true)
    bce_loss = bce(logits_pred, logits_true)
    endpoint_loss = func(l1_loss)
    if "none" == reduction:
        endpoint_loss = endpoint_loss.mean((1, 2))
        bce_loss = bce_loss.mean((1))
    else:
        endpoint_loss = endpoint_loss.mean()
    loss = convex(bce_loss, endpoint_loss, bce_weight)
    return loss


from functools import partial

pseudoLap1_loop = partial(pseudoLap1_func, func=pseudoLap1_weight_loop)
pseudoLap1_no_loop = partial(pseudoLap1_func, func=pseudoLap1_weight_no_loop)


def variable_vectran_loss(y_pred, y_true, mask=None, l2_weight=0.5, bce_weight=0.5, reduction="mean", **kwargs):
    """
    Loss function for variable-length primitive sequences.

    :param y_pred: predicted primitives [batch, max_primitives, features]
    :param y_true: ground truth primitives [batch, max_primitives, features]
    :param mask: mask indicating valid primitives [batch, max_primitives]
    :param l2_weight: weight of l2 loss in convex function
    :param bce_weight: weight of bce loss in convex function
    :param reduction: default mean, if none return loss per sample
    :return: loss value
    """
    if mask is None:
        # If no mask provided, assume all primitives are valid
        mask = torch.ones_like(y_true[..., 0])

    l1 = torch.nn.L1Loss(reduction="none")
    mse = torch.nn.MSELoss(reduction="none")
    bce = torch.nn.BCELoss(reduction="none")

    # For variable length, we assume the last feature is a confidence/logit
    # Split predictions and targets
    cpoints_pred, logits_pred = y_pred[..., :-1], y_pred[..., -1]
    cpoints_true, logits_true = y_true[..., :-1], y_true[..., -1]

    # Compute losses
    l1_loss = l1(cpoints_pred, cpoints_true)  # [batch, max_primitives, features-1]
    l2_loss = mse(cpoints_pred, cpoints_true)  # [batch, max_primitives, features-1]
    # BCE is unsafe under autocast; compute in full precision.
    with torch.cuda.amp.autocast(enabled=False):
        bce_loss = bce(logits_pred.float(), logits_true.float())  # [batch, max_primitives]

    # Apply mask
    l1_loss = (l1_loss * mask.unsqueeze(-1)).sum(dim=-1)  # [batch, max_primitives]
    l2_loss = (l2_loss * mask.unsqueeze(-1)).sum(dim=-1)  # [batch, max_primitives]
    bce_loss = bce_loss * mask  # [batch, max_primitives]

    # Average over valid primitives
    valid_count = mask.sum(dim=-1, keepdim=True).clamp(min=1)  # Avoid division by zero
    l1_loss = l1_loss.sum(dim=-1, keepdim=True) / valid_count
    l2_loss = l2_loss.sum(dim=-1, keepdim=True) / valid_count
    bce_loss = bce_loss.sum(dim=-1, keepdim=True) / valid_count

    # Combine losses
    endpoint_loss = convex(l2_loss, l1_loss, l2_weight)
    loss = convex(bce_loss, endpoint_loss, bce_weight)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss.squeeze(-1)
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def non_autoregressive_vectran_loss(y_pred, count_pred, y_true, count_true, mask=None, 
                                   l2_weight=0.5, bce_weight=0.5, count_weight=0.1, 
                                   reduction="mean", **kwargs):
    """
    Loss function for non-autoregressive transformer decoder with count prediction.

    :param y_pred: predicted primitives [batch, max_primitives, features]
    :param count_pred: predicted count logits [batch, max_primitives + 1]
    :param y_true: ground truth primitives [batch, max_primitives, features]
    :param count_true: ground truth count [batch] (number of valid primitives)
    :param mask: mask indicating valid primitives [batch, max_primitives]
    :param l2_weight: weight of l2 loss in convex function
    :param bce_weight: weight of bce loss in convex function
    :param count_weight: weight of count prediction loss
    :param reduction: default mean, if none return loss per sample
    :return: loss value
    """
    if mask is None:
        # If no mask provided, create mask based on count_true
        batch_size, max_primitives = y_true.shape[:2]
        mask = torch.zeros(batch_size, max_primitives, device=y_true.device)
        for b in range(batch_size):
            mask[b, :count_true[b]] = 1

    # Primitive prediction loss (same as variable_vectran_loss)
    l1 = torch.nn.L1Loss(reduction="none")
    mse = torch.nn.MSELoss(reduction="none")
    bce = torch.nn.BCELoss(reduction="none")

    # Split predictions and targets
    cpoints_pred, logits_pred = y_pred[..., :-1], y_pred[..., -1]
    cpoints_true, logits_true = y_true[..., :-1], y_true[..., -1]

    # Compute primitive losses
    l1_loss = l1(cpoints_pred, cpoints_true)  # [batch, max_primitives, features-1]
    l2_loss = mse(cpoints_pred, cpoints_true)  # [batch, max_primitives, features-1]
    # BCE is unsafe under autocast; compute in full precision.
    with torch.cuda.amp.autocast(enabled=False):
        bce_loss = bce(logits_pred.float(), logits_true.float())  # [batch, max_primitives]

    # Apply mask
    l1_loss = (l1_loss * mask.unsqueeze(-1)).sum(dim=-1)  # [batch, max_primitives]
    l2_loss = (l2_loss * mask.unsqueeze(-1)).sum(dim=-1)  # [batch, max_primitives]
    bce_loss = bce_loss * mask  # [batch, max_primitives]

    # Average over valid primitives
    valid_count = mask.sum(dim=-1, keepdim=True).clamp(min=1)  # Avoid division by zero
    l1_loss = l1_loss.sum(dim=-1, keepdim=True) / valid_count
    l2_loss = l2_loss.sum(dim=-1, keepdim=True) / valid_count
    bce_loss = bce_loss.sum(dim=-1, keepdim=True) / valid_count

    # Combine primitive losses
    endpoint_loss = convex(l2_loss, l1_loss, l2_weight)
    primitive_loss = convex(bce_loss, endpoint_loss, bce_weight)

    # Count prediction loss
    count_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    count_loss = count_loss_fn(count_pred, count_true)  # [batch]

    # Combine all losses
    total_loss = primitive_loss.squeeze(-1) + count_weight * count_loss

    if reduction == "mean":
        return total_loss.mean()
    elif reduction == "sum":
        return total_loss.sum()
    elif reduction == "none":
        return total_loss
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


prepare_losses = {
    "vectran_loss": vectran_loss,
    "vectran_mapping_L2": vectran_mapping_L2,
    "vectran_mapping_L1": vectran_mapping_L1,
    "variable_vectran_loss": variable_vectran_loss,
    "non_autoregressive_vectran_loss": non_autoregressive_vectran_loss,
    "pseudoLap1_loop": pseudoLap1_loop,
    "pseudoLap1_no_loop": pseudoLap1_no_loop,
    "lovasz_hinge": lovasz_hinge,
}
