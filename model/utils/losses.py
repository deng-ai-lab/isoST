import geomloss
import torch

def one_nn_mse(pred_coords, pred_features, true_coords, true_features):
    """
    Compute the 1-NN (nearest neighbor) mean squared error (MSE) between predicted features
    and the nearest true features based on spatial coordinates.

    Parameters:
    pred_coords (torch.Tensor): Tensor of shape (N, D) containing the coordinates of the predicted points.
    pred_features (torch.Tensor): Tensor of shape (N, F) containing the features of the predicted points.
    true_coords (torch.Tensor): Tensor of shape (M, D) containing the coordinates of the true points.
    true_features (torch.Tensor): Tensor of shape (M, F) containing the features of the true points.

    Returns:
    torch.Tensor: The computed MSE between the predicted features and the nearest true features.
    """
    # Compute the pairwise Euclidean distances between each predicted and true point (N, M)
    distances = torch.cdist(pred_coords, true_coords)

    # For each predicted point, find the index of the nearest true point (N,)
    nearest_indices = torch.argmin(distances, dim=1)

    # Retrieve the features of the nearest true points
    nearest_features = true_features[nearest_indices]

    # Calculate the mean squared error (MSE) between the predicted features and the nearest true features
    mse = torch.mean((pred_features - nearest_features) ** 2)

    return mse


def SinkhornLoss(true_obs, est_obs, blur=0.05, scaling=0.5):
    '''
    Wasserstein distance computed by Sinkhorn algorithm.
    :param true_obs (torch.FloatTensor): True expression data.
    :param est_obs (torch.FloatTensor): Predicted expression data.
    :param blur (float): Sinkhorn algorithm hyperparameter. Default as 0.05.
    :param scaling (float): Sinkhorn algorithm hyperparameter. Default as 0.5.
    :param batch_size (None or int): Either None indicates using all true cell in computation, or an integer indicates
                                     using only a batch of true cells to save computational costs.
    :return: (float) Wasserstein distance.
    '''
    ot_solver = geomloss.SamplesLoss("sinkhorn", p=2, blur=blur, scaling=scaling, debias=True, backend="tensorized")
    loss = ot_solver(true_obs, est_obs)
    return loss



def one_nn_mse(pred_coords, pred_features, true_coords, true_features):
    """
    Compute the 1-NN (nearest neighbor) mean squared error (MSE) between predicted features
    and the nearest true features based on spatial coordinates.

    Parameters:
    pred_coords (torch.Tensor): Tensor of shape (N, D) containing the coordinates of the predicted points.
    pred_features (torch.Tensor): Tensor of shape (N, F) containing the features of the predicted points.
    true_coords (torch.Tensor): Tensor of shape (M, D) containing the coordinates of the true points.
    true_features (torch.Tensor): Tensor of shape (M, F) containing the features of the true points.

    Returns:
    torch.Tensor: The computed MSE between the predicted features and the nearest true features.
    """
    # Compute the pairwise Euclidean distances between each predicted and true point (N, M)
    distances = torch.cdist(pred_coords, true_coords)

    # For each predicted point, find the index of the nearest true point (N,)
    nearest_indices = torch.argmin(distances, dim=1)

    # Retrieve the features of the nearest true points
    nearest_features = true_features[nearest_indices]

    # Calculate the mean squared error (MSE) between the predicted features and the nearest true features
    mse = torch.mean((pred_features - nearest_features) ** 2)

    return mse


def chamfer_distance(pred_coords, template_coords, batch_size=1024, sample_size=None):
    template_coords
    device = pred_coords.device

    # Shuffle the template coordinates
    perm = torch.randperm(template_coords.size(0))
    template_coords = template_coords[perm]

    # If sample_size is specified, sample the template coordinates
    if sample_size is not None and sample_size < template_coords.size(0):
        template_coords = template_coords[:sample_size]

    # Compute the minimum distance from each predicted point to the template
    min_distances_pred_to_true = []
    N_pred = pred_coords.size(0)
    for start in range(0, N_pred, batch_size):
        end = min(start + batch_size, N_pred)
        pred_batch = pred_coords[start:end]
        distances = torch.cdist(pred_batch, template_coords)
        min_distances, _ = distances.min(dim=1)
        min_distances_pred_to_true.append(min_distances)
    min_distances_pred_to_true = torch.cat(min_distances_pred_to_true)
    mse_pred_to_true = torch.mean(min_distances_pred_to_true ** 2)

    # Compute the minimum distance from each template point to the predicted points
    min_distances_true_to_pred = []
    N_template = template_coords.size(0)
    for start in range(0, N_template, batch_size):
        end = min(start + batch_size, N_template)
        template_batch = template_coords[start:end]
        distances = torch.cdist(template_batch, pred_coords)
        min_distances, _ = distances.min(dim=1)
        min_distances_true_to_pred.append(min_distances)
    min_distances_true_to_pred = torch.cat(min_distances_true_to_pred)
    mse_true_to_pred = torch.mean(min_distances_true_to_pred ** 2)

    # Compute the Chamfer distance
    chamfer_dist = mse_pred_to_true + mse_true_to_pred

    return chamfer_dist


def one_nn_mse(pred_coords, pred_features, true_coords, true_features):
    """
    Compute the 1-NN (nearest neighbor) mean squared error (MSE) between predicted features
    and the nearest true features based on spatial coordinates.

    Parameters:
    pred_coords (torch.Tensor): Tensor of shape (N, D) containing the coordinates of the predicted points.
    pred_features (torch.Tensor): Tensor of shape (N, F) containing the features of the predicted points.
    true_coords (torch.Tensor): Tensor of shape (M, D) containing the coordinates of the true points.
    true_features (torch.Tensor): Tensor of shape (M, F) containing the features of the true points.

    Returns:
    torch.Tensor: The computed MSE between the predicted features and the nearest true features.
    """
    # Compute the pairwise Euclidean distances between each predicted and true point (N, M)
    distances = torch.cdist(pred_coords, true_coords)

    # For each predicted point, find the index of the nearest true point (N,)
    nearest_indices = torch.argmin(distances, dim=1)

    # Retrieve the features of the nearest true points
    nearest_features = true_features[nearest_indices]

    # Calculate the mean squared error (MSE) between the predicted features and the nearest true features
    mse = torch.mean((pred_features - nearest_features) ** 2)

    return mse