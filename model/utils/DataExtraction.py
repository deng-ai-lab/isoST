import torch

    
def extract_data_values(data, downrate, coords, direction, spacing, origin, scale, padding_value=None):
    """
    Extract data values from a 3D data tensor at specified coordinates.

    Args:
        data (torch.Tensor): The 3D data tensor of shape (D, H, W).
        coords (torch.Tensor): The coordinates tensor of shape (N, 3), in mm.
        direction (torch.Tensor): The direction matrix of shape (3, 3).
        spacing (torch.Tensor): The spacing vector of shape (3,), in mm.
        origin (torch.Tensor): The origin vector of shape (3,), in mm.
        scale (torch.Tensor): The scale vector of shape (3,).
        padding_value (float, optional): The value to use for coordinates outside the valid range.
                                         Defaults to data[0, 0, 0] if not specified.

    Returns:
        data_values (torch.Tensor): The data values at the specified coordinates, shape (N,).
    """

    # Set default padding_value if not specified
    dim = data.shape[-1]
    device = data.device
    coords = coords.to(device)
    direction = direction.to(device)
    spacing = spacing.to(device)

    if padding_value is None:
        padding_value = data[0, 0, 0].item()  # Convert tensor to Python float

    physical_coords = coords * scale + origin  # Convert mm coordinates to physical coordinates
    # Compute indices
    indices = torch.matmul(physical_coords, direction.T)  # Shape: (N, 3)
    indices_scaled = indices / spacing  # Scale coordinates to voxel space
    indices_int = torch.round(indices_scaled / downrate).long()  # Convert to integer indices

    # Create valid mask
    valid_mask = (
        (indices_int[:, 0] >= 0) & (indices_int[:, 0] < data.shape[0]) &
        (indices_int[:, 1] >= 0) & (indices_int[:, 1] < data.shape[1]) &
        (indices_int[:, 2] >= 0) & (indices_int[:, 2] < data.shape[2])
    )

    # Initialize data_values with padding_value
    data_values = torch.full((coords.shape[0], dim), padding_value, dtype=data.dtype, device=device)

    # For valid indices, extract data values
    valid_indices_int = indices_int[valid_mask]
    data_values_valid = data[valid_indices_int[:, 0], valid_indices_int[:, 1], valid_indices_int[:, 2]]
    
    # Assign the valid data values into the data_values tensor
    data_values[valid_mask] = data_values_valid
    
    return data_values.reshape(-1, dim)