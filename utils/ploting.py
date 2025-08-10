import pandas as pd
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

def plot_multi_gene_3d(data_df, gene_roi, noise_std=0.01,
                       offset=0.0, threathold=0,
                       opacity=0.5, size=0.5, cmin=0.2, cmax=0.8,
                       gene_colorscales=None, custom_colorscales_dict=None, depth_list=None):
    """
    Plot multiple genes simultaneously in 3D space using scatter plots.

    This function normalizes gene expression values, adds optional coordinate noise for visualization,
    and displays the spatial distribution of multiple genes. Each gene is assigned a separate colorscale,
    with an option to display either a global bounding box or cross-sectional rectangles at specified depths.

    Parameters
    ----------
    data_df : pandas.DataFrame
        A DataFrame containing at least the following columns:
        - 'x', 'y', 'z': spatial coordinates.
        - One column for each gene in `gene_roi`, containing expression values.
    gene_roi : list of str
        List of gene names (column names in `data_df`) to be plotted.
    noise_std : float, optional, default=0.01
        Standard deviation of Gaussian noise added to coordinates (to reduce visual overlap).
    offset : float, optional, default=0.0
        Base x-axis offset applied per gene. The i-th gene is shifted by `i * offset`.
    threathold : float, optional, default=0
        Minimum expression value for a point to be displayed.
    opacity : float, optional, default=0.5
        Marker transparency (0 = fully transparent, 1 = fully opaque).
    size : float, optional, default=0.5
        Marker size.
    cmin : float, optional, default=0.2
        Minimum color scale value.
    cmax : float, optional, default=0.8
        Maximum color scale value.
    gene_colorscales : list of list, optional
        List of Plotly-compatible colorscales for each gene.
        Each colorscale is a list of [position, color] pairs.
        If fewer colorscales than genes are provided, they are repeated cyclically.
        If None, `custom_colorscales_dict` or default colorscales are used.
    custom_colorscales_dict : dict, optional
        Dictionary mapping gene names to their corresponding Plotly colorscales.
    depth_list : list of float, optional
        If None, a full 3D bounding box is drawn.
        If provided, draws rectangular cross-sections at the specified depths along the z-axis.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        A Plotly 3D scatter plot figure showing the spatial distribution of the specified genes.
    """

    from copy import deepcopy
    import numpy as np
    import plotly.graph_objects as go

    # Normalize gene expression values across the selected genes
    data_df[gene_roi] = scaler.fit_transform(data_df[gene_roi])

    # Add small Gaussian noise to spatial coordinates to reduce overlap
    mean = 0
    data_df['x'] += np.random.normal(loc=mean, scale=noise_std, size=data_df['x'].values.shape)
    data_df['y'] += np.random.normal(loc=mean, scale=noise_std, size=data_df['y'].values.shape)
    data_df['z'] += np.random.normal(loc=mean, scale=noise_std, size=data_df['z'].values.shape)

    # Combine coordinates and gene expression into a single NumPy array
    data = data_df[['x', 'y', 'z'] + gene_roi].values
    data_all = deepcopy(data)
    coords = data_all[:, :3]
    gene_values = data_all[:, 3:]
    M = gene_values.shape[1]

    # Default colorscales for genes (white to various colors)
    default_colorscales = [
        [[0, 'white'], [1, '#FF0000']],  # Red
        [[0, 'white'], [1, '#00FF00']],  # Green
        [[0, 'white'], [1, '#0000FF']],  # Blue
        [[0, 'white'], [1, '#800080']],  # Purple
        [[0, 'white'], [1, '#FFA500']],  # Orange
        [[0, 'white'], [1, '#FFFF00']],  # Yellow
        [[0, 'white'], [1, '#00FFFF']],  # Cyan
        [[0, 'white'], [1, '#FF00FF']],  # Magenta
        [[0, 'white'], [1, '#FF69B4']],  # Pink
        [[0, 'white'], [1, '#00FF7F']],  # Spring Green
        [[0, 'white'], [1, '#FFD700']],  # Gold
        [[0, 'white'], [1, '#ADFF2F']],  # Green Yellow
        [[0, 'white'], [1, '#7B68EE']],  # Medium Slate Blue
        [[0, 'white'], [1, '#FF4500']],  # Orange Red
        [[0, 'white'], [1, '#DA70D6']],  # Orchid
    ]

    # Determine colorscales for each gene
    if custom_colorscales_dict is None:
        gene_colorscales = [default_colorscales[i % len(default_colorscales)] for i in range(M)]
    else:
        gene_colorscales = [custom_colorscales_dict[g] for g in gene_roi]

    # Identify the gene with the highest expression at each point
    max_gene_indices = np.argmax(gene_values, axis=1)
    fig = go.Figure()

    # Set x-axis positions for colorbars, spaced evenly across the figure
    colorbar_x_positions = np.linspace(0.05, 0.95, M)
    
    # Add a 3D scatter trace for each gene
    for i in range(M):
        # Filter points where this gene has the highest expression
        mask = (max_gene_indices == i)
        data_gene_1 = data_all[mask]
        over_t = data_gene_1[:, 3 + i] > threathold
        data_gene = data_gene_1[over_t]

        if data_gene.shape[0] == 0:
            continue  # Skip if no points remain after filtering

        colorbar_x = colorbar_x_positions[i]
        colorbar_y = -0.2  # Position colorbars below the plot

        # Add 3D scatter plot for this gene
        fig.add_trace(go.Scatter3d(
            x=data_gene[:, 0],
            y=data_gene[:, 1],
            z=data_gene[:, 2],
            mode='markers',
            marker=dict(
                size=size,
                color=data_gene[:, 3 + i],
                colorscale=gene_colorscales[i],
                colorbar=dict(
                    title=dict(
                        text=gene_roi[i],
                        font=dict(color='white'),
                        side='top'
                    ),
                    x=colorbar_x,
                    y=colorbar_y,
                    len=0.2,
                    thickness=10,
                    bgcolor='#080808',
                    tickvals=[],
                ),
                showscale=True,
                opacity=opacity,
                cmin=cmin,
                cmax=cmax,
            ),
            name=gene_roi[i]
        ))

    # Calculate spatial bounds for bounding box or cross-sections
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    z_min, z_max = coords[:, 2].min(), coords[:, 2].max()

    # Add padding to avoid tight edges
    padding_x = (x_max - x_min) * 0.01
    padding_y = (y_max - y_min) * 0.01
    padding_z = (z_max - z_min) * 0.01

    x_min_padded = x_min - padding_x
    x_max_padded = x_max + padding_x
    y_min_padded = y_min - padding_y
    y_max_padded = y_max + padding_y
    z_min_padded = z_min - padding_z
    z_max_padded = z_max + padding_z

    # Draw bounding box or depth-specific cross-sections
    if depth_list is None:
        # Bounding box corner coordinates
        box_corners = np.array([
            [x_min_padded, y_min_padded, z_min_padded],
            [x_max_padded, y_min_padded, z_min_padded],
            [x_max_padded, y_max_padded, z_min_padded],
            [x_min_padded, y_max_padded, z_min_padded],
            [x_min_padded, y_min_padded, z_max_padded],
            [x_max_padded, y_min_padded, z_max_padded],
            [x_max_padded, y_max_padded, z_max_padded],
            [x_min_padded, y_max_padded, z_max_padded],
        ])

        # Edges of the bounding box
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]

        # Prepare line coordinates
        box_x, box_y, box_z = [], [], []
        for start, end in edges:
            box_x += [box_corners[start][0], box_corners[end][0], None]
            box_y += [box_corners[start][1], box_corners[end][1], None]
            box_z += [box_corners[start][2], box_corners[end][2], None]

        fig.add_trace(go.Scatter3d(
            x=box_x, y=box_y, z=box_z,
            mode='lines',
            line=dict(color='white', width=2),
            name='Bounding Box'
        ))

    else:
        # Draw XY-rectangle frames at each specified Z depth
        for d in depth_list:
            rect_corners = np.array([
                [x_min_padded, y_min_padded, d],
                [x_min_padded, y_max_padded, d],
                [x_max_padded, y_max_padded, d],
                [x_max_padded, y_min_padded, d],
            ])

            rect_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

            rect_x, rect_y, rect_z = [], [], []
            for start, end in rect_edges:
                rect_x += [rect_corners[start][0], rect_corners[end][0], None]
                rect_y += [rect_corners[start][1], rect_corners[end][1], None]
                rect_z += [rect_corners[start][2], rect_corners[end][2], None]

            fig.add_trace(go.Scatter3d(
                x=rect_x, y=rect_y, z=rect_z,
                mode='lines',
                line=dict(color='white', width=2),
                name=f'Rect @ z={d}'
            ))

    # Layout and style settings
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=50),
        scene=dict(
            aspectmode='data',
            bgcolor='#080808',
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            xaxis=dict(color='white'),
            yaxis=dict(color='white'),
            zaxis=dict(color='white'),
        ),
        paper_bgcolor='#080808',
        legend=dict(
            title='Label',
            font=dict(color='white'),
            bgcolor='#080808'
        ),
        font=dict(color='white')
    )

    return fig
