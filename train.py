"""
To launch all the tasks, create tmux sessions (separately for each of the following) 
and run (for instance):

python canvi_sbibm.py --task two_moons --cuda_idx 0
python canvi_sbibm.py --task slcp --cuda_idx 1
python canvi_sbibm.py --task gaussian_linear_uniform --cuda_idx 2
python canvi_sbibm.py --task bernoulli_glm --cuda_idx 3
python canvi_sbibm.py --task gaussian_mixture --cuda_idx 4
python canvi_sbibm.py --task gaussian_linear --cuda_idx 5
python canvi_sbibm.py --task slcp_distractors --cuda_idx 6
python canvi_sbibm.py --task bernoulli_glm_raw --cuda_idx 7
"""

import pandas as pd
import numpy as np
import sbibm
import torch
import math
import torch.distributions as D
import matplotlib.pyplot as plt

from pyknos.nflows import flows, transforms
from functools import partial
from typing import Optional
from warnings import warn

from pyknos.nflows import distributions as distributions_
from pyknos.nflows import flows, transforms
from pyknos.nflows.nn import nets
from pyknos.nflows.transforms.splines import rational_quadratic
from torch import Tensor, nn, relu, tanh, tensor, uint8

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['text.usetex'] = True
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'

sns.set_theme()

from sbi.utils.sbiutils import (
    standardizing_net,
    standardizing_transform,
    z_score_parser,
)
from sbi.utils.torchutils import create_alternating_binary_mask
from sbi.utils.user_input_checks import check_data_device, check_embedding_net_device

import os
import pickle
import argparse


class ContextSplineMap(nn.Module):
    """
    Neural network from `context` to the spline parameters.
    We cannot use the resnet as conditioner to learn each dimension conditioned
    on the other dimensions (because there is only one). Instead, we learn the
    spline parameters directly. In the case of conditinal density estimation,
    we make the spline parameters conditional on the context. This is
    implemented in this class.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        context_features: int,
        hidden_layers: int,
    ):
        """
        Initialize neural network that learns to predict spline parameters.
        Args:
            in_features: Unused since there is no `conditioner` in 1D.
            out_features: Number of spline parameters.
            hidden_features: Number of hidden units.
            context_features: Number of context features.
        """
        super().__init__()
        # `self.hidden_features` is only defined such that nflows can infer
        # a scaling factor for initializations.
        self.hidden_features = hidden_features

        # Use a non-linearity because otherwise, there will be a linear
        # mapping from context features onto distribution parameters.

        # Initialize with input layer.
        layer_list = [nn.Linear(context_features, hidden_features), nn.ReLU()]
        # Add hidden layers.
        layer_list += [
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
        ] * hidden_layers
        # Add output layer.
        layer_list += [nn.Linear(hidden_features, out_features)]
        self.spline_predictor = nn.Sequential(*layer_list)

    def __call__(self, inputs: Tensor, context: Tensor, *args, **kwargs) -> Tensor:
        """
        Return parameters of the spline given the context.
        Args:
            inputs: Unused. It would usually be the other dimensions, but in
                1D, there are no other dimensions.
            context: Context features.
        Returns:
            Spline parameters.
        """
        return self.spline_predictor(context)

# Declan: this code from SBI library
def build_nsf(
    batch_x: Tensor,
    batch_y: Tensor,
    z_score_x: Optional[str] = "independent",
    z_score_y: Optional[str] = "independent",
    hidden_features: int = 50,
    num_transforms: int = 5,
    num_bins: int = 10,
    embedding_net: nn.Module = nn.Identity(),
    tail_bound: float = 3.0,
    hidden_layers_spline_context: int = 1,
    num_blocks: int = 2,
    dropout_probability: float = 0.0,
    use_batch_norm: bool = False,
    **kwargs,
) -> nn.Module:
    """Builds NSF p(x|y).
    Args:
        batch_x: Batch of xs, used to infer dimensionality and (optional) z-scoring.
        batch_y: Batch of ys, used to infer dimensionality and (optional) z-scoring.
        z_score_x: Whether to z-score xs passing into the network, can be one of:
            - `none`, or None: do not z-score.
            - `independent`: z-score each dimension independently.
            - `structured`: treat dimensions as related, therefore compute mean and std
            over the entire batch, instead of per-dimension. Should be used when each
            sample is, for example, a time series or an image.
        z_score_y: Whether to z-score ys passing into the network, same options as
            z_score_x.
        hidden_features: Number of hidden features.
        num_transforms: Number of transforms.
        num_bins: Number of bins used for the splines.
        embedding_net: Optional embedding network for y.
        tail_bound: tail bound for each spline.
        hidden_layers_spline_context: number of hidden layers of the spline context net
            for one-dimensional x.
        num_blocks: number of blocks used for residual net for context embedding.
        dropout_probability: dropout probability for regularization in residual net.
        use_batch_norm: whether to use batch norm in residual net.
        kwargs: Additional arguments that are passed by the build function but are not
            relevant for maf and are therefore ignored.
    Returns:
        Neural network.
    """
    x_numel = batch_x[0].numel()
    # Infer the output dimensionality of the embedding_net by making a forward pass.
    check_data_device(batch_x, batch_y)
    check_embedding_net_device(embedding_net=embedding_net, datum=batch_y)
    y_numel = embedding_net(batch_y[:1]).numel()

    # Define mask function to alternate between predicted x-dimensions.
    def mask_in_layer(i):
        return create_alternating_binary_mask(features=x_numel, even=(i % 2 == 0))

    # If x is just a scalar then use a dummy mask and learn spline parameters using the
    # conditioning variables only.
    if x_numel == 1:
        # Conditioner ignores the data and uses the conditioning variables only.
        conditioner = partial(
            ContextSplineMap,
            hidden_features=hidden_features,
            context_features=y_numel,
            hidden_layers=hidden_layers_spline_context,
        )
    else:
        # Use conditional resnet as spline conditioner.
        conditioner = partial(
            nets.ResidualNet,
            hidden_features=hidden_features,
            context_features=y_numel,
            num_blocks=num_blocks,
            activation=relu,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )

    # Stack spline transforms.
    transform_list = []
    for i in range(num_transforms):
        block = [
            transforms.PiecewiseRationalQuadraticCouplingTransform(
                mask=mask_in_layer(i) if x_numel > 1 else tensor([1], dtype=uint8),
                transform_net_create_fn=conditioner,
                num_bins=num_bins,
                tails="linear",
                tail_bound=tail_bound,
                apply_unconditional_transform=False,
            )
        ]
        # Add LU transform only for high D x. Permutation makes sense only for more than
        # one feature.
        if x_numel > 1:
            block.append(
                transforms.LULinear(x_numel, identity_init=True),
            )
        transform_list += block

    z_score_x_bool, structured_x = z_score_parser(z_score_x)
    if z_score_x_bool:
        # Prepend standardizing transform to nsf transforms.
        transform_list = [
            standardizing_transform(batch_x, structured_x)
        ] + transform_list

    z_score_y_bool, structured_y = z_score_parser(z_score_y)
    if z_score_y_bool:
        # Prepend standardizing transform to y-embedding.
        embedding_net = nn.Sequential(
            standardizing_net(batch_y, structured_y), embedding_net
        )

    distribution = distributions_.StandardNormal((x_numel,))

    # Combine transforms.
    transform = transforms.CompositeTransform(transform_list)
    neural_net = flows.Flow(transform, distribution, embedding_net)

    return neural_net


class EmbeddingNet(nn.Module):
    def __init__(self, dim):
        super(EmbeddingNet, self).__init__()
        self.context_dim = dim
        self.dense = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )


    def forward(self, x):
        '''
        Assumes context x is of shape (batch_size, self.context_dim)
        '''
        return self.dense(x)

def generate_data(priors, simulators, n_pts, return_theta=False, combination_method=None):
    thetas, xs, view_dims = [], [], []
    for prior, simulator in zip(priors, simulators):
        theta = prior(num_samples=n_pts)
        x = simulator(theta)

        thetas.append(theta)
        xs.append(x)
        view_dims.append(x.shape[-1])

    if combination_method is None or combination_method == "stack":
        theta, x = torch.hstack(thetas), torch.hstack(xs)
    elif combination_method == "sum":
        # thetas[1] /= 10
        theta, x = (thetas[0][:,:2] + thetas[1][:,:2]), torch.hstack(xs)
    elif combination_method == "repeat":
        theta, x = torch.tile(theta, (1,1)), torch.hstack(xs)
        
    proj_dim = 2 # to consider a projected, lower-dimensional version of the problem
    if proj_dim is not None:
        theta = theta[:,:proj_dim]

    if return_theta: 
        return theta, x, view_dims
    else:
        return x

def ci_len(encoder, q_hat, theta_grid, test_X_grid, test_sims, discretization):
    grid_scores = 1 / encoder.log_prob(theta_grid, test_X_grid).detach().cpu().exp().numpy()
    grid_scores = grid_scores.reshape(test_sims, -1) # reshape back to 2D grid per-trial

    # hacky solution to vectorize this computation, but hey, I like it
    confidence_mask = np.zeros(grid_scores.shape)
    confidence_mask[grid_scores < q_hat] = discretization
    interval_lengths = np.sum(confidence_mask, axis=1)
    return np.mean(interval_lengths)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks")
    parser.add_argument("--cuda_idx")
    parser.add_argument("--view_idx")
    parser.add_argument("--start_dim")
    parser.add_argument("--end_dim")
    args = parser.parse_args()

    task_names = args.tasks.split(",")
    tasks      = [sbibm.get_task(task_name) for task_name in task_names]
    priors     = [task.get_prior() for task in tasks]
    simulators = [task.get_simulator() for task in tasks]

    combination_method = "repeat"
    setup_theta, setup_x, x_dims = generate_data(priors, simulators, 100, return_theta=True, combination_method=combination_method) 
    
    if args.start_dim is not None and args.end_dim is not None:
        start_dim, end_dim = int(args.start_dim), int(args.end_dim)
    else:
        view_idx = int(args.view_idx)
        start_dim, end_dim = int(np.sum(x_dims[:view_idx])), int(np.sum(x_dims[:view_idx+1]))

    setup_x = setup_x[:,start_dim:end_dim]
    mb_size = 50
    device = f"cuda:0"

    # EXAMPLE BATCH FOR SHAPES
    z_dim = setup_theta.shape[-1]
    x_dim = setup_x.shape[-1]
    num_obs_flow = mb_size
    fake_zs = torch.randn((mb_size, z_dim))
    fake_xs = torch.randn((mb_size, x_dim))
    encoder = build_nsf(fake_zs, fake_xs, z_score_x='none', z_score_y='none')

    encoder.to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    
    fn = f"{args.tasks}_{start_dim}-{end_dim}.nf"
    save_iterate = 1_000
    for j in range(5_001):
        print(f"Training step: {j}")
        theta, x, _ = generate_data(priors, simulators, mb_size, return_theta=True, combination_method=combination_method)
        x = x[:,start_dim:end_dim]
        optimizer.zero_grad()
        loss = -1 * encoder.log_prob(theta.to(device), x.to(device)).mean()
        loss.backward()
        optimizer.step()

        if j % save_iterate == 0:    
            cached_fn = os.path.join("trained", fn)
            with open(cached_fn, "wb") as f:
                pickle.dump(encoder, f)

    encoder.to("cpu")
    with open(os.path.join("trained_cpu", fn), "wb") as f:
        pickle.dump(encoder, f)