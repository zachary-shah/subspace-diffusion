"""
Code for sampling the Diffusion Model
"""

import gc, io, os, random, time, logging, wandb, torch
import numpy as np
import tensorflow as tf
from models import ddpm, ncsnpp
import losses, sampling, datasets, likelihood, sde_lib
from models import utils as mutils
from models.ema import ExponentialMovingAverage
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from utils import restore_checkpoint
from upsampling import upsampling_fn
from absl import app, flags
from ml_collections.config_flags import config_flags
from configs.ve.cifar10_ncsnpp_continuous import get_config as get_cifar_config
from time import perf_counter

"""
Helper functions
"""
def get_config(number_samples, subspace_dim):
    """
    Helper to set up config for CIFAR-10
    """
    config = get_cifar_config()
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # default parameters
    config.eval.batch_size = number_samples
    config.langevin_steps = 2
    config.langevin_snr = 0.22
    config.conditional_langevin = True
    config.sampling_eps = 1e-5

    # Model paths
    if subspace_dim == 8:
        config.ckpt_subspace = './pretrained/cifar10_ncsn_8x8.pth'
    elif subspace_dim == 16:
        config.ckpt_subspace = './pretrained/cifar10_ncsn_16x16.pth'
    else:
        raise NotImplementedError("Subspace count must be either 8 or 16.")
    config.ckpt_full: str = './pretrained/cifar10_ncsn_full.pth'
    
    # Create data normalizer and its inverse
    config.scaler = datasets.get_data_scaler(config)
    config.inverse_scaler = datasets.get_data_inverse_scaler(config)
    return config

def postprocess_samples(samples, config):
    """
    Helper to postprocess samples as np arrays with [0,255] scaling and shape (number_samples, H, W, 3)
    """
    samples = samples.permute(0, 2, 3, 1).cpu().numpy() 
    samples = np.clip(samples * 255., 0, 255).astype(np.uint8)
    samples = samples.reshape((-1, config.data.image_size, config.data.image_size, config.data.num_channels))
    return samples

def instantiate_models(path, image_size, config, isLast):
    """
    Helper to gather the score model, langevin corrector, and sampling function given the model path, the 
    image size to sample, and the config. Additionally, on the last diffusion model, we need additional denoising. 
    Returns a dict containing the relevant pieces for diffusion sampling (score_model, langevin_fn, sampling_fn).
    """
    models = {}
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    config.data.image_size = image_size
    models['score_model'] = score_model = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
    models['state'] = restore_checkpoint(path, state, device=config.device)
    ema.copy_to(score_model.parameters())

    # Noise Corrector using langevin dynamics
    models['langevin_fn'] = sampling.LangevinCorrector(sde, score_model, config.langevin_snr,
                                                                        config.langevin_steps).update_fn  
    
    # function which samples the SDE with noise
    sampling_shape = (config.eval.batch_size, config.data.num_channels,
                        config.data.image_size, config.data.image_size)
    models['sampling_fn'] = sampling.get_sampling_fn(config, sde, sampling_shape, config.inverse_scaler,
                                                                     config.sampling_eps, denoise=isLast)
    
    return models


"""
Main sampling function for a-d
"""
def sample_diffusion(
        number_samples: int = 10,
        prop_subspace: float = 0.0,
        subspace_dim: int = 8,        
    ):  

    """
    Sample a diffusion model with subspace compression. 
    
    Parameters
    ---------
    - number_samples : int
        Number of samples to query.
    - subspace : int
        Dimension of the subspace model. Must be either 8 or 16 for this assignment.
        Default = 8.
    - prop_subspace : float
        Proportion of time diffusing in subspace. Must be float between (0,1).
        Default = 0.0, meaning no subspace sampling.
    - batch : int
        Sampling batch size

    Returns
    --------
    - samples : np.ndarray
        Samples from model, in shape (number_samples, 32, 32, 3)
    - inference_time : float
        Time for model inference
    """

    # Get config for CIFAR-10
    config = get_config(number_samples, subspace_dim)    

    # Define the sampling steps
    t_subspace_switch = 1.0 - prop_subspace
    steps = [
        {'size': subspace_dim, 'pth': config.ckpt_subspace,
         'start': 1., 'end': t_subspace_switch},
        {'size': config.data.image_size, 'pth': config.ckpt_full,
         'start': t_subspace_switch, 'end': 0.},
    ]

    # handle edge case where there is no subspace switch
    steps = [step for step in steps if step['start'] != step['end']]

    # Get the score model, optimizer, pretrained weights, and sampling functions for the full and subspace models
    print(f"Preparing models...")
    for i, step in enumerate(steps):
        step["mdls"] = instantiate_models(step['pth'], step['size'], config, (i == len(steps) - 1))
    
    print("Sampling...")
    start_time = perf_counter()
    for i, step in enumerate(steps):
        
        # Unpack parameters
        config.data.image_size = size = step['size']
        start, end = step['start'], step['end']
        sampling_fn = step["mdls"]['sampling_fn']
        score_model = step["mdls"]['score_model']
        langevin_fn = step["mdls"]['langevin_fn']
        
        # Correction after initial sampling using langevin dynamics
        if i != 0 and config.langevin_steps > 0:
            with torch.no_grad() :
                remove_subspace = int(np.log2(size/steps[i-1]['size']))
                samples, _ = langevin_fn(samples, t_vec, remove_subspace=remove_subspace)

        # Sample the diffusion model. The first sample does not have a starting point, so x=None
        if i == 0:
            samples, t_vec = sampling_fn(score_model, start=start, end=end)
            samples = config.scaler(samples)
        else:
            samples, t_vec = sampling_fn(score_model, x=samples, start=start, end=end)

        # Upsample image if going from subspace to full model
        if i != len(steps) - 1:
            t = end
            sigma_min, sigma_max = config.model.sigma_min, config.model.sigma_max
            sigma = sigma_min * (sigma_max / sigma_min) ** t
            alpha = 1
            samples = upsampling_fn(samples, alpha=alpha, sigma=sigma, dataset='cifar')

    samples = postprocess_samples(samples, config)
    
    end_time = perf_counter()
    inference_time = end_time - start_time
        
    del steps, step, score_model, sampling_fn
    torch.cuda.empty_cache()

    return samples, inference_time

"""
Helper function for Part (e): Upsampling with Diffusion Model
"""
def diffusion_upsample(
        seeds: np.ndarray,
        iters: int = 100,
        alpha: float = 0.0,
    ):

    """
    Upsample seed images with a diffusion model.

    Parameters
    ---------
    - seeds : np.ndarray
        Starting images for diffusive sampling. In shape (number_samples, 3, subpsace_dim, subspace_dim)
    - iters : int
        Number of iterations to run diffusive upsampling for. Must be between [0,1000]
    - alpha : float
        Noise scale to perturb input data to (higher noise results in higher deviations from seed)

    Returns
    --------
    - samples : np.ndarray
        Samples from model, in shape (number_samples, 32, 32, 3)
    """

    # Get config for CIFAR-10
    number_samples, subspace_dim = seeds.shape[0], seeds.shape[-1]
    config = get_config(number_samples, subspace_dim)

    # Get the score model, optimizer, pretrained weights, and sampling functions for the full and subspace models
    print(f"Preparing models...")
    mdls = instantiate_models(config.ckpt_full, config.data.image_size, config, True)

    print("Sampling...")
    start, end = iters / 1000 , 0
    sampling_fn, score_model = mdls['sampling_fn'], mdls['score_model']

    seeds = torch.from_numpy(seeds).type(torch.float32).to(config.device)

    # Upsample image to go to full model
    seeds = upsampling_fn(seeds, alpha=alpha, sigma=0, dataset='cifar')

    # Sample the diffusion model
    samples, t_vec = sampling_fn(score_model, x=seeds, start=start, end=end)

    samples = postprocess_samples(samples, config)
    samples = samples * int(config.data.image_size/subspace_dim)

    del score_model, sampling_fn
    torch.cuda.empty_cache()

    return samples
