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
from configs.ve.cifar10_ncsnpp_continuous import get_config

#import evaluation

def _get_def_flag():
    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file(
        "config", 'configs/ve/cifar10_ncsnpp_continuous.py', "Training configuration.", lock_config=True)
    return FLAGS

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        pass

def sample(
        eval_folder: str = None,
        save_name: str = None,
        number_samples: int = 10,
        batch: int = 5,
        langevin_steps: int = 2,
        conditional_langevin: bool = True,
        langevin_snr: float = 0.22,
        time: float = 0.5,
        subspace: int = 16,
        ckpt_subspace: str = None,
        ckpt_full: str = None,
    ):  

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = get_config()
    config.eval.batch_size = batch
    
    eval_dir = os.path.join(eval_folder)
    os.makedirs(eval_dir, exist_ok=True)

    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    ###################################
    # Setup SDEs
    
    if config.training.sde == "vesde":
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    elif config.training.sde == "subvpsde":
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-5

    steps = [
        {'size': subspace, 'pth': ckpt_subspace,
         'start': 1., 'end': time},
        {'size': config.data.image_size, 'pth': ckpt_full,
         'start': time, 'end': 0.},
    ]

    steps = [step for step in steps if step['start'] != step['end']]

    print(f"Preparing models...")
    for i, step in enumerate(steps):

        config.data.image_size = step['size']
        step['score_model'] = score_model = mutils.create_model(config)
        step['optimizer'] = optimizer = losses.get_optimizer(config, score_model.parameters())
        step['ema'] = ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
        state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
        step['state'] = restore_checkpoint(step['pth'], state, device=config.device)
        ema.copy_to(score_model.parameters())
        
        step['langevin_fn'] = langevin_fn = sampling.LangevinCorrector(sde, score_model, langevin_snr,
                                                                           langevin_steps).update_fn  

        sampling_shape = (config.eval.batch_size,
                          config.data.num_channels,
                          config.data.image_size, config.data.image_size)
        
        denoise = (i == len(steps) - 1)
        step['sampling_fn'] = sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler,
                                                                     sampling_eps, denoise=denoise)

    all_samples = []
    num_sampling_rounds = (number_samples - 1) // config.eval.batch_size + 1
    
    print(f"Sampling {num_sampling_rounds}...")
    for r in range(num_sampling_rounds):    

        for i, step in enumerate(steps):
            
            config.data.image_size = size = step['size']
            start, end = step['start'], step['end']
            sampling_fn, score_model = step['sampling_fn'].to(device), step['score_model'].to(device)
            langevin_fn = step['langevin_fn']
            
            if i != 0 and langevin_steps > 0:
                with torch.no_grad() :
                    if conditional_langevin:
                        remove_subspace = int(np.log2(size/steps[i-1]['size']))
                        samples, _ = langevin_fn(samples, t_vec, remove_subspace=remove_subspace)
                    else:
                        samples, _ = langevin_fn(samples, t_vec)
            
            if i == 0:
                samples, t_vec = sampling_fn(score_model, start=start, end=end)
                samples = scaler(samples)
               
            else:
                samples, t_vec = sampling_fn(score_model, x=samples, start=start, end=end)

            if i != len(steps) - 1:
                t = end
                if config.training.sde == 'vesde':
                    sigma_min, sigma_max = config.model.sigma_min, config.model.sigma_max
                    sigma = sigma_min * (sigma_max / sigma_min) ** t
                    alpha = 1
                    samples = upsampling_fn(samples, alpha=alpha, sigma=sigma, dataset='cifar')
                
                elif config.training.sde == 'subvpsde':
                    beta_min, beta_max = config.model.beta_min, config.model.beta_max
                    beta = beta_min + (beta_max - beta_min)*t
                    Beta = 1/2*t**2*(beta_max-beta_min) + t*beta_min
                    sigma = 1-np.exp(-Beta)
                    alpha = np.exp(-Beta/2)
                     
                    # These adjustments are necessary because of the data scaler used to train DDPM++ models.  Works for CIFAR-10 ONLY!
                    if size == 8:
                        samples = samples - 3*alpha
                    elif size == 16:
                        samples = samples - 1*alpha
                       
                    samples = upsampling_fn(samples, alpha=2*alpha, sigma=sigma, dataset='cifar')
                
        samples = samples.permute(0, 2, 3, 1).cpu().numpy() 
        samples = np.clip(samples * 255., 0, 255).astype(np.uint8)
        samples = samples.reshape(
            (-1, config.data.image_size, config.data.image_size, config.data.num_channels))
        
        all_samples.append(samples)
        
    all_samples = np.concatenate(all_samples)
    path = os.path.join(eval_dir, f"{save_name}.npy")
    np.save(path, all_samples)    
    del steps, step, state, ema, score_model, sampling_fn, samples
    torch.cuda.empty_cache()
        
    # evaluate FID, IS, KID
    # is_, fid, kid = evaluation.evaluate_samples(all_samples)
    # print('IS', is_, 'FID', fid, 'KID', kid)

sample(
    eval_folder="./eval",
    save_name="test_run",
    number_samples = 10,
    batch = 5,
    langevin_steps = 2,
    conditional_langevin = True,
    langevin_snr = 0.22,
    time = 0.5,
    subspace = 8,
    ckpt_subspace = './pretrained/cifar10_ncsn_8x8.pth',
    ckpt_full = './pretrained/cifar10_ncsn_full.pth',
    device = 0,
) 