import argparse
import os
import copy

import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')

import mrcfile
from tqdm import trange
import open3d as o3d
from skimage.measure import marching_cubes
import numpy as np

from graf.config import get_data, build_models, update_config
from graf.utils import count_trainable_parameters

import sys
sys.path.append('submodules')        # needed to make imports work in GAN_stability

from submodules.GAN_stability.gan_training.checkpoints import CheckpointIO
from submodules.GAN_stability.gan_training.distributions import get_zdist
from submodules.GAN_stability.gan_training.config import load_config

def make_3D_grid(N=256, voxel_origin=[0, 0, 0], cube_length=5.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length / 2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    return samples, voxel_origin, voxel_size


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(
        description='Train a GAN with different regularization strategies.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--marching_cubes_levels', nargs='+', default=[20, 30, 40, 50])
    parser.add_argument('--num_samples', type=int, default=2000)
    parser.add_argument('--output_dir', type=str, default='shapes')
    parser.add_argument('--out_mrc', action='store_true')
    opt, unknown = parser.parse_known_args()
    
    config = load_config(opt.config, 'configs/default.yaml')
    config['data']['fov'] = float(config['data']['fov'])
    config = update_config(config, unknown)

    name = opt.config.rsplit('/', 1)[-1].rsplit('.', 1)[-2] + '_graf'
    out_folder = f'{opt.output_dir}/{name}'

    # make output folder 
    os.makedirs(f'{out_folder}/sigmas', exist_ok=True)
    for marching_cubes_level in opt.marching_cubes_levels:
        os.makedirs(f'{out_folder}/{marching_cubes_level}', exist_ok=True)

    out_dir = os.path.join(config['training']['outdir'], config['expname'])
    config['expname'] = '%s_%s' % (config['data']['type'], config['data']['imsize'])
    out_dir = os.path.join(config['training']['outdir'], config['expname'] + '_from_pretrained')
    checkpoint_dir = os.path.join(out_dir, 'chkpts')

    # Logger
    checkpoint_io = CheckpointIO(
        checkpoint_dir=checkpoint_dir
    )

    device = torch.device("cuda:0")

    # Dataset
    _, hwfr, _ = get_data(config)
    if config['data']['orthographic']:
        hw_ortho = (config['data']['far']-config['data']['near'], config['data']['far']-config['data']['near'])
        hwfr[2] = hw_ortho
    config['data']['hwfr'] = hwfr         # add for building generator

    # Create models
    generator, _ = build_models(config, disc=False)
    print('Generator params: %d' % count_trainable_parameters(generator))

    # Put models on gpu if needed
    generator = generator.to(device)
    
    # Register modules to checkpoint
    checkpoint_io.register_modules(
        **generator.module_dict  # treat NeRF specially
    )

    # Get model file
    if config.get('model_file', None) is not None:
        model_file = config['model_file']
    else:
        config_pretrained = load_config('configs/pretrained_models.yaml', 'configs/pretrained_models.yaml')
        model_file = config_pretrained[config['data']['type']][config['data']['imsize']]
    print("Load from", model_file)

    # Test generator, use model average
    generator_test = copy.deepcopy(generator)
    generator_test.parameters = lambda: generator_test._parameters
    generator_test.named_parameters = lambda: generator_test._named_parameters
    checkpoint_io.register_modules(**{k + '_test': v for k, v in generator_test.module_dict.items()})

    generator = generator_test

    # Load checkpoint
    print('load %s' % os.path.join(checkpoint_dir, model_file))
    load_dict = checkpoint_io.load(model_file)
    it = load_dict.get('it', -1)
    epoch_idx = load_dict.get('epoch_idx', -1)

    # Distributions
    zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'], device=device)
    
    z = zdist.sample((opt.num_samples,)).to(device)
    voxel_resolution = 128
    cube_size = 2.0
    points = make_3D_grid(N=voxel_resolution, voxel_origin=[0, 0, 0], cube_length=cube_size)[0]

    with torch.no_grad():
        for z_i, seed in zip(z, trange(opt.num_samples, desc="Create samples")):
            torch.manual_seed(seed)

            sigma_i, _ = generator.out_sigma(z_i.unsqueeze(0), points=points)
            sigma_i = sigma_i.reshape(voxel_resolution, voxel_resolution, voxel_resolution).cpu().numpy()

            if opt.out_mrc:
                # save sdf as mrc
                with mrcfile.new_mmap(f'{out_folder}/sigmas/{seed}.mrc', overwrite=True, shape=sigma_i.shape, mrc_mode=2) as mrc:
                    mrc.data[:] = sigma_i
            
            for marching_cubes_level in opt.marching_cubes_levels:
                verts, faces, normals, _ = marching_cubes(
                    sigma_i,
                    level=marching_cubes_level,
                    spacing=(1.0, 1.0, 1.0),
                    # spacing = (opt.voxel_resolution, opt.voxel_resolution, opt.voxel_resolution),
                    gradient_direction='descent',
                    step_size=1,
                    allow_degenerate=True,
                    method='lewiner',
                    mask=None
                )

                verts_vec = o3d.utility.Vector3dVector(verts)
                faces_vec = o3d.utility.Vector3iVector(faces)
                verts_normals_vec = o3d.utility.Vector3dVector(normals)

                mesh = o3d.geometry.TriangleMesh(verts_vec, faces_vec)
                mesh.vertex_normals = verts_normals_vec

                samples = torch.from_numpy(
                    (verts - voxel_resolution / 2) / (voxel_resolution / 2) * (cube_size / 2))
                samples = samples.unsqueeze(0)
                samples = samples.to(device)

                _, color = generator.out_sigma(z_i.unsqueeze(0), points=samples[0])
                # color = (color / 2 + 0.5).clamp(0, 1)
                mesh.vertex_colors = o3d.utility.Vector3dVector(color.detach().cpu().squeeze(0).numpy())

                o3d.io.write_triangle_mesh(f'{out_folder}/{marching_cubes_level}/{seed}.ply', mesh)
