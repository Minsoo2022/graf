import torch
import numpy as np
import os
from tqdm import tqdm

from submodules.GAN_stability.gan_training.train import toggle_grad, Trainer as TrainerBase
from submodules.GAN_stability.gan_training.eval import Evaluator as EvaluatorBase
from submodules.GAN_stability.gan_training.metrics import FIDEvaluator, KIDEvaluator

from .utils import save_video, color_depth_map
import mrcfile
import open3d as o3d
from skimage.measure import marching_cubes


class Trainer(TrainerBase):
    def __init__(self, *args, use_amp=False, **kwargs):
        super(Trainer, self).__init__(*args, **kwargs)
        self.use_amp = use_amp
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

    def generator_trainstep(self, y, z):
        if not self.use_amp:
            return super(Trainer, self).generator_trainstep(y, z)
        assert (y.size(0) == z.size(0))
        toggle_grad(self.generator, True)
        toggle_grad(self.discriminator, False)
        self.generator.train()
        self.discriminator.train()
        self.g_optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            x_fake = self.generator(z, y)
            d_fake = self.discriminator(x_fake, y)
            gloss = self.compute_loss(d_fake, 1)
        self.scaler.scale(gloss).backward()

        self.scaler.step(self.g_optimizer)
        self.scaler.update()

        return gloss.item()

    def discriminator_trainstep(self, x_real, y, z):
        return super(Trainer, self).discriminator_trainstep(x_real, y, z)       # spectral norm raises error for when using amp


class Evaluator(EvaluatorBase):
    def __init__(self, eval_fid_kid, *args, **kwargs):
        super(Evaluator, self).__init__(*args, **kwargs)
        if eval_fid_kid:
            self.inception_eval = FIDEvaluator(
              device=self.device,
              batch_size=self.batch_size,
              resize=True,
              n_samples=20000,
              n_samples_fake=1000,
            )

    def get_rays(self, pose):
        return self.generator.val_ray_sampler(self.generator.H, self.generator.W,
                                              self.generator.focal, pose)[0]

    def create_samples(self, z, poses=None):
        self.generator.eval()

        N_samples = len(z)
        device = self.generator.device
        z = z.to(device).split(self.batch_size)
        if poses is None:
            rays = [None] * len(z)
        else:
            rays = torch.stack([self.get_rays(poses[i].to(device)) for i in range(N_samples)])
            rays = rays.split(self.batch_size)

        rgb, disp, acc = [], [], []
        with torch.no_grad():
            for z_i, rays_i in tqdm(zip(z, rays), total=len(z), desc='Create samples...'):
                bs = len(z_i)
                if rays_i is not None:
                    rays_i = rays_i.permute(1, 0, 2, 3).flatten(1, 2)       # Bx2x(HxW)xC -> 2x(BxHxW)x3
                rgb_i, disp_i, acc_i, _ = self.generator(z_i, rays=rays_i)

                reshape = lambda x: x.view(bs, self.generator.H, self.generator.W, x.shape[1]).permute(0, 3, 1, 2)  # (NxHxW)xC -> NxCxHxW
                rgb.append(reshape(rgb_i).cpu())
                disp.append(reshape(disp_i).cpu())
                acc.append(reshape(acc_i).cpu())

        rgb = torch.cat(rgb)
        disp = torch.cat(disp)
        acc = torch.cat(acc)

        depth = self.disp_to_cdepth(disp)

        return rgb, depth, acc

    def make_video(self, basename, z, poses, as_gif=True):
        """ Generate images and save them as video.
        z (N_samples, zdim): latent codes
        poses (N_frames, 3 x 4): camera poses for all frames of video
        """
        N_samples, N_frames = len(z), len(poses)

        # reshape inputs
        z = z.unsqueeze(1).expand(-1, N_frames, -1).flatten(0, 1)  # (N_samples x N_frames) x z_dim
        poses = poses.unsqueeze(0) \
            .expand(N_samples, -1, -1, -1).flatten(0, 1)  # (N_samples x N_frames) x 3 x 4

        rgbs, depths, accs = self.create_samples(z, poses=poses)

        reshape = lambda x: x.view(N_samples, N_frames, *x.shape[1:])
        rgbs = reshape(rgbs)
        depths = reshape(depths)
        print('Done, saving', rgbs.shape)

        fps = min(int(N_frames / 2.), 25)          # aim for at least 2 second video
        for i in range(N_samples):
            save_video(rgbs[i], basename + '{:04d}_rgb.mp4'.format(i), as_gif=as_gif, fps=fps)
            save_video(depths[i], basename + '{:04d}_depth.mp4'.format(i), as_gif=as_gif, fps=fps)

    def make_3D_grid(self, N=256, voxel_origin=[0, 0, 0], cube_length=5.0):
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

        num_samples = N ** 3

        return samples.unsqueeze(0), voxel_origin, voxel_size

    def make_mesh(self, path, z):
        self.generator.eval()
        self.batch_size = 1
        N_samples = len(z)
        device = self.generator.device
        z = z.to(device).split(self.batch_size)
        with torch.no_grad():
            voxel_resolution = 128
            cube_size = 2.0
            points = self.make_3D_grid(N=voxel_resolution, voxel_origin=[0, 0, 0], cube_length=cube_size)[0][0]
            # points = points.repeat(self.batch_size, 1, 1)
            for z_i in tqdm(z, total=len(z), desc='Create samples...'):
                bs = len(z_i)
                sigma_i, _ = self.generator.out_sigma(z_i, points=points)
                sigma_i = sigma_i.reshape(voxel_resolution, voxel_resolution, voxel_resolution)
                with mrcfile.new_mmap(os.path.join(path), overwrite=True, shape=sigma_i.shape,
                                      mrc_mode=2) as mrc:
                    mrc.data[:] = sigma_i

                marching_cubes_level = 54
                verts, faces, normals, values = marching_cubes(
                    np.array(sigma_i),
                    level=marching_cubes_level,
                    spacing=(1.0, 1.0, 1.0),
                    # spacing = (opt.voxel_resolution, opt.voxel_resolution, opt.voxel_resolution),
                    gradient_direction='descent',
                    step_size=1,
                    allow_degenerate=True,
                    method='lewiner',
                    mask=None)

                verts_vec = o3d.utility.Vector3dVector(verts)
                faces_vec = o3d.utility.Vector3iVector(faces)
                verts_normals_vec = o3d.utility.Vector3dVector(normals)

                mesh1 = o3d.geometry.TriangleMesh(verts_vec, faces_vec)
                mesh1.vertex_normals = verts_normals_vec

                samples = torch.from_numpy(
                    (verts - voxel_resolution / 2) / (voxel_resolution / 2) * (cube_size / 2))
                samples = samples.unsqueeze(0)
                samples = samples.to(device)

                sigma, color = self.generator.out_sigma(z_i, points=samples[0])
                mesh1.vertex_colors = o3d.utility.Vector3dVector(color.detach().cpu().squeeze(0).numpy())
                o3d.io.write_triangle_mesh(f'./albedo.ply', mesh1)

    def disp_to_cdepth(self, disps):
        """Convert depth to color values"""
        if (disps == 2e10).all():           # no values predicted
            return torch.ones_like(disps)

        near, far = self.generator.render_kwargs_test['near'], self.generator.render_kwargs_test['far']

        disps = disps / 2 + 0.5  # [-1, 1] -> [0, 1]

        depth = 1. / torch.max(1e-10 * torch.ones_like(disps), disps)  # disparity -> depth
        depth[disps == 1e10] = far  # set undefined values to far plane

        # scale between near, far plane for better visualization
        depth = (depth - near) / (far - near)

        depth = np.stack([color_depth_map(d) for d in depth[:, 0].detach().cpu().numpy()])  # convert to color
        depth = (torch.from_numpy(depth).permute(0, 3, 1, 2) / 255.) * 2 - 1  # [0, 255] -> [-1, 1]

        return depth

    def compute_fid_kid(self, sample_generator=None):
        if sample_generator is None:
            def sample():
                while True:
                    z = self.zdist.sample((self.batch_size,))
                    rgb, _, _ = self.create_samples(z)
                    # convert to uint8 and back to get correct binning
                    rgb = (rgb / 2 + 0.5).mul_(255).clamp_(0, 255).to(torch.uint8).to(torch.float) / 255. * 2 - 1
                    yield rgb.cpu()
            
            sample_generator = sample()

        fid, (kids, vars) = self.inception_eval.get_fid_kid(sample_generator)
        kid = np.mean(kids)
        return fid, kid
