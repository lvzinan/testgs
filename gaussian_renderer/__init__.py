#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

def rotation_matrix(angles):
    """
    Generate a rotation matrix for given angles (roll, pitch, yaw).

    :param angles: A tuple of three angles in radians (roll, pitch, yaw)
    :return: A 3x3 rotation matrix
    """
    # print(angles.shape)
    # print(angles)
    # roll, pitch, yaw = angles
    # print(roll,pitch,yaw)

    # Rotation around the X-axis (roll)
    Rx = torch.tensor([[1, 0, 0],
                   [0, torch.cos(angles[0]), -torch.sin(angles[0])],
                   [0, torch.sin(angles[0]), torch.cos(angles[0])]], requires_grad=True).cuda()

    # Rotation around the Y-axis (pitch)
    Ry = torch.tensor([[torch.cos(angles[1]), 0, torch.sin(angles[1])],
                   [0, 1, 0],
                   [-torch.sin(angles[1]), 0, torch.cos(angles[1])]], requires_grad=True).cuda()

    # Rotation around the Z-axis (yaw)
    Rz = torch.tensor([[torch.cos(angles[2]), -torch.sin(angles[2]), 0],
                   [torch.sin(angles[2]), torch.cos(angles[2]), 0],
                   [0, 0, 1]], requires_grad=True).cuda()

    # Combined rotation matrix
    R = Rz @ Ry @ Rx
    return R

def rotate_point(point, angles):
    # 提取旋转角
    alpha, beta, gamma = angles[0], angles[1], angles[2]

    # 计算旋转后的坐标
    # 绕 x 轴旋转
    x_rotated = point[0]
    y_rotated = point[1] * torch.cos(alpha) - point[2] * torch.sin(alpha)
    z_rotated = point[1] * torch.sin(alpha) + point[2] * torch.cos(alpha)

    # 更新点坐标
    point_rotated = torch.tensor([x_rotated, y_rotated, z_rotated])

    # 绕 y 轴旋转
    x_rotated = point_rotated[0] * torch.cos(beta) + point_rotated[2] * torch.sin(beta)
    y_rotated = point_rotated[1]
    z_rotated = -point_rotated[0] * torch.sin(beta) + point_rotated[2] * torch.cos(beta)

    # 更新点坐标
    point_rotated = torch.tensor([x_rotated, y_rotated, z_rotated])

    # 绕 z 轴旋转
    x_rotated = point_rotated[0] * torch.cos(gamma) - point_rotated[1] * torch.sin(gamma)
    y_rotated = point_rotated[0] * torch.sin(gamma) + point_rotated[1] * torch.cos(gamma)
    z_rotated = point_rotated[2]

    # 返回最终旋转后的点
    return torch.tensor([x_rotated, y_rotated, z_rotated])

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, uid = None, iteration=0, trans=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # print(uid)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)


    # if iteration > 0:
    #     world_view_transform = torch.tensor(getWorld2View2(viewpoint_camera.R, viewpoint_camera.T, viewpoint_camera.trans, viewpoint_camera.scale)).transpose(0, 1).cuda()
    #     # print(viewpoint_camera.colmap_id)
    #     # if viewpoint_camera.colmap_id == 108:
    #     #     print(uid)
    #     # print(uid[viewpoint_camera.colmap_id])
    #     pc.train_cam_r.retain_grad()
    #     pc.train_cam_xyz.retain_grad()
    #
    #     R = rotation_matrix(pc.train_cam_r[uid[viewpoint_camera.colmap_id]]).cuda()
    #     T = pc.train_cam_xyz[uid[viewpoint_camera.colmap_id]].reshape(1,3)
    #     if iteration % 100 == 0:
    #         print(pc.train_cam_r[uid[viewpoint_camera.colmap_id]], T, pc._xyz[0])
    #     # print(torch.cat([R, T], axis=0))
    #     # tran = torch.cat([torch.cat([R,T],axis=0),torch.tensor([0,0,0,1]).reshape(4,1).cuda()],axis=1)
    #     # print(world_view_transform,tran)
    #     # world_view_transform = tran @ world_view_transform
    #     # print(world_view_transform)
    #     R0 = world_view_transform[:3, :3]
    #     T0 = world_view_transform[3, :3]
    #     R0 = R @ R0
    #     T0 = T0 + T
    #     # print(viewpoint_camera.world_view_transform)
    #
    #     world_view_transform = torch.cat([torch.cat([R0,T0],axis=0),torch.tensor([0,0,0,1]).reshape(4,1).cuda()],axis=1)
    #     # print(world_view_transform)
    #     # print(world_view_transform)
    #     projection_matrix = getProjectionMatrix(znear=viewpoint_camera.znear, zfar=viewpoint_camera.zfar, fovX=viewpoint_camera.FoVx,
    #                                                  fovY=viewpoint_camera.FoVy).transpose(0, 1).cuda()
    #     full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    #     camera_center = world_view_transform.inverse()[3, :3]
    # else:
    #     world_view_transform = viewpoint_camera.world_view_transform
    #     projection_matrix = viewpoint_camera.projection_matrix
    #     full_proj_transform = viewpoint_camera.full_proj_transform
    #     camera_center = viewpoint_camera.camera_center

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    # iteration_cam = 50000
    if iteration > 40000:
        t0 = pc.get_xyz
        # t1 = pc.get_xyz
        # print("former", pc.get_xyz[0])
        pc.train_cam_r.retain_grad()
        pc.train_cam_xyz.retain_grad()
        # R = rotation_matrix(pc.train_cam_r[uid[viewpoint_camera.colmap_id]].reshape(3,1))
        q = torch.nn.functional.normalize(pc.train_cam_r[uid[viewpoint_camera.colmap_id]].reshape(1,4), p=2, dim=1).reshape(4,)

        w, x, y, z = q[0], q[1], q[2], q[3]

        R = torch.stack((1 - 2 * (y * y) - 2 * (z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                             2 * (x * y + w * z), 1 - 2 * (x * x) - 2 * (z * z), 2 * (y * z - w * x),
                             2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x) - 2 * (y * y)),-1)
        R = R.reshape(3,3)

        t0 = R @ t0.transpose(0,1)

        T = pc.train_cam_xyz[uid[viewpoint_camera.colmap_id]].reshape(3,1)
        t0 = (t0 + T).transpose(0,1)
        # print(t0[0])
        if iteration % 100 == 0:
            print(R, T, pc._xyz[0], q)
    else:
        t0 = pc.get_xyz

    means3D = t0
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        if iteration > 40000:
            cov3D_precomp = pc.get_covariance(scaling_modifier, R)
        else:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (t0 - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
