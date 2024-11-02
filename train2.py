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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, Scene_init, Scene_init_point
# from scene.camera_optimise import optimise_cam
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.system_utils import searchForMaxIteration
from plyfile import PlyData, PlyElement
import numpy as np
from scene.cameras import Camera

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

R_mah = np.array([[0.28226713, -0.94030478, -0.19013728],
                  [-0.89575313, -0.32928481, 0.29866007],
                  [-0.34344081, 0.08601414, -0.93522723]])


def read_plyblock(ply_path, x_min, x_max, y_min, y_max):
    plydata = PlyData.read(ply_path)  # 读取文件
    # print(plydata)
    # print(plydata.elements[0].data[0])
    # print(len(plydata.elements[0].data))
    num1 = len(plydata.elements[0].data)
    # data = np.empty([1,62])
    num = 0

    # data = np.asarray(plydata.elements[0]["x"]).reshape(4221190,1)
    # print(data.shape)
    # print(plydata.elements[0].properties)
    data = np.stack((np.asarray(plydata.elements[0]["x"]),
                     np.asarray(plydata.elements[0]["y"]),
                     np.asarray(plydata.elements[0]["z"])), axis=1)

    for p in plydata.elements[0].properties:
        if p.name == "x" or p.name == "y" or p.name == "z":
            continue

        data = np.hstack((data, np.asarray(plydata.elements[0][p.name]).reshape(num1, 1)))

    t = 0
    print(data.shape)

    x = np.asarray(plydata.elements[0]["x"])
    y = np.asarray(plydata.elements[0]["y"])
    z = np.asarray(plydata.elements[0]["z"])

    data1 = data[data[:, 0] >= x_min]
    data1 = data1[data1[:, 0] < x_max]
    data1 = data1[data1[:, 1] >= y_min]
    data1 = data1[data1[:, 1] < y_max]
    data1 = data1[
        float(R_mah[2, 0]) * data1[:, 0] + float(R_mah[2, 1]) * data1[:, 1] + float(R_mah[2, 2]) * data1[:, 2] <= -1.4]
    print(data1.shape)

    return data1


def create_output(data, filename):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2']
    for i in range(45):
        stri = 'f_rest_' + str(i)
        l += [stri]
    l += ['opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
    # print(l)
    dtype_full = [(attribute, 'f4') for attribute in l]

    elements = np.empty(data.shape[0], dtype=dtype_full)
    attributes = data
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(filename)


def rotation_matrix(angles):
    """
    Generate a rotation matrix for given angles (roll, pitch, yaw).

    :param angles: A tuple of three angles in radians (roll, pitch, yaw)
    :return: A 3x3 rotation matrix
    """
    roll, pitch, yaw = angles

    # Rotation around the X-axis (roll)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])

    # Rotation around the Y-axis (pitch)
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])

    # Rotation around the Z-axis (yaw)
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])

    # Combined rotation matrix
    R = Rz @ Ry @ Rx
    return R


def rotate_point(point, angles):
    """
    Rotate a point in 3D space using the given angles.

    :param point: A 3-element list or array representing the point (x, y, z)
    :param angles: A tuple of three angles in radians (roll, pitch, yaw)
    :return: A rotated point as a numpy array
    """
    # print(angles)
    R = rotation_matrix(angles)
    # print(R)
    rotated_point = R @ np.array(point)
    return rotated_point

# 定义一个正交化函数
def orthogonalize_matrix(R):
    U, _, Vt = torch.svd(R)  # 使用奇异值分解
    return U @ Vt  # 返回正交矩阵


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,
             num_block=2):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    max_x, min_x, max_y, min_y, max_z = Scene_init(dataset)
    max_x1 = max_x
    max_y1 = max_y
    min_x1 = min_x
    min_y1 = min_y
    print(max_x1, min_x1, max_y1, min_y1)
    # max_z, min_z = Scene_init_point(dataset, R=R_mah)
    # print("z", max_z, min_z)
    # print(num_block)
    blocks = []
    for i in range(num_block * num_block):
        x_limit_min = min_x1 + (max_x1 - min_x1) / num_block * (i % num_block)
        x_limit_max = min_x1 + (max_x1 - min_x1) / num_block * (i % num_block + 1)
        y_limit_min = min_y1 + (max_y1 - min_y1) / num_block * (i // num_block)
        y_limit_max = min_y1 + (max_y1 - min_y1) / num_block * (i // num_block + 1)
        blocks.append((x_limit_min, x_limit_max, y_limit_min, y_limit_max))
        print(x_limit_min, x_limit_max, y_limit_min, y_limit_max)

    # print(dataset.model_path)
    # print(os.path.join(dataset.model_path, "point_cloud"))
    # iteration = searchForMaxIteration(os.path.join(args.model_path, "point_cloud"))

    # iteration = searchForMaxIteration(os.path.join(args.model_path, "point_cloud"))
    # point_cloud_path = os.path.join(args.model_path, "point_cloud/iteration_{}".format(iteration))
    # for i in range(len(blocks)):
    #     # ply_path = os.path.join(point_cloud_path, "point_cloud" + str(i) + ".ply")
    #     ply_path = "E:\\download\\" + "point_cloud" + str(i) + ".ply"
    #     ply = read_plyblock(ply_path)
    #     if i == 0:
    #         plys = ply
    #     else:
    #         plys = np.vstack([plys, ply])
    # create_output(plys, "E:\\download\\" + "point_cloud" + ".ply")

    for i in range(len(blocks)):
        first_iter = 0
        x_min = blocks[i][0]
        x_max = blocks[i][1]
        y_min = blocks[i][2]
        y_max = blocks[i][3]
        print(x_min, x_max, y_min, y_max)

        # optimise_cam0 = optimise_cam().cuda()

        gaussians = GaussianModel(dataset.sh_degree)
        print("gaussians end")
        scene = Scene(dataset, gaussians, train=True, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, block=i)
        print("scene end")
        gaussians.training_setup(opt)
        if checkpoint:
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)

        viewpoint_stack = None
        ema_loss_for_log = 0.0
        progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
        first_iter += 1
        print("begin")
        for iteration in range(first_iter, opt.iterations + 1):
            if iteration % 1000 == 1:
                scene.scene_batch(dataset)

            if network_gui.conn == None:
                network_gui.try_connect()
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                    if custom_cam != None:
                        net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                                   0).contiguous().cpu().numpy())
                    network_gui.send(net_image_bytes, dataset.source_path)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    network_gui.conn = None

            iter_start.record()

            gaussians.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
            # print(viewpoint_cam.image_name)
            # print(viewpoint_cam.original_image.shape)

            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True

            bg = torch.rand((3), device="cuda") if opt.random_background else background

            gt_image = viewpoint_cam.original_image.cuda()
            # print(viewpoint_cam.R,viewpoint_cam.T)
            # if iteration > 0:
            #     R, T = gaussians.train_cam_r, gaussians.train_cam_xyz
            #     # print(viewpoint_cam.colmap_id)
            #     # print(scene.uid)
            #     # print(viewpoint_cam.colmap_id)
            #     # print(scene.uid[viewpoint_cam.colmap_id])
            #     r = gaussians.train_cam_r.cpu().detach().numpy()[scene.uid[viewpoint_cam.colmap_id]]
            #     t = T.cpu().detach().numpy()[scene.uid[viewpoint_cam.colmap_id]]
            #     if iteration % 100 == 0:
            #         print(r, t)
            #     R1 = rotate_point(viewpoint_cam.R, r)
            #     T1 = viewpoint_cam.T + t
            #     print(viewpoint_cam.R,viewpoint_cam.T)
            #     viewpoint_cam1 = Camera(colmap_id=viewpoint_cam.colmap_id, R=R1, T=T1,
            #                             FoVx=viewpoint_cam.FoVx, FoVy=viewpoint_cam.FoVy,
            #                             image=viewpoint_cam.image, gt_alpha_mask=viewpoint_cam.gt_alpha_mask,
            #                             image_name=viewpoint_cam.image_name, uid=viewpoint_cam.uid,
            #                             data_device=viewpoint_cam.data_device)
            # else:
            #     viewpoint_cam1 = viewpoint_cam

            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, uid=scene.uid, iteration=iteration, trans=True)
            # print(render_pkg["render"].shape)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[
                "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            # Loss
            # gt_image = viewpoint_cam.original_image.cuda()
            # print(gt_image.shape)
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            if iteration > 40000 and iteration < 50000:
                gaussians._xyz.requires_grad = False
                gaussians._features_dc.requires_grad = False
                gaussians._features_rest.requires_grad = False
                gaussians._opacity.requires_grad = False
                gaussians._scaling.requires_grad = False
                gaussians._rotation.requires_grad = False
            else:
                gaussians._xyz.requires_grad = True
                gaussians._features_dc.requires_grad = True
                gaussians._features_rest.requires_grad = True
                gaussians._opacity.requires_grad = True
                gaussians._scaling.requires_grad = True
                gaussians._rotation.requires_grad = True
            loss.backward()



            iter_end.record()

            with torch.no_grad():
                # if iteration > 3000:
                #     gaussians.train_cam_r[scene.uid[viewpoint_cam.colmap_id]] = orthogonalize_matrix(
                #         gaussians.train_cam_r[scene.uid[viewpoint_cam.colmap_id]])

                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                # Log and save
                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                testing_iterations, scene, render, (pipe, background), dataset.model_path)
                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)

                # Densification
                if iteration < opt.densify_until_iter or iteration > 60000 and iteration < 80000:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                         radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                    # print(scene.cameras_extent)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent,
                                                    size_threshold, x_min, x_max, y_min, y_max, R=R_mah)

                    if iteration % opt.opacity_reset_interval == 0 or (
                            dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()
                if iteration % 1000 == 0:
                    gaussians.delet(x_min, x_max, y_min, y_max)

                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)

                if (iteration in checkpoint_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    iteration = searchForMaxIteration(os.path.join(dataset.model_path, "point_cloud"))
    point_cloud_path = os.path.join(dataset.model_path, "point_cloud/iteration_{}".format(iteration))
    for i in range(len(blocks)):
        x_min = blocks[i][0]
        x_max = blocks[i][1]
        y_min = blocks[i][2]
        y_max = blocks[i][3]
        ply_path = os.path.join(point_cloud_path, "point_cloud" + str(i) + ".ply")
        # ply_path = "E:\\download\\" + "point_cloud" + str(i) + ".ply"
        ply = read_plyblock(ply_path, x_min, x_max, y_min, y_max)
        if i == 0:
            plys = ply
        else:
            plys = np.vstack([plys, ply])
    ply_path = os.path.join(point_cloud_path, "point_cloud" + ".ply")
    create_output(plys, ply_path)

    # gaussians = GaussianModel(dataset.sh_degree)
    # scene = Scene(dataset, gaussians, load_iteration=-1, train=True, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, block=i)
    # gaussians.training_setup(opt)
    # if checkpoint:
    #     (model_params, first_iter) = torch.load(checkpoint)
    #     gaussians.restore(model_params, opt)
    #
    # bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    # background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    #
    # iter_start = torch.cuda.Event(enable_timing = True)
    # iter_end = torch.cuda.Event(enable_timing = True)
    #
    # viewpoint_stack = None
    # ema_loss_for_log = 0.0
    # progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    # first_iter += 1
    #
    # for iteration in range(first_iter, opt.iterations_f + 1):
    #     if network_gui.conn == None:
    #         network_gui.try_connect()
    #     while network_gui.conn != None:
    #         try:
    #             net_image_bytes = None
    #             custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
    #             if custom_cam != None:
    #                 net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
    #                 net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
    #                                                                                                            0).contiguous().cpu().numpy())
    #             network_gui.send(net_image_bytes, dataset.source_path)
    #             if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
    #                 break
    #         except Exception as e:
    #             network_gui.conn = None
    #
    #     iter_start.record()
    #
    #     gaussians.update_learning_rate(iteration)
    #
    #     # Every 1000 its we increase the levels of SH up to a maximum degree
    #     if iteration % 1000 == 0:
    #         gaussians.oneupSHdegree()
    #
    #     # Pick a random Camera
    #     if not viewpoint_stack:
    #         viewpoint_stack = scene.getTrainCameras().copy()
    #     viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
    #
    #     # Render
    #     if (iteration - 1) == debug_from:
    #         pipe.debug = True
    #
    #     bg = torch.rand((3), device="cuda") if opt.random_background else background
    #
    #     render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
    #     image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
    #     render_pkg["visibility_filter"], render_pkg["radii"]
    #
    #     # Loss
    #     gt_image = viewpoint_cam.original_image.cuda()
    #     Ll1 = l1_loss(image, gt_image)
    #     loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
    #     loss.backward()
    #
    #     iter_end.record()
    #
    #     with torch.no_grad():
    #         # Progress bar
    #         ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
    #         if iteration % 10 == 0:
    #             progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
    #             progress_bar.update(10)
    #         if iteration == opt.iterations:
    #             progress_bar.close()
    #
    #         # Log and save
    #         training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
    #                         testing_iterations, scene, render, (pipe, background))
    #         if (iteration in saving_iterations):
    #             print("\n[ITER {}] Saving Gaussians".format(iteration))
    #             scene.save(iteration)
    #
    #         # Densification
    #         if iteration < opt.densify_until_iter:
    #             # Keep track of max radii in image-space for pruning
    #             gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
    #                                                                  radii[visibility_filter])
    #             gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
    #             # print(scene.cameras_extent)
    #
    #             if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
    #                 size_threshold = 20 if iteration > opt.opacity_reset_interval else None
    #                 # print(scene.cameras_extent)
    #                 gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold,
    #                                             x_min, x_max, y_min, y_max)
    #
    #             if iteration % opt.opacity_reset_interval == 0 or (
    #                     dataset.white_background and iteration == opt.densify_from_iter):
    #                 gaussians.reset_opacity()
    #         if iteration % 1000 == 0 and iteration < opt.densify_until_iter:
    #             gaussians.delet(x_min, x_max, y_min, y_max)
    #
    #         # Optimizer step
    #         if iteration < opt.iterations:
    #             gaussians.optimizer.step()
    #             gaussians.optimizer.zero_grad(set_to_none=True)
    #
    #         if (iteration in checkpoint_iterations):
    #             print("\n[ITER {}] Saving Checkpoint".format(iteration))
    #             torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, model_path):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    # lambda_cam = iteration / (opt.iterations + 1)
                    # if lambda_cam > 0.5:
                    #     trans = True
                    # else:
                    #     trans = False
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, uid=scene.uid,iteration=iteration, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                print(model_path)
                with open(model_path + "/log.txt", "a") as f:
                    f.writelines([str(iteration), str(l1_test), str(psnr_test), "\n"])
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[1_000, 3_000, 7_000, 10_000, 20_000, 30_000,33_000,36_000,39_000, 40_000,45_000,46_000,47_000,48_000, 49_000,
                                 50_000,53_000,56_000,58_000, 60_000,65_000,68_000, 70_000,73_000,75_000,78_000,80_000,83_000,85_000,87_000,
                                 90_000,95_000,100_000,105_000, 110_000, 120_000, 150_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000, 60_000, 90_000,100_000,120_000, 150_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--num_block", type=int, default=2)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from, num_block=args.num_block)

    # All done
    print("\nTraining complete.")
