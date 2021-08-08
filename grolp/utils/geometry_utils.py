import math

import numpy as np
import open3d as o3d


def pixel_coord_np(width, height):
    """
    Pixel in homogenous coordinate
    Returns:
        Pixel coordinate:       [3, width * height]
    """
    x = np.linspace(0, width - 1, width).astype(np.int)
    y = np.linspace(0, height - 1, height).astype(np.int)
    [x, y] = np.meshgrid(x, y)
    return np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))


def intrinsic_from_fov(height, width, fov=90):
    """
    Basic Pinhole Camera Model
    intrinsic params from fov and sensor width and height in pixels
    Returns:
        K:      [4, 4]
    """
    px, py = (width / 2, height / 2)
    hfov = fov / 360. * 2. * np.pi
    fx = width / (2. * np.tan(hfov / 2.))

    vfov = 2. * np.arctan(np.tan(hfov / 2) * height / width)
    fy = height / (2. * np.tan(vfov / 2.))

    return np.array([[fx, 0, px, 0.],
                     [0, fy, py, 0.],
                     [0, 0, 1., 0.],
                     [0., 0., 0., 1.]])


def get_bbox_world_coordinates(boxes, rgb, depth, fov):
    # Get intrinsic parameters
    height, width, _ = rgb.shape
    K = intrinsic_from_fov(height, width, fov)  # +- 45 degrees
    K_inv = np.linalg.inv(K)

    # Get pixel coordinates
    # this should be a matrix (3, npoints) so given the bounding boxes we compute the centres
    pixel_coords = np.stack(
        [(boxes[:, 0] + boxes[:, 2]) // 2, (boxes[:, 1] + boxes[:, 3]) // 2, np.ones(boxes.shape[0], )], 1)
    pixel_coords = np.transpose(pixel_coords, (1, 0))
    # Apply back-projection: K_inv @ pixels * depth
    depth_for_coords = np.array([depth[x, y] for x, y in zip(*pixel_coords[:2, :].astype(np.int32))])
    cam_coords = K_inv[:3, :3] @ pixel_coords * depth_for_coords
    # Limit points to 150m in the z-direction for visualisation
    # cam_coords = cam_coords[:, np.where(cam_coords[2] <= 150)[0]]

    # Visualize
    pcd_cam = o3d.geometry.PointCloud()
    pcd_cam.points = o3d.utility.Vector3dVector(cam_coords.T[:, :3])
    # Flip it, otherwise the pointcloud will be upside down
    pcd_cam.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    return np.asarray(pcd_cam.points)


def calculate_angles(center_x, center_y, h_view_angle, v_view_angle):
    # angles are in degrees, not in radians
    # (center_x, center_y): top left corner of the image is (0, 0)
    # h_view_angle: forward is 0 degree, increases to 360 degree clockwise (e.g., right is 90 degrees)
    # v_view_angle: h view angle is 0 degree, highest view angle is 30 degree,
    # lowest view angle is -60 degree. In the output of this function, we use -15 degee as the new center (0 degree).
    h_field_of_view = 90  # this is the default value used by AI2Thor
    v_field_of_view = 90

    h_object_angle = np.arctan((center_x - 0.5) * math.tan(math.radians(h_field_of_view / 2)) * 2)
    v_object_angle = np.arctan((0.5 - center_y) * math.tan(math.radians(v_field_of_view / 2)) * 2)
    h_object_angle = h_view_angle + np.degrees(h_object_angle)
    v_object_angle = v_view_angle + 15 + np.degrees(v_object_angle)

    return h_object_angle, v_object_angle
