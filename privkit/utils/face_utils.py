"""Facial utility methods."""

import os
import sys
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from pathlib import Path
from open3d.cpu.pybind.geometry import PointCloud

from privkit.utils import dev_utils as du


def block_print():
    """
    Temporarily redirects standard output to suppress printing
    """
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    """
    Restores standard output, allowing printing to the console
    """
    sys.stdout = sys.__stdout__


def list_folders(directory: str):
    """
    Retrieve a list of folder paths within a given directory

    :param str directory: The directory to search for folders
    :return: A list containing the absolute paths of folders
    """
    folders = os.listdir(directory)
    folder_paths = [directory + '/' + folder for folder in folders if not folder.startswith('.')]
    folder_paths = [folder_path for folder_path in folder_paths if not os.path.isfile(folder_path)]

    return folder_paths


def list_files(directory: str, extension: tuple):
    """
    Retrieve a list of file paths with a specified extension within a given directory

    :param str directory: The directory to search for files
    :param tuple extension: The file extension(s) to filter the results (e.g., ('.png', '.jpg'))
    :return: A list containing the absolute paths of files matching the specified extension(s)
    """
    files = os.listdir(directory)
    path_files = [f'{directory}/{file}' for file in files if not file.startswith('.') and file.endswith(extension)]
    path_files = [path_file for path_file in path_files if os.path.isfile(path_file)]

    return path_files


def pcd_to_img(pcd: PointCloud, xlim: tuple, ylim: tuple, store_path: str, axis_face: str, point_size: float = 25):
    """
    Stores a 'png' image resulting from an orthographic projection of a point cloud

    :param PointCloud pcd: the point cloud to be projected
    :param tuple xlim: the interval of the Y axis of the scatter plot
    :param tuple ylim: the interval of the X axis of the scatter plot
    :param str store_path: the path where the image should be stored
    :param str axis_face: the axis where the face is pointing ('x', 'y' or 'z')
    :param int point_size: the size of each point on the scatter plot
    """
    if axis_face not in ['x', 'y', 'z']:
        raise TypeError("Invalid axis_face.")

    point_pcd = np.array(pcd.points)
    color_pcd = np.array(pcd.colors)

    ind = np.argsort(np.array(pcd.points)[:, 1])  # to allow Point-mesh-point to be sorted by 0
    x_pcd = point_pcd[:, 0][ind]
    y_pcd = point_pcd[:, 1][ind]
    z_pcd = point_pcd[:, 2][ind]
    color_pcd = color_pcd[ind]

    fig = plt.gcf()

    if axis_face == 'x':
        plt.scatter(z_pcd, y_pcd, c=color_pcd, s=point_size)

    elif axis_face == 'y':
        plt.scatter(z_pcd, x_pcd, c=color_pcd, s=point_size)

    elif axis_face == 'z':
        plt.scatter(y_pcd, x_pcd, c=color_pcd, s=point_size)

    else:
        return 'Invalid axis_face'

    plt.xlim(xlim[0], xlim[1])  # -0.25, 0.25
    plt.ylim(ylim[0], ylim[1])  # -0.18, 0.18

    plt.axis('off')
    plt.xticks([]), plt.yticks([])

    fig.savefig(f'{store_path}', bbox_inches='tight')
    plt.close(fig)
    fig.clear()

    du.log('Image saved!')


def folder_to_img(folder: str, store_dir: str, xlim: tuple, ylim: tuple, axis_face: str, point_size: int = 25):
    """
    Creates a new folder and stores 'png' images resulting from an orthographic projection of point cloud files within a folder

    :param str folder: the folder path to search for files
    :param tuple xlim: the interval of the Y axis of the scatter plot
    :param tuple ylim: the interval of the X axis of the scatter plot
    :param str store_dir: the path where the projected images should be stored
    :param str axis_face: the axis where the face is pointing ('x', 'y' or 'z')
    :param int point_size: the size of each point on the scatter plot
    """
    if axis_face not in ['x', 'y', 'z']:
        raise TypeError("Invalid axis_face.")

    Path(store_dir).mkdir(parents=True, exist_ok=True)
    for file_path in list_files(folder, ('.ply')):
        file_name = Path(file_path).stem
        store_path = f'{store_dir}/{file_name}.png'

        pcd = o3d.io.read_point_cloud(file_path)
        block_print()
        pcd_to_img(pcd, xlim=xlim, ylim=ylim, store_path=store_path, axis_face=axis_face, point_size=point_size)

    enable_print()
    du.log('Folder saved!')
