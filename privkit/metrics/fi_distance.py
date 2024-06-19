import numpy as np
import matplotlib.pyplot as plt

from numpy import cov
from numpy import trace
from numpy import asarray
from numpy import iscomplexobj
from scipy.linalg import sqrtm
from skimage.transform import resize
from pandas.core.frame import DataFrame
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input

from privkit.data import FacialData
from privkit.metrics import Metric
from privkit.utils import face_utils as fu


class FID(Metric):
    METRIC_ID = "frechet_inception_distance"
    METRIC_NAME = "Frechét Inception Distance"
    METRIC_INFO = "The FID measures the quality of generated images quantitatively with the similarity " \
                  "between the distribution of features extracted from real and generated images. This " \
                  "implementation is based on [https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/]."
    DATA_TYPE_ID = [FacialData.DATA_TYPE_ID]

    def __init__(self, probe_path: str, baseline_path: str):
        """
        Initializes the Frechét Inception Distance (FID) metric with the specified parameters

        :param str probe_path: directory path where the probe identities images are located
        :param str baseline_path: directory path where the baseline identity images are located
        """
        super().__init__()
        self.probe_path = probe_path
        self.baseline_path = baseline_path

    def execute(self):
        """
        Executes the FID metric

        :return: dictionary with the computed metric
        """
        imgs_clear = np.array([plt.imread(file)[:, :, :3] for file in fu.list_files(self.baseline_path, 'png')])
        imgs_clear = self.scale_images(imgs_clear, (299, 299, 3))
        imgs_clear = imgs_clear.astype('float32')
        imgs_clear = self.scale_images(imgs_clear, (299, 299, 3))
        imgs_clear = preprocess_input(imgs_clear)

        imgs_anon = np.array([plt.imread(file)[:, :, :3] for file in fu.list_files(self.probe_path, 'png')])
        imgs_anon = self.scale_images(imgs_anon, (299, 299, 3))
        imgs_anon = imgs_anon.astype('float32')
        imgs_anon = self.scale_images(imgs_anon, (299, 299, 3))
        imgs_anon = preprocess_input(imgs_anon)

        fid = self.calculate_fid(imgs_clear, imgs_anon)
        return {'fid': fid}

    @staticmethod
    def scale_images(images: list, new_shape: tuple):
        """
        Returns a list of resized images

        :param list images: the images to be resized
        :param tuple new_shape: the shape of the new resized images
        :return: a list with resized images
        """
        images_list = list()
        for image in images:
            # resize with nearest neighbor interpolation
            new_image = resize(image, new_shape, 0)
            # store
            images_list.append(new_image)
        return asarray(images_list)

    @staticmethod
    def calculate_fid(images1: DataFrame, images2: DataFrame):
        """
        Returns the FID value between two image collections with the InceptionV3 model

        :param pd.DataFrame images1: the first image collection
        :param pd.DataFrame images2: the second image collection
        :return: the FID value 
        """
        model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
        # calculate activations
        act1 = model.predict(images1)
        act2 = model.predict(images2)
        # calculate mean and covariance statistics
        mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
        # calculate sum squared difference between means
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        # calculate sqrt of product between cov
        covmean = sqrtm(sigma1.dot(sigma2))
        # check and correct imaginary numbers from sqrt
        if iscomplexobj(covmean):
            covmean = covmean.real
        # calculate score
        fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

    def plot(self):
        pass
