import matplotlib.pyplot as plt

from pathlib import Path
from skimage.metrics import structural_similarity  

from privkit.data import FacialData
from privkit.metrics import Metric
from privkit.utils import face_utils as fu


class SSIM(Metric):
    METRIC_ID = "ssim"
    METRIC_NAME = "Structure Similarity Index (SSIM)"
    METRIC_INFO = "SSIM quantifies the similarity between two images by assessing their structural information and " \
                  "pixel-wise content. "
    DATA_TYPE_ID = [FacialData.DATA_TYPE_ID]

    def __init__(self, probe_path: str, baseline_path: str):
        """
        Initializes the SSIM metric with the specified parameters

        :param str probe_path: directory path where the probe identities images are located
        :param str baseline_path: directory path where the baseline identity images are located
        """
        super().__init__()
        self.probe_path = probe_path 
        self.baseline_path = baseline_path     

    def execute(self):
        """
        Executes the SSIM metric.

        :return: dictionary with the SSIM values
        """
        ssim = list()
        for identity_img in fu.list_files(self.probe_path, ('.png')):
            
            ID = Path(identity_img).stem
            file_path_baseline = self.baseline_path + f'/{ID}.png'
            img_clear = plt.imread(file_path_baseline)
            img_anon = plt.imread(identity_img)
            
            ssim_value = structural_similarity(img_clear, img_anon, channel_axis=-1)  # clean, noise, ...
            ssim.append(ssim_value)
        
        return {'ssim': ssim}
    
    def plot(self):
        pass
