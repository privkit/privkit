from typing import List
from abc import ABC, abstractmethod


class PPM(ABC):
    """
    PPM is an abstract class for a generic Privacy-Preserving Mechanism (PPM). Defines a series of methods common to all PPMs.
    Provides a general method to execute the PPM.
    Requires the definition of a PPM_ID, PPM_NAME, PPM_INFO, PPM_REF, DATA_TYPE_ID (of the data types this mechanism
    can be applied to), and METRIC_ID (of the metric that can be used to assess this PPM).
    """
    @property
    def PPM_ID(self) -> str:
        """Identifier of the PPM"""
        raise NotImplementedError

    @property
    def PPM_NAME(self) -> str:
        """Name of the PPM"""
        raise NotImplementedError

    @property
    def PPM_INFO(self) -> str:
        """Information about the PPM and how it works"""
        raise NotImplementedError

    @property
    def PPM_REF(self) -> str:
        """Reference to the work that proposes this PPM"""
        raise NotImplementedError

    @property
    def DATA_TYPE_ID(self) -> List[str]:
        """Identifiers of the data types that the PPM is applied to"""
        raise NotImplementedError

    @property
    def METRIC_ID(self) -> List[str]:
        """Identifiers of the metrics that can be used to assess the PPM"""
        raise NotImplementedError

    @abstractmethod
    def execute(self, *args):
        """Executes the privacy-preserving mechanism. This is specific to the mechanism"""
        pass
