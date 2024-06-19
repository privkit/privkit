from typing import List
from abc import ABC, abstractmethod


class Attack(ABC):
    """
    Attack is an abstract class for a generic attack. Defines methods common to all attacks.
    Provides a general function to execute the attack.
    Requires the definition of the ATTACK_ID, ATTACK_NAME, ATTACK_INFO, ATTACK_REF, DATA_TYPE_ID (of the data types
    this attack can be applied to), and METRIC_ID (of the metrics that can be used to assess this attack).
    """
    @property
    def ATTACK_ID(self) -> str:
        """Identifier of the attack"""
        raise NotImplementedError

    @property
    def ATTACK_NAME(self) -> str:
        """Name of the attack"""
        raise NotImplementedError

    @property
    def ATTACK_INFO(self) -> str:
        """Information about the attack and how it works"""
        raise NotImplementedError

    @property
    def ATTACK_REF(self) -> str:
        """Reference to the work that proposes this attack"""
        raise NotImplementedError

    @property
    def DATA_TYPE_ID(self) -> List[str]:
        """Identifiers of the data types that the attack is applied to"""
        raise NotImplementedError

    @property
    def METRIC_ID(self) -> List[str]:
        """Identifiers of the metrics that can be used to assess the attack"""
        raise NotImplementedError

    @abstractmethod
    def execute(self, *args):
        """Executes the attack mechanism. This is specific to the attack"""
        pass
