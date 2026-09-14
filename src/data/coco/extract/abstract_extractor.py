from abc import ABC, abstractmethod
from pathlib import Path

# This file's location → walk up to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # adjust .parent count
DATA_DIR = PROJECT_ROOT / "data"

class AbstractExtractor(ABC):

    @abstractmethod
    def extract(self, data):
        """
        Abstract method to extract data from the given input.
        This method should be implemented by subclasses.

        :param data: The input data to extract information from.
        :return: Extracted information.
        """
        pass