from abc import ABC, abstractmethod

class AbstractExtractor(ABC):

    @abstractmethod
    def extract(self):
        """
        Abstract method to extract data from the given input.
        This method should be implemented by subclasses.

        :param data: The input data to extract information from.
        :return: Extracted information.
        """
        pass