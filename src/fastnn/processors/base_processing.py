from abc import ABC, abstractmethod


class Processor(ABC):
    """Abstract class for `Processor` objects in fastnn for the purpose of processing tensor objects to and from human interfacing data."""

    @abstractmethod
    def process(self):
        """ Returns a dataset at the minimum """
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def process_batch(self):
        """ """
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def process_output(self):
        """ """
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def process_output_batch(self):
        """ """
        raise NotImplementedError("Please Implement this method")
