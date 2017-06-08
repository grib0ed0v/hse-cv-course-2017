from abc import ABCMeta, abstractmethod


class AbstractProcessor:

    __metaclass__ = ABCMeta

    @abstractmethod
    def process(self, image):
        pass
