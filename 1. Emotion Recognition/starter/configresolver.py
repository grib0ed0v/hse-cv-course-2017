import configparser
import logging

from preprocessor.processor.impl.noise import NoiseProcessor
from preprocessor.processor.impl.tonal import TonalProcessor


class ConfigResolver:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('./resources/config.ini')

    def get_processor_chain(self):
        chain_description = self.config.get('ProcessorChain', 'chain', fallback=None)

        if chain_description is None:
            logging.warning('No processor chain configured! Image will be sent to CNN without preprocessing.')
        else:
            return self.__build_processor_chain(chain_description)

    def __build_processor_chain(self, chain_description):
        chain = list()
        for processor_name in chain_description.split():
            if processor_name == 'NoiseProcessor':
                chain.append(self.__build_noise_processor())
            elif processor_name == 'TonalProcessor':
                chain.append(self.__build_tonal_processor())
            else:
                logging.error('No such processor with name: ' + processor_name)

        if len(chain) == 0:
            logging.warning('No processor chain configured! Image will be sent to CNN without preprocessing.')
        return chain

    def __build_noise_processor(self):
        h = self.config.getint('NoiseProcessor', 'h', fallback=10)
        hColor = self.config.getint('NoiseProcessor', 'hColor', fallback=10)
        templateWindowSize = self.config.getint('NoiseProcessor', 'templateWindowSize', fallback=7)
        searchWindowSize = self.config.getint('NoiseProcessor', 'searchWindowSize', fallback=21)
        return NoiseProcessor(h, hColor, templateWindowSize, searchWindowSize)

    def __build_tonal_processor(self):
        gamma = self.config.getfloat('TonalProcessor', 'gamma', fallback=1)
        return TonalProcessor(gamma)