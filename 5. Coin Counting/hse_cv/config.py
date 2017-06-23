import configparser

config = configparser.ConfigParser()
config.read('config.ini')
config.sections()
