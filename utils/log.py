import os
import os.path as osp
from tensorboardX import SummaryWriter

import logging
from logging import Formatter
from logging.handlers import RotatingFileHandler

def setup_logging(log_dir):
    log_file_format = "[%(levelname)s] - %(asctime)s - %(name)s - : %(message)s in %(pathname)s:%(lineno)d"
    log_console_format = "[%(levelname)s]: %(message)s"

    # Main logger
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(Formatter(log_console_format))

    exp_file_handler = RotatingFileHandler('{}exp_debug.log'.format(log_dir), maxBytes=10**6, backupCount=5)
    exp_file_handler.setLevel(logging.DEBUG)
    exp_file_handler.setFormatter(Formatter(log_file_format))

    exp_errors_file_handler = RotatingFileHandler('{}exp_error.log'.format(log_dir), maxBytes=10**6, backupCount=5)
    exp_errors_file_handler.setLevel(logging.WARNING)
    exp_errors_file_handler.setFormatter(Formatter(log_file_format))

    main_logger.addHandler(console_handler)
    main_logger.addHandler(exp_file_handler)
    main_logger.addHandler(exp_errors_file_handler)

class Visualizer():
    """
    Visualizer
    :param:
        config:
    """
    def __init__(self, config):
        self.writer = SummaryWriter(osp.join(config.snapshot_dir, config.exp_name + config.time))
        self.config = config
    def add_scalar(self, name, x, y):
        self.writer.add_scalar(name, x, y)
    def add_image(self, name, image, iter):
        self.writer.add_image(name, image, iter)

class Log():
    """
    Log
    :param:
        config:
    """
    def __init__(self, config):
        self.log_path = osp.join(config.snapshot_dir, config.exp_name + config.time)
        self.log = open(osp.join(self.log_path, 'log_train.txt'), 'w')
        self.config = config
        setup_logging(self.log_path)
        logging.getLogger().info("Hi, This is root.")
        logging.getLogger().info("After the configurations are successfully processed and dirs are created.")
        logging.getLogger().info("The pipeline of the project will begin now.")

    def record_sys_param(self):
        self.log.write(str(self.config) + '\n')

    def record_file(self):
        os.system('cp %s %s'%(self.config.model_file, self.log_path))
        os.system('cp %s %s'%(self.config.agent_file, self.log_path))
        os.system('cp %s %s'%(self.config.config_file, self.log_path))
        os.system('cp %s %s' % (self.config.dataset_file, self.log_path))
        os.system('cp %s %s' % (self.config.transform_file, self.log_path))
        os.system('cp %s %s' % (self.config.module_file, self.log_path))

    def log_string(self, out_str):
        self.log.write(out_str + '\n')
        self.log.flush()
        print(out_str)