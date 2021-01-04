import numpy as np
import os
import os.path as osp
from tqdm import tqdm
import shutil
import random
import torch
from torch.backends import cudnn
from torch.utils import data
import torch.optim as optim
import timeit
from torch.nn import functional as F
import time
from PIL import Image

from data.nyudv2 import NYUDataset_val_full
from utils.metrics import IOUMetric
from utils.utils import get_currect_time
from utils.encoding import DataParallelModel, DataParallelCriterion
from utils.log import Visualizer, Log
from utils.optim import adjust_learning_rate
from utils.misc import print_cuda_statistics
from agents.base import BaseAgent
from utils.utils import predict_multiscale, get_palette

class SGNetAgent(BaseAgent):
    """
    This class will be responsible for handling the whole process of our architecture.
    """
    def __init__(self, config):
        super().__init__(config)
        ## Select network
        if config.spatial_information == 'depth' and config.os == 16 and config.network == "SGNet" and config.mode != "measure_speed":
            from graphs.models.SGNet.SGNet import SGNet
        elif config.spatial_information == 'depth' and config.os == 16 and config.network == "SGNet":
            from graphs.models.SGNet.SGNet_fps import SGNet
        elif config.spatial_information == 'depth' and config.os == 16 and config.network == "SGNet_ASPP" and config.mode != "measure_speed":
            from graphs.models.SGNet.SGNet_ASPP import SGNet
        elif config.spatial_information == 'depth' and config.os == 16 and config.network == "SGNet_ASPP":
            from graphs.models.SGNet.SGNet_ASPP_fps import SGNet

        random.seed(self.config.seed)
        os.environ['PYTHONHASHSEED'] = str(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)
        cudnn.enabled = True
        cudnn.benchmark = True
        cudnn.deterministic = True
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
        # create data loader
        if config.dataset == "NYUD":
            self.testloader = data.DataLoader(NYUDataset_val_full(self.config.val_list_path),
                                         batch_size=1, shuffle=False, pin_memory=True)
        # Create an instance from the Model
        self.logger.info("Loading encoder pretrained in imagenet...")
        self.model = SGNet(self.config.num_classes)
        print(self.model)

        self.model.cuda()
        self.model.train()
        self.model.float()
        print(config.gpu)
        if config.mode != 'measure_speed':
            self.model = DataParallelModel(self.model, device_ids=[0])
            print('parallel....................')

        total = sum([param.nelement() for param in self.model.parameters()])
        print('  + Number of params: %.2fM' % (total / 1e6))
        print_cuda_statistics()

    def load_checkpoint(self, filename):
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['state_dict'])

            # self.optimizer.load_state_dict(checkpoint['optimizer'])

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                  .format(filename, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            self.logger.info("**First time to train**")

    def run(self):
        """
        This function will the operator
        :return:
        """
        assert self.config.mode in ['train', 'test', 'measure_speed', 'train_iters']
        try:
            if self.config.mode == 'test':
                self.test()
            elif self.config.mode == 'measure_speed':
                with torch.no_grad():
                    self.measure_speed(input_size=[1, 3, 480, 640])
        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def test(self):

        tqdm_batch = tqdm(self.testloader, total=len(self.testloader),
                          desc="Testing...")
        self.model.eval()
        metrics = IOUMetric(self.config.num_classes)
        loss_val = 0
        metrics = IOUMetric(self.config.num_classes)
        palette = get_palette(256)
        # if (not os.path.exists(self.config.output_img_dir)):
        #     os.mkdir(self.config.output_img_dir)
        # if (not os.path.exists(self.config.output_gt_dir)):
        #     os.mkdir(self.config.output_gt_dir)
        if (not os.path.exists(self.config.output_predict_dir)):
            os.mkdir(self.config.output_predict_dir)
        self.load_checkpoint(self.config.trained_model_path)
        index = 0
        for batch_val in tqdm_batch:
            image = batch_val['image'].cuda()
            label = batch_val['seg'].cuda()
            label = torch.squeeze(label, 1).long()
            HHA = batch_val['HHA'].cuda()
            depth = batch_val['depth'].cuda()
            size = np.array([label.size(1), label.size(2)])
            input_size = (label.size(1), label.size(2))

            with torch.no_grad():
                if self.config.ms:
                    output = predict_multiscale(self.model, image, depth, input_size, [0.8, 1.0, 2.0],
                                                self.config.num_classes, False)
                else:
                    output = predict_multiscale(self.model, image, depth, input_size, [1.0],
                                                self.config.num_classes, False)
                seg_pred = np.asarray(np.argmax(output, axis=2), dtype=np.int)
                output_im = Image.fromarray(np.asarray(np.argmax(output, axis=2), dtype=np.uint8))
                output_im.putpalette(palette)
                output_im.save(self.config.output_predict_dir + '/' + str(index) + '.png')
                seg_gt = np.asarray(label[0].cpu().numpy(), dtype=np.int)

                ignore_index = seg_gt != 255
                seg_gt = seg_gt[ignore_index]
                seg_pred = seg_pred[ignore_index]

                metrics.add_batch(seg_pred, seg_gt, ignore_index=255)

                index = index + 1
        acc, acc_cls, iu, mean_iu, fwavacc = metrics.evaluate()
        print({'meanIU': mean_iu, 'IU_array': iu, 'acc': acc, 'acc_cls': acc_cls})
        pass

    def finalize(self):
        """
        Finalize all the operations of the 2 Main classes of the process the operator and the data loader
        :return:
        """
        # TODO
        pass
    def measure_speed(self, input_size, iteration=500):
        """
        Measure the speed of model
        :return: speed_time
                 fps
        """
        self.model.eval()
        input = torch.randn(*input_size).cuda()
        depth = torch.randn(*input_size).cuda()
        HHA = torch.randn(*input_size).cuda()

        for _ in range(100):
            self.model(input, depth)
        print('=========Speed Testing=========')
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()

        for _ in range(iteration):
            x = self.model(input, depth)
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        speed_time = elapsed_time / iteration * 1000
        fps = iteration / elapsed_time
        print(iteration)
        print('Elapsed Time: [%.2f s / %d iter]' % (elapsed_time, iteration))
        print('Speed Time: %.2f ms / iter   FPS: %.2f' % (speed_time, fps))
        return speed_time, fps

