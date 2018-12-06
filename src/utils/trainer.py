from __future__ import print_function, division, unicode_literals

import os
import os.path as pt
import time
from datetime import datetime
from torch.autograd import Variable
import logging
import sys
import json
import torch
from utils import save
from torch.optim import lr_scheduler
import utils.classmetric as classmetric
import torch
from utils.logger import Logger, VisdomLogger

#from .nvidia_tools import get_gpu_memory_map, log_gpu_statistics
import utils.nvidia_tools as nvidia_tools
from .image import tiff_to_nd_array
from utils.nvidia_tools import push_ngc_telemetry
import matplotlib.pyplot as plt
import numpy as np
import cv2

USE_MULTI_LOSS = False
LOG_NGC_DICT = False
LOG_NGC_EVERY_N_ITERATIONS = 10
SNAPSHOT_EVERY_EPOCHS = 1
GPU_USAGE_EVERY_N_ITERATIONS = 20
PRINT_EVERY_N_ITERATIONS = 1

# added new pandas-based csv logger and visdom integration
USE_NEW_LOGGERS=False

PRINT_FORMAT_TRAIN = \
    'TRAIN {num:8d} iter, {freq:.2f} iter/s, {metric}'
PRINT_FORMAT_VAL =\
    'VAL {num:8d} iter, {freq:.2f} iter/s, {metric}'


def log_info(**kwargs):
    logging.info(json.dumps(dict(**kwargs)))


def nicetime(seconds):
    hours = seconds // 3600
    minutes = seconds % 3600 // 60
    seconds = seconds % 60
    if hours:
        return '%d:%02d:%02d hours' % (hours, minutes, seconds)
    elif minutes:
        return '%d:%02d minutes' % (minutes, seconds)
    else:
        return '%d seconds' % seconds


def dict2string(d, key='%s', value='%.5f'):
    fmt = key + '=' + value
    return ', '.join(fmt % i for i in d.items())


def get_logfile(
        prefix='',
        dateformat='%Y-%m-%dt%H-%M-%S',
        pathformat='%s%s.log'
):
    s = datetime.strftime(datetime.now(), dateformat)
    return pathformat % (prefix, s)


def print_over(*args, **kwargs):
    end = kwargs.pop('end', '\n')
    kwargs['end'] = ''
    flush = kwargs.pop('flush', False)
    stream = kwargs.pop('file', sys.stdout)
    # return cursor to front and print
    print('\r', *args, **kwargs)
    # clear rest of the line
    print('\033[K', end=end)
    if flush:
        stream.flush()

def make_one_hot(labels, C=2):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.

    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    one_hot = torch.zeros(labels.size(0), C, labels.size(2), labels.size(3)).long()
    one_hot.scatter_(1, labels, 1)

    return one_hot


class IntervalPrinter(object):
    def __init__(self,
                 interval=1,
                 formatstring='{num:12d} updates, {freq:.2f} updates/s',
                 printlines=False):
        self.interval = interval
        if not printlines:
            formatstring = formatstring
        self.formatstring = formatstring
        self.end = None if printlines else ''
        self.new_updates = 0
        self.total_updates = 0
        self.last_print = None
        self._maxlen = 0

    def update(self, **kwargs):
        now = time.time()
        self.new_updates += 1
        self.total_updates += 1
        if self.last_print is None:
            self.last_print = now
        if now - self.last_print > self.interval:
            seconds = now - self.last_print
            print_over(self.formatstring.format(
                num=self.total_updates,
                freq=self.new_updates / seconds,
                **kwargs
            ), end=self.end, flush=True)
            self.last_print = now
            self.new_updates = 0

    def print_total_updates(self):
        print_over('\r%d samples written' % self.total_updates)


class AverageMetric(object):
    def __init__(self):
        self.total = 0
        self.iterations = 0

    def __call__(self, loss):
        self.iterations += 1
        self.total += loss
        return self.total / self.iterations

"""
class PolyPolicy(lr_scheduler.LambdaLR):
    def __init__(self, optimizer, num_epochs=1, power=0.5, last_epoch=-1):
        self.power = power
        self.num_epochs = num_epochs
        self.base_lrs = [g['lr'] for g in optimizer.param_groups]
        lr_scheduler.LambdaLR.__init__(self, optimizer, 1, last_epoch)

    def get_lr(self):
        i = self.last_epoch
        p = self.power
        n = self.num_epochs
        return [base_lr * (1 - i/n) ** p for base_lr in self.base_lrs]
"""
def tensor_to_variable(tensor):
    if torch.cuda.is_available():
        return Variable(tensor).cuda()
    else:
        return Variable(tensor)

class Trainer(object):
    def __init__(
            self,
            network,
            optimizer,
            scheduler,
            loss,
            train_iter,
            val_iter,
            outdir,
            visdom_environment,
            smoketest=False
    ):
        self.state = {
            'network': network,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'loss': loss,
            'train_iter': train_iter,
            'val_iter': val_iter,
            'train_loss': [],
            'train_metric': [],
            'val_loss': [],
            'val_metric': [],
            'outdir': outdir,
            'iterations': 0,
        }
        self.hooks = {}
        if not pt.exists(outdir):
            os.makedirs(outdir)
        logging.basicConfig(
            filename=pt.join(outdir, get_logfile()),
            level=logging.DEBUG,
            format='%(asctime)s - %(message)s',
        )

        self.smoketest = smoketest
        self.logger = Logger(columns=["loss"], modes=["train"], csv=os.path.join(outdir,visdom_environment+".csv"))
        self.vizlogger = VisdomLogger(env=visdom_environment,use_incoming_socket=False) # server='http://129.187.92.97',port=1234,log_to_filename="visdom.log",

    def hook(self, name):
        if name in self.hooks:
            self.hooks[name](self.state)

    def train(self, num_epochs, start_epoch=0):
        try:
            self.hook('on_start_training')
            scheduler = self.state['scheduler']

            # Print memory usage for GPU before training starts
            #if torch.cuda.is_available():
                #nvidia_tools.log_gpu_statistics()

            for epoch in range(start_epoch + 1, num_epochs + 1):
                self.logger.update_epoch(epoch)
                # Print current memory usage for GPU
                # print(torch.cuda.memory_allocated())
                print('Epoch', epoch)
                self.print_info()
                a = time.time()
                scheduler.step()
                train_metrics = self.train_epoch(epoch)
                print_over('TRAIN', dict2string(train_metrics),nicetime(time.time() - a))
                a = time.time()
                with torch.no_grad():
                    val_metrics = self.validate_epoch(epoch)
                print_over('VAL', dict2string(val_metrics),
                           nicetime(time.time() - a))
                if epoch % SNAPSHOT_EVERY_EPOCHS == 0:
                    self.snapshot(epoch)

                if self.logger is not None:
                    self.logger.save_csv()

            self.hook('on_end_training')
            return self.state
        except Exception as e:
            logging.exception(e)
            raise

    def _param_groups(self):
        return self.state['optimizer'].param_groups

    def _lrs(self):
        return [g['lr'] for g in self._param_groups()]

    def print_info(self):
        s = 'learning rates ' + (', '.join(map(str, self._lrs())))
        print(s)
        log_info(lrs=self._lrs())

    def snapshot(self, epoch):
        path = pt.join(self.state['outdir'], 'epoch_%02d_classes_02.pth' % (epoch))
        save(
            path,
            self.state['iterations'],
            self.state['network'],
            self.state['optimizer']
        )

    def train_epoch(self, epoch):
        self.logger.set_mode("train")
        printer = IntervalPrinter(1, PRINT_FORMAT_TRAIN)
        network = self.state['network']
        loss = self.state['loss']
        optimizer = self.state['optimizer']
        metric = classmetric.ClassMetric()
        train_metric = dict()
        self.hook('train_start')

        dataloader = self.state['train_iter']
        for iteration, data in enumerate(dataloader):
            optimizer.zero_grad()

            tile, input, target_tensor = data
            target = tensor_to_variable(target_tensor[0])

            for key in input.keys():
                input[key] = tensor_to_variable(input[key])

            output = network.forward(input)

            # pspnet_fused return tuples if intermediate classification maps -> take last, that is the fused one
            if type(output)==tuple:
                output = output[-1]

            # force the output label map to match the target dimensions
            b, h, w = target.shape
            output = torch.nn.functional.upsample(output, size=(h, w), mode='bilinear')
            l = loss(output, target.long())

            l.backward()
            optimizer.step()

            train_metric = metric(target, output)
            train_metric['loss'] = l.data.cpu().numpy()

            if self.logger is not None:
                self.logger.log(train_metric.copy(), iteration)

            if iteration % PRINT_EVERY_N_ITERATIONS == 0:
                printer.update(metric=dict2string(train_metric))

                if self.vizlogger.is_available:
                    self.vizlogger.plot_steps(self.logger.get_data())

            if iteration % LOG_NGC_EVERY_N_ITERATIONS == 0 and LOG_NGC_DICT:
                nvidia_tools.log_ngc_dict(train_metric, prefix="train")

            if iteration % GPU_USAGE_EVERY_N_ITERATIONS == 0 and \
                    torch.cuda.is_available() and LOG_NGC_DICT:
                nvidia_tools.log_gpu_statistics()

            if iteration % LOG_NGC_EVERY_N_ITERATIONS == 0 and not LOG_NGC_DICT:
                push_ngc_telemetry("train_iou_building", train_metric["iou_building"])
                push_ngc_telemetry("train_loss", train_metric["loss"])

            if self.smoketest:
                print("Smoketest! Stopping training early after one iteration")
                break

        self.hook('train_end')

        return train_metric

    def validate_epoch(self, epoch):
        self.logger.set_mode("test")
        printer = IntervalPrinter(1, PRINT_FORMAT_VAL)
        network = self.state['network']
        loss = self.state['loss']
        # average_metric = AverageMetric()
        metric = classmetric.ClassMetric()
        # val_loss = dict()
        val_metric = dict()
        self.hook('validation_start')

        dataloader = self.state['val_iter']
        for iteration, data in enumerate(dataloader):

            tile, input, target_tensor = data
            target = tensor_to_variable(target_tensor[0])

            for key in input.keys():
                input[key] = tensor_to_variable(input[key])

            output = network.forward(input)

            # pspnet_fused return tuples if intermediate classification maps -> take last, that is the fused one
            if type(output) == tuple:
                output = output[-1]

            # force the output label map to match the target dimensions
            b, h, w = target.shape
            output = torch.nn.functional.upsample(output, size=(h, w), mode='bilinear')
            l = loss(output, target.long())

            val_metric = metric(target, output)
            val_metric['loss'] = l.data.cpu().numpy()  # loss_metric(l.data[0]).cpu().numpy()

            if self.logger is not None:
                self.logger.log(val_metric.copy(), iteration)

            if iteration % PRINT_EVERY_N_ITERATIONS == 0:
                printer.update(metric=dict2string(val_metric))

                if self.vizlogger.is_available:
                    self.vizlogger.plot_steps(self.logger.get_data())
                    self.vizlogger.plot_images(target=target.cpu().numpy(), output=output.cpu().numpy())

            if iteration % LOG_NGC_EVERY_N_ITERATIONS == 0 and LOG_NGC_DICT:
                nvidia_tools.log_ngc_dict(val_metric, prefix="val")

            if iteration % GPU_USAGE_EVERY_N_ITERATIONS == 0 and \
                    torch.cuda.is_available() and LOG_NGC_DICT:
                nvidia_tools.log_gpu_statistics()

            if iteration % LOG_NGC_EVERY_N_ITERATIONS == 0 and not LOG_NGC_DICT:
                push_ngc_telemetry("val_iou_building", val_metric["iou_building"])
                push_ngc_telemetry("val_loss", val_metric["loss"])

            if self.smoketest:
                print("Smoketest! Stopping validation early after one iteration")
                break

        if self.vizlogger.is_available:
            self.vizlogger.update(self.logger.get_data())
            self.vizlogger.plot_images(target=target.cpu().numpy(), output=output.cpu().numpy())

        self.hook('validation_end')

        return val_metric
