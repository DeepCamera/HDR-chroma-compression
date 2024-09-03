"""Test script for HDR chroma compression of RGB images (paired image-to-image translation) using generative adversarial networks.
Adapted from: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix. Download and use the original implementation for alternative models
and scripts to prepare your own custom dataset.

Once you have trained your model with train.py, you can use this script to test the model. At a minimum, you need to specify 
the dataset path ('--dataroot'). It is also suggested to specify the number of test images ('--num_test').
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.
See the files within the 'options' folder for all available parameters.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example:
    python test.py --dataroot ./datasets/hdr --name hdr_exp1 --num_test 100

See further information at: ..................README.md.................... #######################################
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import time  # XM 11/05/2023
import csv  # XM 11/05/2023
import torch  # XM 11/05/2023


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.

    file = open(os.path.join(opt.results_dir, opt.name, 'time_cpu.csv'), 'w')  # XM 11/05/2023
    file2 = open(os.path.join(opt.results_dir, opt.name, 'time_cuda.csv'), 'w')  # XM 11/05/2023
    t_start2 = torch.cuda.Event(enable_timing=True)
    t_stop2 = torch.cuda.Event(enable_timing=True)

    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break

        t_start = time.perf_counter()  # XM 11/05/2023
        t_start2.record()
        
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results

        t_stop = time.perf_counter()  # XM 11/05/2023
        t_stop2.record()
        torch.cuda.synchronize()

        file.write(str(t_stop - t_start))
        file.write("\n")
        file2.write(str(t_start2.elapsed_time(t_stop2) / 1000))
        file2.write("\n")

        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    webpage.save()  # save the HTML

    file.close()  # XM 11/05/2023
    file2.close()  # XM 11/05/2023