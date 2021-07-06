import os
import tqdm

from configs import Config
from data import DataGenerator
from kernelGAN import KernelGAN
from learner import Learner
from util import plot_train_results, plot_header_results, read_image, move2cpu, create_next_results_dir, post_process_k, save_kernels
import pickle

import torch
import scipy.io as sio


def train(conf):
    gan = KernelGAN(conf)
    if conf.load_weights_path != '':
        print('loading weights...')
        gan.G.load_state_dict(torch.load(conf.load_weights_path))
    learner = Learner()
    data = DataGenerator(conf, gan)
    for iteration in tqdm.tqdm(range(conf.max_iters), ncols=60):
        [g_in, d_in] = data.__getitem__(iteration)
        gan.train(g_in, d_in)
        learner.update(iteration, gan)
        if conf.save_results and iteration % conf.save_results_interval == 0:
            gan.calc_curr_k()
            real_kernel = move2cpu(gan.curr_k)
            final_kernel = post_process_k(gan.curr_k, n=gan.conf.n_filtering)
            save_kernels(real_kernel, final_kernel, conf, iteration)
    if conf.save_results:
        pickle.dump(gan.g_loss_tracker, open(os.path.join(conf.output_dir_path, 'loss_tracker.pkl'), "wb"))
        pickle.dump(gan.nn_tracker, open(os.path.join(conf.output_dir_path, 'nn_tracker.pkl'), "wb"))
        pickle.dump(learner, open(os.path.join(conf.output_dir_path, 'learner.pkl'), "wb"))
    final_kernel, real_kernel, loss_tracker, nn_tracker = gan.finish()
    learner_special_iterations = learner.similar_to_bicubic_iteration, learner.constraints_inserted_iteration
    return final_kernel, real_kernel, loss_tracker, nn_tracker, learner_special_iterations


def main():
    """The main function - performs kernel estimation (+ ZSSR) for all images in the 'test_images' folder"""
    import argparse
    # Parse the command line arguments
    prog = argparse.ArgumentParser()
    prog.add_argument('--input-dir', '-i', type=str, default='test_images', help='path to image input directory.')
    prog.add_argument('--output-dir', '-o', type=str, default='results', help='path to image output directory.')
    prog.add_argument('--X4', action='store_true', help='The wanted SR scale factor')
    prog.add_argument('--SR', action='store_true', help='when activated - ZSSR is not performed')
    prog.add_argument('--real', action='store_true', help='ZSSRs configuration is for real images')
    prog.add_argument('--noise_scale', type=float, default=1., help='ZSSR uses this to partially de-noise images')
    args = prog.parse_args()
    # Run the KernelGAN sequentially on all images in the input directory
    for filename in os.listdir(os.path.abspath(args.input_dir)):
        conf = Config().parse(create_params(filename, args))
        train(conf)
    prog.exit(0)


def create_params(filename, args):
    params = ['--input_image_path', os.path.join(args.input_dir, filename),
              '--output_dir_path', os.path.abspath(args.output_dir),
              '--noise_scale', str(args.noise_scale)]
    if args.X4:
        params.append('--X4')
    if args.SR:
        params.append('--do_ZSSR')
    if args.real:
        params.append('--real_image')
    return params


if __name__ == '__main__':
    main()


def my_create_conf(input_image_path, output_dir_path, num_iters=3000, new_loss=False, save_results=False, load_weights_path=''):
    params = ['--input_image_path', input_image_path,
              '--output_dir_path', output_dir_path,
              '--load_weights_path', load_weights_path,
              '--noise_scale', str(1),
              '--max_iters', str(num_iters)]
    if new_loss: params.append('--new_loss')
    if save_results: params.append('--save_results')
    conf = Config().parse(params)
    return conf


def my_main(input_image_indices=[30], input_kernel_indices=[0], num_iters=3000, old_loss=True, new_loss=False, save_results=False, load_weights_path=''):
    print("My main...")
    dataset_dir = '/home/labs/waic/itaian/Project/KernelNN_Dataset'
    results_dir = '/home/labs/waic/itaian/Project/KernelNN/results'
    output_dir_path = '' if not save_results else create_next_results_dir(results_dir)

    for image_num in input_image_indices:
        for kernel_num in input_kernel_indices:
            # get original image
            input_image_path = os.path.join(dataset_dir, 'lr_x2', 'im_{im}_ker_{ker}.png'.format(im=image_num, ker=kernel_num))
            input_image = read_image(input_image_path) / 255.
            kernelGT = sio.loadmat(os.path.join(dataset_dir, 'kernels', 'ker_{ker}.mat'.format(ker=kernel_num)))['ker']
            # plot header
            plot_header_results(image_num, kernel_num, input_image, kernelGT)

            new_losses = []
            if old_loss: new_losses.append(False)
            if new_loss: new_losses.append(True)
            for new_loss in new_losses:
                # create config
                conf = my_create_conf(input_image_path=input_image_path, output_dir_path=output_dir_path,
                                      num_iters=num_iters, new_loss=new_loss, save_results=save_results, load_weights_path=load_weights_path)
                # train the model
                final_kernel, real_kernel, loss_tracker, nn_tracker, learner_special_iterations = train(conf)
                # plot results
                plot_train_results(input_image, final_kernel, real_kernel, loss_tracker, nn_tracker, learner_special_iterations, new_loss)
    # clear cuda cache
    torch.cuda.empty_cache()
