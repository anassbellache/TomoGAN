import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.image_utils import save2image
import sys, os, time, argparse, shutil, scipy, h5py, glob
from models.gan_model import TomoGAN
from data_processor import bkgdGen, gen_train_batch_bg, get1batch4test
import torchvision.models as models
from options.train_options import TrainOptions
from skimage.measure import compare_ssim
import mlflow

def eval_metrics(actual, pred):
    ssim = compare_ssim(actual, pred)
    mse = np.mean((actual - pred) ** 2) 
    if(mse == 0): 
        psnr = 100
    else:
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return ssim, psnr

args = TrainOptions('./config.yaml')
#os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable printing INFO, WARNING, and ERROR

mb_size = args.batch_size
img_size = args.image_size
in_depth = args.depth
disc_iters, gene_iters = args.itd, args.itg
lambda_mse, lambda_adv, lambda_perc = args.lmse, args.ladv, args.lperc
EPOCHS = 40001



itr_out_dir = args.name + '-itrOut'
if os.path.isdir(itr_out_dir): 
    shutil.rmtree(itr_out_dir)
os.mkdir(itr_out_dir) # to save temp output


if not os.path.exists("outputs"):
        os.makedirs("outputs")

sys.stdout = open('%s/%s' % (itr_out_dir, 'iter-prints.log'), 'w') 
print('X train: {}\nY train: {}\nX test: {}\nY test: {}'.format(args.xtrain, args.ytrain, args.xtest, args.ytest))

#with open("outputs/iter_logs.txt", "w") as f:
#    f.write('X train: {}\nY train: {}\nX test: {}\nY test: {}'.format(args.xtrain, args.ytrain, args.xtest, args.ytest))

#print('X train: {}\nY train: {}\nX test: {}\nY test: {}'.format(args.xtrain, args.ytrain, args.xtest, args.ytest))

mb_data_iter = bkgdGen(data_generator=gen_train_batch_bg(x_fn=args.xtrain, \
                                      y_fn=args.ytrain, mb_size=mb_size, \
                                      in_depth=in_depth, img_size=img_size), 
                                      max_prefetch=16)  

tomogan = TomoGAN(args) 

with mlflow.start_run() as run:
    mlflow.log_param("batch_size", mb_size)
    mlflow.log_param("depth", in_depth)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("image_size", img_size)
    mlflow.log_param("generator_iterations", gene_iters)
    mlflow.log_param("discrimiator_iterations", disc_iters)
    
    for epoch in range(EPOCHS):
        time_git_st = time.time()
        for _ge in range(gene_iters):
            X_mb, y_mb = mb_data_iter.next()
            tomogan.set_input((X_mb, y_mb))
            tomogan.backward_G()

        itr_prints_gen = '[Info] Epoch: %05d, gloss: %.2f (mse%.3f, adv%.3f, perc:%.3f), gen_elapse: %.2fs/itr' % (\
                     epoch, tomogan.loss_G, tomogan.loss_G_MSE*lambda_mse, tomogan.loss_G_GAN*lambda_adv, \
                         tomogan.loss_G_Perc*lambda_perc, (time.time() - time_git_st)/gene_iters, )
        time_dit_st = time.time()

        for de in range(disc_iters):
            X_mb, y_mb = mb_data_iter.next()
            tomogan.set_input((X_mb, y_mb))
            tomogan.backward_D()
        
        with open("outputs/iter_logs.txt", "w") as f:
            print('%s; dloss: %.2f (r%.3f, f%.3f), disc_elapse: %.2fs/itr, gan_elapse: %.2fs/itr' % (itr_prints_gen,\
            tomogan.loss_D, tomogan.loss_D_real.detach().cpu().numpy().mean(), tomogan.loss_D_fake.detach().cpu().numpy().mean(), \
            (time.time() - time_dit_st)/disc_iters, time.time()-time_git_st))

    
        if epoch % (200//gene_iters) == 0:
            with torch.no_grad():
                X222, y222 = get1batch4test(x_fn=args.xtest, y_fn=args.ytest, in_depth=in_depth)
                tomogan.set_input((X222, y222))
                tomogan.forward()
                pred_img = tomogan.fake_C

                save2image(pred_img[0,0,:,:].detach().cpu().numpy(), '%s/it%05d.png' % (itr_out_dir, epoch))
                if epoch == 0: 
                    save2image(y222[0,0,:,:], '%s/gtruth.png' % (itr_out_dir))
                    save2image(X222[0,in_depth//2,:,:], '%s/noisy.png' % (itr_out_dir))
                    
    mlflow.log_artifacts(itr_out_dir)
    (ssim, psnr) = eval_metrics(y222[0,0,:,:], pred_img[0,0,:,:].detach().cpu().numpy())
    mlflow.log_metric("ssim", ssim)
    mlflow.log_metric("psnr", psnr)

sys.stdout.flush()

