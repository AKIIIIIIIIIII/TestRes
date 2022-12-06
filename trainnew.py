"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from utils import sample, recon_criterion, save_checkpoint_sp,get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images, Timer, save_training_images, vgg_preprocess_color, weights_init, get_model_list, vgg_preprocess, load_vgg16, get_scheduler
import argparse
from torch.autograd import Variable
from trainer_norecon import MUNIT_Trainer, UNIT_Trainer
from VGGPytorch import VGGNet
import torch.backends.cudnn as cudnn
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import os
import sys
import tensorboardX
from tqdm import tqdm
import shutil
from networksold import AdaINGen, MsImageDis# VAEGen Discriminator,
from losses import VariationLoss
from texture_extractor import ColorShift
from surface_extractor import GuidedFilter
from structure_extractor import SuperPixel
import torch
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/edges2handbags_folder.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument("--resume", action="store_true")
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
opts = parser.parse_args()

cudnn.benchmark = False

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
display_size = config['display_size']
config['vgg_model_path'] = opts.output_path

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

# init
lr = config['lr']
# Initiate the networks
gen_a = AdaINGen(config['input_dim_a'], config['gen']).to(config["DEVICE"])  # auto-encoder for domain a
disc_surface = MsImageDis(config['input_dim_a'], config['dis']).to(config["DEVICE"])  # discriminator for surface
disc_texture = MsImageDis(config['input_dim_b'], config['dis']).to(config["DEVICE"])  # discriminator for texture
style_dim = config['gen']['style_dim']

# fix the noise used in sampling
display_size = int(config['display_size'])
s_a = torch.randn(display_size, style_dim, 1, 1).cuda()
s_b = torch.randn(display_size, style_dim, 1, 1).cuda()

# Setup the optimizers
beta1 = config['beta1']
beta2 = config['beta2']
dis_params = list(disc_surface.parameters()) + list(disc_texture.parameters())
gen_params = list(gen_a.parameters())
dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                lr=lr, betas=(beta1, beta2), weight_decay=config['weight_decay'])
gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                lr=lr, betas=(beta1, beta2), weight_decay=config['weight_decay'])
dis_scheduler = get_scheduler(dis_opt, config)
gen_scheduler = get_scheduler(gen_opt, config)

# Load VGG model if needed
if 'vgg_w' in config.keys() and config['vgg_w'] > 0:
#    vgg = load_vgg16(config['vgg_model_path'] + '/models')
#    vgg.eval()
#    for param in vgg.parameters():
#        param.requires_grad = False
    VGG19 = VGGNet(in_channels=3, VGGtype="VGG19", init_weights="vgg19-dcbb9e9d.pth", batch_norm=False, feature_mode=True)
    VGG19 = VGG19.to(config["DEVICE"])
    VGG19.eval()


train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config)
train_display_images_a = torch.stack([train_loader_a.dataset[i] for i in range(display_size)]).to(config["DEVICE"])
train_display_images_b = torch.stack([train_loader_b.dataset[i] for i in range(display_size)]).to(config["DEVICE"])
test_display_images_a = torch.stack([test_loader_a.dataset[i] for i in range(display_size)]).to(config["DEVICE"])
test_display_images_b = torch.stack([test_loader_b.dataset[i] for i in range(display_size)]).to(config["DEVICE"])

extract_structure = SuperPixel(config["DEVICE"], mode='simple')
extract_texture = ColorShift(config["DEVICE"], mode='uniform', image_format='rgb')
extract_surface = GuidedFilter()

L1_Loss = nn.L1Loss()
MSE_Loss = nn.MSELoss()  # went through the author's code and found him using LSGAN, LSGAN should gives better training
var_loss = VariationLoss(1)


gen_a.apply(weights_init(config['init']))
disc_texture.apply(weights_init('gaussian'))
disc_surface.apply(weights_init('gaussian'))

## Start training
# iterations = resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
it=0
if config["INI"]:
    for epoch in range(config["INI"]):
        loop = tqdm(train_loader_a, leave=True)
        losses = []

        for idx, (sample_photo) in enumerate(loop):
            sample_photo = sample_photo.to(config["DEVICE"])
            c_a, s_a_prime = gen_a.encode(sample_photo)
            reconstructed = gen_a.decode(c_a, s_a_prime)

            sample_photo_feature = VGG19(sample_photo)
            reconstructed_feature = VGG19(reconstructed)
            reconstruction_loss = L1_Loss(reconstructed_feature, sample_photo_feature.detach()) * 255

            losses.append(reconstruction_loss.item())

            gen_opt.zero_grad()

            reconstruction_loss.backward()
            gen_opt.step()
            train_writer.add_scalar('G_loss_INI', reconstruction_loss.data.cpu().numpy(), global_step=it)
            it = it+1
            loop.set_postfix(epoch=epoch)

        print('[%d/%d] - Recon loss: %.8f' % ((epoch + 1), config["INI"], torch.mean(torch.FloatTensor(losses))))
        save_training_images(torch.cat((sample_photo * 0.5 + 0.5, reconstructed * 0.5 + 0.5), axis=3),
                             epoch=epoch, step=0, dest_folder=output_directory, suffix_filename="initial_io")

step = 0
for epoch in range(max_iter):
    loop = tqdm(zip(train_loader_a, train_loader_b), leave=True)

    # Training
    for idx, (sample_photo, sample_cartoon) in enumerate(loop):
        sample_photo = sample_photo.to(config["DEVICE"])
        sample_cartoon = sample_cartoon.to(config["DEVICE"])
        # Train Discriminator
        fake_cartoon_c, fake_cartoon_s = gen_a.encode(sample_photo)
        s_a = Variable(torch.randn(sample_cartoon.size(0), style_dim, 1, 1).cuda())
        fake_cartoon = gen_a.decode(fake_cartoon_c, s_a)
        output_photo = extract_surface.process(sample_photo, fake_cartoon, r=1)

        # Surface Representation
        blur_fake = extract_surface.process(output_photo, output_photo, r=5, eps=2e-1)
        blur_cartoon = extract_surface.process(sample_cartoon, sample_cartoon, r=5, eps=2e-1)
        d_loss_surface = disc_surface.calc_dis_loss(blur_fake.detach(), blur_cartoon)

        # Textural Representation
        gray_fake, gray_cartoon = extract_texture.process(output_photo, sample_cartoon)
        d_loss_texture = disc_texture.calc_dis_loss(gray_fake.detach(), gray_cartoon)

        d_loss_total = d_loss_surface + d_loss_texture

        dis_opt.zero_grad()
        d_loss_total.backward()
        dis_opt.step()

        # ===============================================================================

        # Train Generator
        fake_cartoon_c, fake_cartoon_s = gen_a.encode(sample_photo)
        s_a = Variable(torch.randn(sample_cartoon.size(0), style_dim, 1, 1).cuda())
        recon = gen_a.decode(fake_cartoon_c, fake_cartoon_s)
        _, s_cartoon = gen_a.encode(sample_cartoon)
        output_photo = extract_surface.process(sample_photo, fake_cartoon, r=1)

        # Guided Filter
        blur_fake = extract_surface.process(output_photo, output_photo, r=5, eps=2e-1)
        g_loss_surface = config["LAMBDA_SURFACE"] * disc_surface.calc_gen_loss(blur_fake)

        # Color Shift
        gray_fake, = extract_texture.process(output_photo)
        g_loss_texture = config["LAMBDA_TEXTURE"] * disc_texture.calc_gen_loss(gray_fake)

        # SuperPixel
        input_superpixel = extract_structure.process(output_photo.detach())
        vgg_output = VGG19(output_photo)
        _, c, h, w = vgg_output.shape
        vgg_superpixel = VGG19(input_superpixel)
        superpixel_loss = config["LAMBDA_STRUCTURE"] * L1_Loss(vgg_superpixel, vgg_output) * 255 / (c * h * w)
        # ^ Original author used CaffeVGG model which took (0-255)BGR images as input,
        # while we used PyTorch model which takes (0-1)BGB images as input. Therefore we multply the l1 with 255.

        # Content Loss
        vgg_photo = VGG19(sample_photo)
        content_loss = config["LAMBDA_CONTENT"] * L1_Loss(vgg_photo, vgg_output) * 255 / (c * h * w)
        # ^ Original author used CaffeVGG model which took (0-255)BGR images as input,
        # while we used PyTorchVGG model which takes (0-1)BGB images as input. Therefore we multply the l1 with 255.

        # encode again
        c_aba, s_aba_prime = gen_a.encode(fake_cartoon)
        # reconstruction loss
        loss_gen_recon_c_a = config["recon_c_w"] * recon_criterion(c_aba, fake_cartoon_c)
        loss_gen_recon_x_a = config["recon_x_w"] * recon_criterion(recon, sample_photo)
        loss_gen_recon_s_a = config["recon_s_w"] * recon_criterion(s_aba_prime, fake_cartoon_s)

        # Variation Loss
        tv_loss = config["LAMBDA_VARIATION"] * var_loss(output_photo)

        # NOTE Equation 6 in the paper
        g_loss_total = g_loss_surface + g_loss_texture + superpixel_loss + content_loss + tv_loss + loss_gen_recon_c_a + loss_gen_recon_x_a + loss_gen_recon_s_a

        gen_opt.zero_grad()
        g_loss_total.backward()
        gen_opt.step()

        # ===============================================================================

        train_writer.add_scalar('D_loss_surface', d_loss_surface.data.cpu().numpy(), global_step=step)
        train_writer.add_scalar("D_loss_texture", d_loss_texture.data.cpu().numpy(), global_step=step)
        train_writer.add_scalar('G_loss_surface', g_loss_surface.data.cpu().numpy(), global_step=step)
        train_writer.add_scalar("G_loss_texture", g_loss_texture.data.cpu().numpy(), global_step=step)
        train_writer.add_scalar('G_loss_superpixel', superpixel_loss.data.cpu().numpy(), global_step=step)
        train_writer.add_scalar('G_loss_content', content_loss.data.cpu().numpy(), global_step=step)
        train_writer.add_scalar('G_loss_recon_ca', loss_gen_recon_c_a.data.cpu().numpy(), global_step=step)
        train_writer.add_scalar('G_loss_recon_xa', loss_gen_recon_x_a.cpu().numpy(), global_step=step)
        train_writer.add_scalar('G_loss_recon_sa', loss_gen_recon_s_a.cpu().numpy(), global_step=step)
        train_writer.add_scalar('G_loss_tv', tv_loss.data.cpu().numpy(), global_step=step)

        if (step+1) % config["image_save_iter"] == 0:
            save_training_images(
                torch.cat((blur_fake * 0.5 + 0.5, gray_fake * 0.5 + 0.5, input_superpixel * 0.5 + 0.5), axis=3),
                epoch=epoch, step=step, dest_folder=output_directory, suffix_filename="photo_rep")

            save_training_images(
                torch.cat((sample_photo * 0.5 + 0.5, fake_cartoon * 0.5 + 0.5, output_photo * 0.5 + 0.5), axis=3),
                epoch=epoch, step=step, dest_folder=output_directory, suffix_filename="io")

        step += 1
        loop.set_postfix(step=step, epoch=epoch + 1)

    if (epoch+1) % config['snapshot_save_iter'] == 0:
        save_checkpoint_sp(disc_texture,disc_surface,gen_a,dis_opt,gen_opt,epoch,checkpoint_directory)

    if (epoch+1) % config['image_display_iter'] == 0:
        with torch.no_grad():
            image_outputs = sample(train_display_images_a, train_display_images_b, gen_a)
        write_2images(image_outputs, display_size, image_directory, 'train_current')

if config.SAVE_MODEL:
        save_checkpoint_sp(disc_texture, disc_surface, gen_a, dis_opt, gen_opt, epoch, checkpoint_directory, "latest")

