"""Loss functions for training encoder."""
import tensorflow as tf
from dnnlib.tflib.autosummary import autosummary
import numpy as np


#----------------------------------------------------------------------------
# Convenience func that casts all of its arguments to tf.float32.

def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]


#----------------------------------------------------------------------------
# Encoder loss function .
def E_loss(E, G, D, perceptual_model, reals, feature_scale=0.00005, D_scale=0.1, perceptual_img_size=256):
    num_layers, latent_dim = G.components.synthesis.input_shape[1:3]
    latent_w = E.get_output_for(reals, is_training=True)
    latent_wp = tf.reshape(latent_w, [reals.shape[0], num_layers, latent_dim])
    fake_X = G.components.synthesis.get_output_for(latent_wp, randomize_noise=False)
    fake_scores_out = fp32(D.get_output_for(fake_X, None))

    with tf.variable_scope('recon_loss'):
        vgg16_input_real = tf.transpose(reals, perm=[0, 2, 3, 1])
        vgg16_input_real = tf.image.resize_images(vgg16_input_real, size=[perceptual_img_size, perceptual_img_size], method=1)
        vgg16_input_real = ((vgg16_input_real + 1) / 2) * 255
        vgg16_input_fake = tf.transpose(fake_X, perm=[0, 2, 3, 1])
        vgg16_input_fake = tf.image.resize_images(vgg16_input_fake, size=[perceptual_img_size, perceptual_img_size], method=1)
        vgg16_input_fake = ((vgg16_input_fake + 1) / 2) * 255
        vgg16_feature_real = perceptual_model(vgg16_input_real)
        vgg16_feature_fake = perceptual_model(vgg16_input_fake)
        recon_loss_feats = feature_scale * tf.reduce_mean(tf.square(vgg16_feature_real - vgg16_feature_fake))
        recon_loss_pixel = tf.reduce_mean(tf.square(fake_X - reals))
        recon_loss_feats = autosummary('Loss/scores/loss_feats', recon_loss_feats)
        recon_loss_pixel = autosummary('Loss/scores/loss_pixel', recon_loss_pixel)
        recon_loss = recon_loss_feats + recon_loss_pixel
        recon_loss = autosummary('Loss/scores/recon_loss', recon_loss)

    with tf.variable_scope('adv_loss'):
        D_scale = autosummary('Loss/scores/d_scale', D_scale)
        adv_loss = D_scale * tf.reduce_mean(tf.nn.softplus(-fake_scores_out))
        adv_loss = autosummary('Loss/scores/adv_loss', adv_loss)

    loss = recon_loss + adv_loss

    return loss, recon_loss, adv_loss

#----------------------------------------------------------------------------
# Discriminator loss function.
def D_logistic_simplegp(E, G, D, reals, r1_gamma=10.0):

    num_layers, latent_dim = G.components.synthesis.input_shape[1:3]
    latent_w = E.get_output_for(reals, is_training=True)
    latent_wp = tf.reshape(latent_w, [reals.shape[0], num_layers, latent_dim])
    fake_X = G.components.synthesis.get_output_for(latent_wp, randomize_noise=False)
    real_scores_out = fp32(D.get_output_for(reals, None))
    fake_scores_out = fp32(D.get_output_for(fake_X, None))

    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss_fake = tf.reduce_mean(tf.nn.softplus(fake_scores_out))
    loss_real = tf.reduce_mean(tf.nn.softplus(-real_scores_out))

    with tf.name_scope('R1Penalty'):
        real_grads = fp32(tf.gradients(real_scores_out, [reals])[0])
        r1_penalty = tf.reduce_mean(tf.reduce_sum(tf.square(real_grads), axis=[1, 2, 3]))
        r1_penalty = autosummary('Loss/r1_penalty', r1_penalty)
        loss_gp = r1_penalty * (r1_gamma * 0.5)
    loss = loss_fake + loss_real + loss_gp
    return loss, loss_fake, loss_real, loss_gp


def E_loss_nei(E, G, D, perceptual_model, reals, feature_scale=0.00005, D_scale=0.1, perceptual_img_size=256, return_radius=False, latent_discriminator=None, return_reject_ratio=False):
    num_layers, latent_dim = G.components.synthesis.input_shape[1:3]
    if return_reject_ratio:
        latent_w, latent_radius, reject_ratio_before, reject_ratio_after = E.get_output_for(reals, return_reject_ratio=True, is_training=True)
        latent_radius = tf.tile(latent_radius, [reals.shape[0], 1])

    else:
        reject_ratio = None
        latent_w, latent_radius = E.get_output_for(reals, is_training=True)
    latent_wp = tf.reshape(latent_w, [reals.shape[0], num_layers, latent_dim])
    latent_radius = tf.reshape(latent_radius, [reals.shape[0], num_layers, 1])

    truncate = tf.random.truncated_normal(shape=latent_wp.shape, stddev=1)

    latent_wp_Gaussian = truncate * tf.rsqrt(tf.reduce_mean(tf.reduce_sum(tf.square(truncate), axis=2))) * \
                         latent_radius + latent_wp
    # latent_wp_Uniform = tf.random.uniform(shape=latent_wp.shape, minval=-1.0, maxval=1.0) * \
    #                     latent_radius / np.sqrt(latent_dim / 3.0) + latent_wp
    latent_wp_Uniform = tf.random.uniform(shape=latent_wp.shape, minval=-1.0, maxval=1.0) * \
                        2.0 / np.sqrt(latent_dim / 3.0) + latent_wp

    fake_X_Gaussian = G.components.synthesis.get_output_for(latent_wp_Gaussian, randomize_noise=False)
    fake_X_Uniform = G.components.synthesis.get_output_for(latent_wp_Uniform, randomize_noise=False)

    # fake_X = G.components.synthesis.get_output_for(latent_wp, randomize_noise=False)
    # fake_scores_out = fp32(D.get_output_for(fake_X, None))
    fake_scores_out_Uniform = fp32(D.get_output_for(fake_X_Uniform, None))

    with tf.variable_scope('recon_loss'):
        vgg16_input_real = tf.transpose(reals, perm=[0, 2, 3, 1])
        vgg16_input_real = tf.image.resize_images(vgg16_input_real, size=[perceptual_img_size, perceptual_img_size], method=1)
        vgg16_input_real = ((vgg16_input_real + 1) / 2) * 255
        vgg16_input_fake = tf.transpose(fake_X_Gaussian, perm=[0, 2, 3, 1])
        vgg16_input_fake = tf.image.resize_images(vgg16_input_fake, size=[perceptual_img_size, perceptual_img_size], method=1)
        vgg16_input_fake = ((vgg16_input_fake + 1) / 2) * 255
        vgg16_feature_real = perceptual_model(vgg16_input_real)
        vgg16_feature_fake = perceptual_model(vgg16_input_fake)
        recon_loss_feats = feature_scale * tf.reduce_mean(tf.square(vgg16_feature_real - vgg16_feature_fake))
        recon_loss_pixel = tf.reduce_mean(tf.square(fake_X_Gaussian - reals))
        recon_loss_feats = autosummary('Loss/scores/loss_feats', recon_loss_feats)
        recon_loss_pixel = autosummary('Loss/scores/loss_pixel', recon_loss_pixel)
        recon_loss = recon_loss_feats + recon_loss_pixel
        recon_loss = autosummary('Loss/scores/recon_loss', recon_loss)

    with tf.variable_scope('adv_loss'):
        D_scale = autosummary('Loss/scores/d_scale', D_scale)
        adv_loss = D_scale * tf.reduce_mean(tf.nn.softplus(-fake_scores_out_Uniform))
        adv_loss = autosummary('Loss/scores/adv_loss', adv_loss)

    loss = recon_loss + adv_loss
    if latent_discriminator is not None:
        fake_latent_score_Uniform = fp32(latent_discriminator.get_output_for(latent_wp_Uniform))
        with tf.variable_scope('dlatent_adv_loss'):
            dadv_loss = 0.1 * tf.reduce_mean(tf.nn.softplus(-fake_latent_score_Uniform))
            dadv_loss = autosummary('Loss/scores/dadv_loss', dadv_loss)
        loss = recon_loss + adv_loss + dadv_loss
        if return_radius:
            return loss, recon_loss, adv_loss, dadv_loss, tf.reduce_mean(latent_radius)
        else:
            return loss, recon_loss, adv_loss, dadv_loss

    if return_radius:
        if return_reject_ratio:
            return loss, recon_loss, adv_loss, tf.reduce_mean(latent_radius), reject_ratio_before, reject_ratio_after

        return loss, recon_loss, adv_loss, tf.reduce_mean(latent_radius)
    else:
        return loss, recon_loss, adv_loss


def D_logistic_simplegp_3(E, G, D, reals, r1_gamma=10.0, latent_discriminator=None):

    num_layers, latent_dim = G.components.synthesis.input_shape[1:3]
    latent_w, latent_radius = E.get_output_for(reals, is_training=True)
    latent_wp = tf.reshape(latent_w, [reals.shape[0], num_layers, latent_dim])
    # latent_radius = tf.reshape(latent_radius, [reals.shape[0], num_layers, 1])
    latent_radius = 2.0
    latent_wp_Uniform = tf.random.uniform(shape=latent_wp.shape, minval=-1.0, maxval=1.0) * \
                        latent_radius / np.sqrt(latent_dim / 3.0) + latent_wp

    fake_X_Uniform = G.components.synthesis.get_output_for(latent_wp_Uniform, randomize_noise=False)
    # fake_X = G.components.synthesis.get_output_for(latent_wp, randomize_noise=False)
    real_scores_out = fp32(D.get_output_for(reals, None))
    fake_scores_out = fp32(D.get_output_for(fake_X_Uniform, None))

    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss_fake = tf.reduce_mean(tf.nn.softplus(fake_scores_out))
    loss_real = tf.reduce_mean(tf.nn.softplus(-real_scores_out))

    with tf.name_scope('R1Penalty'):
        real_grads = fp32(tf.gradients(real_scores_out, [reals])[0])
        r1_penalty = tf.reduce_mean(tf.reduce_sum(tf.square(real_grads), axis=[1, 2, 3]))
        r1_penalty = autosummary('Loss/r1_penalty', r1_penalty)
        loss_gp = r1_penalty * (r1_gamma * 0.5)
    loss = loss_fake + loss_real + loss_gp
    if latent_discriminator is not None:
        z = tf.random.normal([reals.shape[0].value, latent_dim])
        w = G.components.mapping.get_output_for(z, None)
        w_score_out = fp32(latent_discriminator.get_output_for(w))
        fake_latent_score_Uniform = fp32(latent_discriminator.get_output_for(latent_wp_Uniform))
        loss_w = tf.reduce_mean(tf.nn.softplus(-w_score_out))
        loss_w = autosummary('Loss/scores/w', loss_w)
        loss_fake_latent = tf.reduce_mean(tf.nn.softplus(fake_latent_score_Uniform))
        loss_fake_latent = autosummary('Loss/scores/w_fake', loss_fake_latent)
        loss_ld = loss_fake_latent + loss_w
        # w_grads = fp32(tf.gradients(w_score_out, [w])[0])
        # w_r1_p = tf.reduce_mean(tf.reduce_sum(tf.square(w_grads), axis=[1, 2]))
        # w_r1_p = autosummary('Loss/wr1p', w_r1_p)
        # loss_gwp = r1_gamma * 0.5 * 0.1 * w_r1_p
        # loss += loss_gwp
        return loss, loss_fake, loss_real, loss_gp, loss_ld
    return loss, loss_fake, loss_real, loss_gp