# Variational-Auto-Encoder
Re-writing a VAE code for better readability

The code shown on this repository is based on: https://github.com/bnsreenu/python_for_microscopists/blob/master/178_179_variational_autoencoders_mnist.py

and it implements the following changes:
- class CustomLayer(keras.layers.Layer) is replaced with a functionally equivalent set of instructions
- z = Lambda(sample_z, output_shape=(latent_dim, ), name='z')([z_mu, z_sigma]) is replaced with a functionally equivalent set of instructions

  More details are presented in this document https://docs.google.com/document/d/1wWYY-gmVcxvUU_0Utr6tRTtO4C8QBZ8ihTzaJ97zfCI/edit
