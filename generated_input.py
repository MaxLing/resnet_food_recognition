'''Visualization of the filters of resnet50, via gradient ascent in input space.
This script can run on CPU in a few minutes.
'''
from __future__ import print_function

from scipy.misc import imsave
import numpy as np
import time
from keras import models
from keras import backend as K
from keras.applications import resnet50

# dimensions of the generated pictures for each filter/class
img_width = 224
img_height = 224

# the name of the layer we want to visualize - last layer
layer_name = 'predictions'

# util function to convert a tensor into a valid image

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255

    x = np.clip(x, 0, 255).astype('uint8')
    return x

K.set_learning_phase(1)

model = models.load_model('model/resnet50_9_norm_rmspop.h5')
# model = models.load_model('resnet50_notop.h5')
print('Model loaded.')

model.summary()

# this is the placeholder for the input images
input_img = model.input

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers])


kept_filters = []

for filter_index in range(101):
    # we only scan through the first 25 filters (actually class)
    print('Processing filter %d' % filter_index)

    # we build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    # layer_output = layer_dict[layer_name].output

    # loss = K.mean(layer_output[:, :, :, filter_index])
    loss = K.mean(model.output[:, filter_index])

    # we compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]

    # normalization a tensor by its L2 norm
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])

    # step size for gradient ascent
    step = 1

    # we start from a gray image with some random noise
    input_img_data = np.random.random((1, img_width, img_height, 3))
    input_img_data = (input_img_data - 0.5) * 20 + 128

    # we run gradient ascent for 30 steps
    for i in range(30):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

        print('Current loss value:', loss_value)
        if loss_value <= 0.:
            # some filters get stuck to 0, we can skip them
            break

    # decode the resulting input image
    if loss_value > 0:
        img = deprocess_image(input_img_data[0])
        kept_filters.append((img, loss_value))

# we will stich the first 25 filters on a 5 x 5 grid.
n = 5

# the filters that have the highest loss are assumed to be better-looking.
# we will only keep the top 25 filters.
# kept_filters.sort(key=lambda x: x[1], reverse=True)
# kept_filters = kept_filters[:n * n]

# # build a black picture with enough space for
# # our 5 x 5 filters of size 224 x 224, with a 5px margin in between
# margin = 5
# width = n * img_width + (n - 1) * margin
# height = n * img_height + (n - 1) * margin
# stitched_filters = np.zeros((width, height, 3))
#
# # fill the picture with our saved filters
# for i in range(n):
#     for j in range(n):
#         img, loss = kept_filters[i * n + j]
#         stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
#                          (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img
#
# # save the result to disk
# imsave('stitched_filters_%dx%d.png' % (n, n), stitched_filters)

with open('ETHZ-FOOD-101/food-101/meta/classes.txt') as f:
    classes = f.read().splitlines()

for i in range(101):
    img, loss = kept_filters[i]
    cla = classes[i]
    imsave('generative/%s.png' % (cla), img)