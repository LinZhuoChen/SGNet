import torch
import torch.nn as nn
from datetime import datetime
from scipy import ndimage
from PIL import Image
import numpy as np

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

# colour map
# label_colours = [(0,0,0)
#                 # 0=background
#                 ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
#                 # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
#                 ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
#                 # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
#                 ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
#                 # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
#                 ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)]
#                 # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def labelcolormap(N):
    if N == 35:  # cityscape
        cmap = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (111, 74, 0), (81, 0, 81),
                (128, 64, 128), (244, 35, 232), (250, 170, 160), (230, 150, 140), (70, 70, 70), (102, 102, 156),
                (190, 153, 153),
                (180, 165, 180), (150, 100, 100), (150, 120, 90), (153, 153, 153), (153, 153, 153), (250, 170, 30),
                (220, 220, 0),
                (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
                (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 142)]
    else:
        cmap = []
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
                color = (r, g, b)
                cmap.append(color)
    return cmap


def decode_labels(mask, num_images=1, num_classes=40):
    """Decode batch of segmentation masks.

    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).

    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
    label_colours = labelcolormap(num_classes)
    mask = mask.data.cpu().numpy()
    n, h, w = mask.shape
    assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
    n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[k_, j_] = label_colours[k]
        outputs[i] = np.array(img)
    return outputs


def decode_predictions(preds, num_images=1, num_classes=40):
    """Decode batch of segmentation masks.

    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).

    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
    label_colours = labelcolormap(num_classes)
    if isinstance(preds, list):
        preds_list = []
        for pred in preds:
            preds_list.append(pred[-1].data.cpu().numpy())
        preds = np.concatenate(preds_list, axis=0)
    else:
        preds = preds.data.cpu().numpy()

    preds = np.argmax(preds, axis=1)
    n, h, w = preds.shape
    assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
    n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new('RGB', (len(preds[i, 0]), len(preds[i])))
        pixels = img.load()
        for j_, j in enumerate(preds[i, :, :]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[k_, j_] = label_colours[k]
        outputs[i] = np.array(img)
    return outputs


def inv_preprocess(imgs, num_images, img_mean):
    """Inverse preprocessing of the batch of images.
       Add the mean vector and convert from BGR to RGB.

    Args:
      imgs: batch of input images.
      num_images: number of images to apply the inverse transformations on.
      img_mean: vector of mean colour values.

    Returns:
      The batch of the size num_images with the same spatial dimensions as the input.
    """
    imgs = imgs.data.cpu().numpy()
    n, c, h, w = imgs.shape
    assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
    n, num_images)
    outputs = np.zeros((num_images, h, w, c), dtype=np.uint8)
    for i in range(num_images):
        outputs[i] = (np.transpose(imgs[i], (1, 2, 0)) + img_mean).astype(np.uint8)
    return outputs


def get_confusion_matrix(gt_label, pred_label, class_num):
    """
    Calcute the confusion matrix by given label and pred
    :param gt_label: the ground truth label
    :param pred_label: the pred label
    :param class_num: the nunber of class
    :return: the confusion matrix
    """
    index = (gt_label * class_num + pred_label).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((class_num, class_num))

    for i_label in range(class_num):
        for i_pred_label in range(class_num):
            cur_index = i_label * class_num + i_pred_label
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

    return confusion_matrix


def get_currect_time():
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

    return TIMESTAMP


def get_metric(hist):
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    return acc, acc_cls, mean_iu, fwavacc, iu[freq > 0]

def predict_whole(net, image, tile_size, recurrence, S):
    image = torch.from_numpy(image)
    S = torch.from_numpy(S)
    interp = nn.Upsample(size=tile_size, mode='bilinear', align_corners=True)
    prediction = net(image.cuda(), S.cuda())
    if isinstance(prediction, list):
        prediction = prediction[0]
    prediction = interp(prediction).cpu().data[0].numpy().transpose(1, 2, 0)
    return prediction

def predict_multiscale(net, image, S, tile_size, scales, classes, flip_evaluation, recurrence=False):
    """
    Predict an image by looking at it with different scales.
        We choose the "predict_whole_img" for the image with less than the original input size,
        for the input of larger size, we would choose the cropping method to ensure that GPU memory is enough.
    """
    image = image.data
    S = S.data
    N_, C_, H_, W_ = image.shape
    full_probs = np.zeros((tile_size[0], tile_size[1], classes))
    for scale in scales:
        scale = float(scale)
        scale_image = ndimage.zoom(image, (1.0, 1.0, scale, scale), order=1, prefilter=False)
        scale_S = ndimage.zoom(S, (1.0, 1.0, scale, scale), order=1, prefilter=False)
        scaled_probs = predict_whole(net, scale_image, tile_size, recurrence, scale_S)
        if flip_evaluation == True:
            flip_scaled_probs = predict_whole(net, scale_image[:, :, :, ::-1].copy(), tile_size, recurrence,
                                              scale_S[:, :, :, ::-1].copy())
            scaled_probs = 0.5 * (scaled_probs + flip_scaled_probs[:, ::-1, :])
        full_probs += scaled_probs
    full_probs /= len(scales)
    return full_probs

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """

    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

def maybe_download(model_name, model_url, model_dir=None, map_location=None):
    import os, sys
    from six.moves import urllib
    if model_dir is None:
        torch_home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch'))
        model_dir = os.getenv('TORCH_MODEL_ZOO', os.path.join(torch_home, 'models'))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = '{}.pth.tar'.format(model_name)
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        url = model_url
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urllib.request.urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)

def expand_model_dict(model_dict, state_dict, num_parallel):
    model_dict_keys = model_dict.keys()
    state_dict_keys = state_dict.keys()
    for model_dict_key in model_dict_keys:
        model_dict_key_re = model_dict_key.replace('module.', '')
        if model_dict_key_re in state_dict_keys:
            model_dict[model_dict_key] = state_dict[model_dict_key_re]
        for i in range(num_parallel):
            bn = '.bn_%d' % i
            replace = True if bn in model_dict_key_re else False
            model_dict_key_re = model_dict_key_re.replace(bn, '')
            if replace and model_dict_key_re in state_dict_keys:
                model_dict[model_dict_key] = state_dict[model_dict_key_re]
    return model_dict