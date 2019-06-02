#!/usr/bin/python
"""
# ==============================
# flowlib.py
# library for optical flow processing
# Author: Ruoteng Li
# Date: 6th Aug 2016
# ==============================
"""
import png
import numpy as np
import matplotlib.colors as cl
import matplotlib.pyplot as plt
from PIL import Image
from imageio import imread


UNKNOWN_FLOW_THRESH = 1e9
SMALLFLOW = 0.0
LARGEFLOW = 1e8
DEBUG = False  # flag to print out verbose information like: range of optical flow, dimensions of matrix, etc.

"""
=============
Flow Section
=============
"""


def show_flow(filename):
    """
    visualize optical flow map using matplotlib
    :param filename: optical flow file
    :return: None
    """
    flow = read_flow(filename)
    img = flow_to_image(flow)
    plt.imshow(img)
    plt.show()


def visualize_flow(flow, mode='Y'):
    """
    this function visualize the input flow
    :param flow: input flow in array
    :param mode: choose which color mode to visualize the flow (Y: Ccbcr, RGB: RGB color)
    :return: None
    """
    if mode == 'Y':
        # Ccbcr color wheel
        img = flow_to_image(flow)
        plt.imshow(img)
        plt.show()
    elif mode == 'RGB':
        (h, w) = flow.shape[0:2]
        du = flow[:, :, 0]
        dv = flow[:, :, 1]
        valid = flow[:, :, 2]
        max_flow = max(np.max(du), np.max(dv))
        img = np.zeros((h, w, 3), dtype=np.float64)
        # angle layer
        img[:, :, 0] = np.arctan2(dv, du) / (2 * np.pi)
        # magnitude layer, normalized to 1
        img[:, :, 1] = np.sqrt(du * du + dv * dv) * 8 / max_flow
        # phase layer
        img[:, :, 2] = 8 - img[:, :, 1]
        # clip to [0,1]
        small_idx = img[:, :, 0:3] < 0
        large_idx = img[:, :, 0:3] > 1
        img[small_idx] = 0
        img[large_idx] = 1
        # convert to rgb
        img = cl.hsv_to_rgb(img)
        # remove invalid point
        img[:, :, 0] = img[:, :, 0] * valid
        img[:, :, 1] = img[:, :, 1] * valid
        img[:, :, 2] = img[:, :, 2] * valid
        # show
        plt.imshow(img)
        plt.show()

    return None


def read_flow(filename):
    """
    read optical flow from Middlebury .flo file
    :param filename: name of the flow file
    :return: optical flow data in matrix
    """
    with open(filename, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
        else:
            w = np.fromfile(f, np.int32, count=1)[0]
            h = np.fromfile(f, np.int32, count=1)[0]
            if DEBUG:
                print("Reading {0} x {1} flo file".format(w, h))
            data = np.fromfile(f, np.float32, count=2*w*h)
            # Reshape data into 3D array (columns, rows, bands)
            return np.resize(data, (h, w, 2))


def read_flow_png(flow_file):
    """
    Read optical flow from KITTI .png file
    :param flow_file: name of the flow file
    :return: optical flow data in matrix
    """
    flow_object = png.Reader(filename=flow_file)
    flow_direct = flow_object.asDirect()
    flow_data = list(flow_direct[2])
    (w, h) = flow_direct[3]['size']
    flow = np.zeros((h, w, 3), dtype=np.float64)
    for i in range(len(flow_data)):
        flow[i, :, 0] = flow_data[i][0::3]
        flow[i, :, 1] = flow_data[i][1::3]
        flow[i, :, 2] = flow_data[i][2::3]

    invalid_idx = (flow[:, :, 2] == 0)
    flow[:, :, 0:2] = (flow[:, :, 0:2] - 2 ** 15) / 64.0
    flow[invalid_idx, 0] = 0
    flow[invalid_idx, 1] = 0
    return flow


def write_flow(flow, filename):
    """
    write optical flow in Middlebury .flo format
    :param flow: optical flow map
    :param filename: optical flow file path to be saved
    :return: None
    """
    f = open(filename, 'wb')
    magic = np.array([202021.25], dtype=np.float32)
    (height, width) = flow.shape[0:2]
    w = np.array([width], dtype=np.int32)
    h = np.array([height], dtype=np.int32)
    magic.tofile(f)
    w.tofile(f)
    h.tofile(f)
    flow.tofile(f)
    f.close()


def segment_flow(flow):
    h = flow.shape[0]
    w = flow.shape[1]
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    idx = ((abs(u) > LARGEFLOW) | (abs(v) > LARGEFLOW))
    idx2 = (abs(u) == SMALLFLOW)
    class0 = (v == 0) & (u == 0)
    u[idx2] = 0.00001
    tan_value = v / u

    class1 = (tan_value < 1) & (tan_value >= 0) & (u > 0) & (v >= 0)
    class2 = (tan_value >= 1) & (u >= 0) & (v >= 0)
    class3 = (tan_value < -1) & (u <= 0) & (v >= 0)
    class4 = (tan_value < 0) & (tan_value >= -1) & (u < 0) & (v >= 0)
    class8 = (tan_value >= -1) & (tan_value < 0) & (u > 0) & (v <= 0)
    class7 = (tan_value < -1) & (u >= 0) & (v <= 0)
    class6 = (tan_value >= 1) & (u <= 0) & (v <= 0)
    class5 = (tan_value >= 0) & (tan_value < 1) & (u < 0) & (v <= 0)

    seg = np.zeros((h, w))

    seg[class1] = 1
    seg[class2] = 2
    seg[class3] = 3
    seg[class4] = 4
    seg[class5] = 5
    seg[class6] = 6
    seg[class7] = 7
    seg[class8] = 8
    seg[class0] = 0
    seg[idx] = 0

    return seg


def get_metrics(metrics):
    dash = "-" * 50
    line = '_' * 50
    title_str = "{:^50}".format('MPI-Sintel Flow Error Metrics')
    headers = '{:<5s}{:^15s}{:^15s}{:^15s}'.format('Mask', 'MANG', 'STDANG', 'MEPE')
    all_string = '{:<5s}{:^15.4f}{:^15.4f}{:^15.4f}'.format('(all)', metrics['mangall'], metrics['stdangall'],
                                                            metrics['EPEall'])
    mat_string = '{:<5s}{:^15.4f}{:^15.4f}{:^15.4f}'.format('(mat)', metrics['mangmat'], metrics['stdangmat'],
                                                            metrics['EPEmat'])
    umat_string = '{:<5s}{:^15.4f}{:^15.4f}{:^15.4f}'.format('(umt)', metrics['mangumat'], metrics['stdangumat'],
                                                             metrics['EPEumat'])
    dis_headers = '{:<5s}{:^15s}{:^15s}{:^15s}'.format('', 'S0-10', 'S10-40', 'S40+')

    dis_string = '{:<5s}{:^15.4f}{:^15.4f}{:^15.4f}'.format('(dis)', metrics['S0-10'], metrics['S10-40'],
                                                            metrics['S40plus'])

    final_string_formatted = "{0}\n{1}\n{2}\n{3}\n{4}\n{5}\n{6}\n{7}\n{8}\n{9}\n{10}\n{11}".format(
        line, title_str, line, headers, dash, all_string, mat_string, umat_string, line, dis_headers, dash, dis_string)

    return final_string_formatted


# TODO: add other metrics available in the MATLAB script (EPEmat, EPEumat, S0-10, S10-40, S40+)
def compute_all_metrics(est_flow, gt_flow, occ_mask=None, inv_mask=None):
    """
    Computes the metrics (if enough masks are provided) of MPI-Sintel (EPEall, EPEmat, EPEumat, S0-10, S10-40, S40+)
        -NOTE: flow_error_mask discards the pixels with the passed value (i.e.: uses the 'complementary' mask).
    Based on the original MATLAB code from Stefan Roth. Changed by Ferran Pérez Gamonal to compute MPI-Sintel metrics.
    Ported here from MATLABs implementation (the original code can be found in the supplemental material of the
    following publication: http://www.ipol.im/pub/art/2019/238/)
    :param est_flow: estimated optical flow with shape: (height, width, 2)
    :param gt_flow: ground truth optical flow with shape: (height, width, 2)
    :param occ_mask: (optional) occlusions mask (1s specify that a pixel is occluded, 0 otherwise)
    :param inv_mask: (optional) invalid mask that specifies which pixels have invalid flow (not considered for error)
    :return: dictionary with computed metrics (0 if it cannot be computed, no masks)
    """
    metrics = dict([])
    bord = 0
    height, width, _ = gt_flow.shape
    # Separate gt flow fields (horizontal + vertical)
    of_gt_x = gt_flow[:, :, 0]
    of_gt_y = gt_flow[:, :, 1]

    # Separate est flow fields (horizontal + vertical)
    of_est_x = est_flow[:, :, 0]
    of_est_y = est_flow[:, :, 1]

    if occ_mask is not None:
        occ_mask = occ_mask == 255  # check that once read the value is 255
    else:
        occ_mask = np.full((height, width), False)

    if inv_mask is not None:
        inv_mask = inv_mask == 255  # check that once read the value is 255
    else:
        inv_mask = np.full((height, width), False)  # e.g.: every pixel has a valid flow

    # EPE all
    mang, stdang, mepe = flow_error_mask(of_gt_x, of_gt_y, of_est_x, of_est_y, inv_mask, True, bord)
    metrics['EPEall'] = mepe
    metrics['mangall'] = mang
    metrics['stdangall'] = stdang

    # Check if there are any occluded pixels
    if occ_mask.size:  # array is not empty
        # EPE-matched (pixels that are not occluded)
        # Always mask out invalid pixels (inv_mask == 1)
        # For matched we want to avoid the 1's
        mat_occ_msk = occ_mask | inv_mask  # 0's are valid and non-occluded ==> gt_value=1 (rejected value)
        mat_mang, mat_stdang, mat_mepe = flow_error_mask(of_gt_x, of_gt_y, of_est_x, of_est_y, mat_occ_msk, True, bord)

        # EPE-unmatched (pixels that are occluded)
        # " " " invalid pixels
        # For unmatched we want to avoid the 0's
        un_occ_msk = occ_mask & ~inv_mask  # 1's are valid and occluded
        umat_mang, umat_stdang, umat_mepe = flow_error_mask(of_gt_x, of_gt_y, of_est_x, of_est_y, un_occ_msk, False,
                                                            bord)
    else:
        # No occluded pixels (umat = 0, mat = all)
        mat_mepe = mepe
        umat_mepe = 0
        mat_mang = mang
        umat_mang = 0
        mat_stdang = stdang
        umat_stdang= 0

    metrics['EPEmat'] = mat_mepe
    metrics['mangmat'] = mat_mang
    metrics['stdangmat'] = mat_stdang
    metrics['EPEumat'] = umat_mepe
    metrics['mangumat'] = umat_mang
    metrics['stdangumat'] = umat_stdang

    # Masks for S0 - 10, S10 - 40 and S40 +)
    l1_of = np.sqrt(of_gt_x ** 2 + of_gt_y ** 2)
    disp_mask = l1_of
    disp_mask[np.asarray(disp_mask < 10).nonzero()] = 0
    disp_mask[(disp_mask >= 10) & (disp_mask <= 40)] = 1
    # careful & takes precedence to <=/>=/== (use parenthesis)
    disp_mask[disp_mask > 40] = 2

    # Actually compute S0 - 10, S10 - 40 and S40 +
    # Note: not correct (ambiguous truth evaluation) in Python (used "number in array") instead
    # pixels_disp_1 = sum(disp_mask[:] == 0)  # S0-10
    # pixels_disp_2 = sum(disp_mask[:] == 1)  # S10-40
    # pixels_disp_3 = sum(disp_mask[:] == 2)  # S40+

    # Remember that flow_error_mask ignores the values equal to gt_value in the mask
    # So, for S0-10, we want to pass only the pixels with a velocity within the 0-10 range
    # We pass 1 in this position, -1 elsewhere (number different than the labels 0 through 2)
    # ======= S0-10 =======
    if 0 in disp_mask[:]:
        # Compute  S0 - 10 nominally
        msk_s010 = disp_mask
        # msk_s010[np.asarray(msk_s010 != 0).nonzero()] = -1
        # msk_s010[(msk_s010 == 0)] = 1
        # msk_s010[np.asarray(msk_s010 == -1).nonzero()] = 0
        # msk_s010 = msk_s010 == 1  # convert to bool! (True/False in python)
        # We want 1's only where 0's (pixels with velocity in range 0-10) in disp_mask, 0 elsewhere
        # Numpy has np.where(condition, value_where_cond_is_met, value_elsewhere)
        # And accepts bools
        msk_s010 = np.where(msk_s010 == 0, True, False)
        # Mask out invalid pixels(defined in the 'invalid' folder)
        # % We want to take into account only the valid and values = 1 in msk_s010
        msk_s010 = (msk_s010) & (~inv_mask)
        _, _, s0_10 = flow_error_mask(of_gt_x, of_gt_y, of_est_x, of_est_y, msk_s010, False, bord)
    else:
        s0_10 = 0

    metrics['S0-10'] = s0_10

    # ======= S10-40 =======
    if 1 in disp_mask[:]:
        # Compute S10 - 40 nominally
        msk_s1040 = disp_mask  # have value 1
        # msk_s1040[np.asarray(msk_s1040 != 1).nonzero()] = -1
        # msk_s1040[np.asarray(msk_s1040 == -1).nonzero()] = 0
        # msk_s1040 = msk_s1040 == -1
        # np.where() to the rescue
        msk_s1040 = np.where(msk_s1040 == 1, True, False)

        # Mask out the invalid pixels
        # Same reasoning as s0 - 10 mask
        msk_s1040 = (msk_s1040) & (~inv_mask)
        # The desired pixels have already value 1, we are done.
        _, _, s10_40 = flow_error_mask(of_gt_x, of_gt_y, of_est_x, of_est_y, msk_s1040, False, bord)
    else:
        s10_40 = 0

    metrics['S10-40'] = s10_40

    # ======= S40+ =======
    if 2 in disp_mask[:]:
        # Compute S40+ nominally
        msk_s40plus = disp_mask
        # msk_s40plus[np.asarray(msk_s40plus != 2).nonzero()] = -1
        # msk_s40plus[np.asarray(msk_s40plus == 2).nonzero()] = 1
        # msk_s40plus[np.asarray(msk_s40plus == -1).nonzero()] = 0
        # msk_s40plus = msk_s40plus == 1
        msk_s40plus = np.where(msk_s40plus == 2, True, False)

        # Mask out the invalid pixels
        # Same reasoning as s0 - 10 and s10 - 40 masks
        msk_s40plus = (msk_s40plus) & (~inv_mask)
        _, _, s40plus = flow_error_mask(of_gt_x, of_gt_y, of_est_x, of_est_y, msk_s40plus, False, bord)
    else:
        s40plus = 0

    metrics['S40plus'] = s40plus

    return metrics


# TODO: should index with tuple not directly with a list of indices/logical values
# A[idx] ==> A[tuple(idx)] : maybe due to idx being multi-dimensional and not a column/row vector?
def flow_error(tu, tv, u, v):
    """
    Calculate average end point error (a.k.a. EPEall, 'all' as for all pixels in the image)
    :param tu: ground-truth horizontal flow map
    :param tv: ground-truth vertical flow map
    :param u:  estimated horizontal flow map
    :param v:  estimated vertical flow map
    :return: End point error of the estimated flow
    """
    smallflow = 0.0
    '''
    stu = tu[bord+1:end-bord,bord+1:end-bord]
    stv = tv[bord+1:end-bord,bord+1:end-bord]
    su = u[bord+1:end-bord,bord+1:end-bord]
    sv = v[bord+1:end-bord,bord+1:end-bord]
    '''
    stu = tu[:]
    stv = tv[:]
    su = u[:]
    sv = v[:]

    idxUnknown = (abs(stu) > UNKNOWN_FLOW_THRESH) | (abs(stv) > UNKNOWN_FLOW_THRESH)
    stu[idxUnknown] = 0
    stv[idxUnknown] = 0
    su[idxUnknown] = 0
    sv[idxUnknown] = 0

    ind2 = [(np.absolute(stu) > smallflow) | (np.absolute(stv) > smallflow)]
    index_su = su[tuple(ind2)]
    index_sv = sv[tuple(ind2)]
    an = 1.0 / np.sqrt(index_su ** 2 + index_sv ** 2 + 1)
    un = index_su * an
    vn = index_sv * an

    index_stu = stu[tuple(ind2)]
    index_stv = stv[tuple(ind2)]
    tn = 1.0 / np.sqrt(index_stu ** 2 + index_stv ** 2 + 1)
    tun = index_stu * tn
    tvn = index_stv * tn

    angle = un * tun + vn * tvn + (an * tn)
    index = [angle == 1.0]
    angle[index] = 0.999
    ang = np.arccos(angle)
    mang = np.mean(ang)
    mang = mang * 180 / np.pi
    stdang = np.std(ang * 180 / np.pi)

    epe = np.sqrt((stu - su) ** 2 + (stv - sv) ** 2)
    epe = epe[tuple(ind2)]
    mepe = np.mean(epe)
    return mang, stdang, mepe


def flow_error_mask(tu, tv, u, v, mask=None, gt_value=False, bord=0):
    """
    Calculate average end point error
    :param tu: ground-truth horizontal flow map
    :param tv: ground-truth vertical flow map
    :param u:  estimated horizontal flow map
    :param v:  estimated vertical flow map
    :param mask: binary mask that specifies a region of interest
    :param gt_value: specifies if we ignore False's (0's) or True's (0's) in the computation of a certain metric
    :return: End point error of the estimated flow
    """
    smallflow = 0.0

    # stu = tu[bord+1:end-bord,bord+1:end-bord]
    # stv = tv[bord+1:end-bord,bord+1:end-bord]
    # su = u[bord+1:end-bord,bord+1:end-bord]
    # sv = v[bord+1:end-bord,bord+1:end-bord]

    stu = tu[:]
    stv = tv[:]
    su = u[:]
    sv = v[:]

    idxUnknown = (abs(stu) > UNKNOWN_FLOW_THRESH) | (abs(stv) > UNKNOWN_FLOW_THRESH) | (mask == gt_value)
    # stu[idxUnknown] = 0
    # stv[idxUnknown] = 0
    # su[idxUnknown] = 0
    # sv[idxUnknown] = 0

    # ind2 = [(np.absolute(stu[:]) >= smallflow) | (np.absolute(stv[:]) >= smallflow)]
    ind2 = (abs(stu) >= smallflow) | (abs(stv) >= smallflow)
    ind2 = (idxUnknown < 1) & ind2
    index_su = su[ind2]  # should be updated to A[tuple(idx_list)]
    index_sv = sv[ind2]
    an = 1.0 / np.sqrt(index_su ** 2 + index_sv ** 2 + 1)
    un = index_su * an
    vn = index_sv * an

    index_stu = stu[ind2]
    index_stv = stv[ind2]
    tn = 1.0 / np.sqrt(index_stu ** 2 + index_stv ** 2 + 1)
    tun = index_stu * tn
    tvn = index_stv * tn

    # angle = un * tun + vn * tvn + (an * tn)
    # index = [angle == 1.0]
    # angle[index] = 0.999
    ang = np.arccos(un * tun + vn * tvn + (an * tn))
    mang = np.mean(ang)
    mang = mang * 180 / np.pi
    stdang = np.std(ang * 180 / np.pi)

    epe = np.sqrt((stu - su) ** 2 + (stv - sv) ** 2)
    epe = epe[ind2]
    mepe = np.mean(epe)
    return mang, stdang, mepe


def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    if DEBUG:
        print("max flow: {.4f}\nflow range:\nu = {.3f} .. {.3f}\nv = {.3f} .. {.3f}".format(maxrad, minu, maxu, minv,
                                                                                            maxv))

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


def evaluate_flow_file(gt, pred):
    """
    evaluate the estimated optical flow end point error according to ground truth provided
    :param gt: ground truth file path
    :param pred: estimated optical flow file path
    :return: end point error, float32
    """
    # Read flow files and calculate the errors
    gt_flow = read_flow(gt)        # ground truth flow
    eva_flow = read_flow(pred)     # predicted flow
    # Calculate errors
    average_pe = flow_error(gt_flow[:, :, 0], gt_flow[:, :, 1], eva_flow[:, :, 0], eva_flow[:, :, 1])
    return average_pe


def evaluate_flow(gt_flow, pred_flow):
    """
    gt: ground-truth flow
    pred: estimated flow
    """
    average_pe = flow_error(gt_flow[:, :, 0], gt_flow[:, :, 1], pred_flow[:, :, 0], pred_flow[:, :, 1])
    return average_pe


"""
==============
Disparity Section
==============
"""


def read_disp_png(file_name):
    """
    Read optical flow from KITTI .png file
    :param file_name: name of the flow file
    :return: optical flow data in matrix
    """
    image_object = png.Reader(filename=file_name)
    image_direct = image_object.asDirect()
    image_data = list(image_direct[2])
    (w, h) = image_direct[3]['size']
    channel = len(image_data[0]) / w
    flow = np.zeros((h, w, channel), dtype=np.uint16)
    for i in range(len(image_data)):
        for j in range(channel):
            flow[i, :, j] = image_data[i][j::channel]
    return flow[:, :, 0] / 256


def disp_to_flowfile(disp, filename):
    """
    Read KITTI disparity file in png format
    :param disp: disparity matrix
    :param filename: the flow file name to save
    :return: None
    """
    f = open(filename, 'wb')
    magic = np.array([202021.25], dtype=np.float32)
    (height, width) = disp.shape[0:2]
    w = np.array([width], dtype=np.int32)
    h = np.array([height], dtype=np.int32)
    empty_map = np.zeros((height, width), dtype=np.float32)
    data = np.dstack((disp, empty_map))
    magic.tofile(f)
    w.tofile(f)
    h.tofile(f)
    data.tofile(f)
    f.close()


"""
==============
Image Section
==============
"""


def read_image(filename):
    """
    Read normal image of any format
    :param filename: name of the image file
    :return: image data in matrix uint8 type
    """
    img = Image.open(filename)
    im = np.array(img)
    return im


def warp_image(im, flow):
    """
    Use optical flow to warp image to the next
    :param im: image to warp
    :param flow: optical flow
    :return: warped image
    """
    from scipy import interpolate
    image_height = im.shape[0]
    image_width = im.shape[1]
    flow_height = flow.shape[0]
    flow_width = flow.shape[1]
    n = image_height * image_width
    (iy, ix) = np.mgrid[0:image_height, 0:image_width]
    (fy, fx) = np.mgrid[0:flow_height, 0:flow_width]
    fx += flow[:, :, 0]
    fy += flow[:, :, 1]
    mask = np.logical_or(fx < 0, fx > flow_width)
    mask = np.logical_or(mask, fy < 0)
    mask = np.logical_or(mask, fy > flow_height)
    fx = np.minimum(np.maximum(fx, 0), flow_width)
    fy = np.minimum(np.maximum(fy, 0), flow_height)
    points = np.concatenate((ix.reshape(n, 1), iy.reshape(n, 1)), axis=1)
    xi = np.concatenate((fx.reshape(n, 1), fy.reshape(n, 1)), axis=1)
    warp = np.zeros((image_height, image_width, im.shape[2]))
    for i in range(im.shape[2]):
        channel = im[:, :, i]
        plt.imshow(channel, cmap='gray')
        values = channel.reshape(n, 1)
        new_channel = interpolate.griddata(points, values, xi, method='cubic')
        new_channel = np.reshape(new_channel, [flow_height, flow_width])
        new_channel[mask] = 1
        warp[:, :, i] = new_channel.astype(np.uint8)

    return warp.astype(np.uint8)


"""
==============
Others
==============
"""


def scale_image(image, new_range):
    """
    Linearly scale the image into desired range
    :param image: input image
    :param new_range: the new range to be aligned
    :return: image normalized in new range
    """
    min_val = np.min(image).astype(np.float32)
    max_val = np.max(image).astype(np.float32)
    min_val_new = np.array(min(new_range), dtype=np.float32)
    max_val_new = np.array(max(new_range), dtype=np.float32)
    scaled_image = (image - min_val) / (max_val - min_val) * (max_val_new - min_val_new) + min_val_new
    return scaled_image.astype(np.uint8)


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel


def test_error_metrics(est_path, gt_path, occ_path=None, inv_path=None):
    """
    Function to test that the flow error metrics are computed correctly (checking against OG Matlab implementation)
    :param est_path: path to predicted/estimated flow
    :param gt_path: path to the corresponding ground truth flow
    :param occ_path: path to the occlusions mask (Sintel and Kitti, although Kitti requires more complex bool logic)
    :param inv_path: path to the invalid pixels mask (Sintel and Kitti)
    :return: nothing, prints metrics to stdout for debugging
    """
    # Read flows and masks
    est_flow = read_flow(est_path)
    gt_flow = read_flow(gt_path)
    if occ_path is not None:
        occ_mask = imread(occ_path)
    else:
        occ_mask = None
    if inv_path is not None:
        inv_mask = imread(inv_path)
    else:
        inv_mask = None

    # Call compute_all_metrics
    metrics = compute_all_metrics(est_flow, gt_flow, occ_mask=occ_mask, inv_mask=inv_mask)  # main debugging
    # Get metrics properly formatted and print them out
    final_string_formated = get_metrics(metrics)
    print(final_string_formated)


if __name__ == '__main__':
    # Test error function
    path_to_est_flow = '../data/samples/sintel/frame_00186_flow.flo'
    path_to_gt_flow = '../data/samples/sintel/frame_00186.flo'
    path_to_occ_mask = '../data/samples/sintel/frame_00186_occ_mask.png'
    path_to_inv_mask = '../data/samples/sintel/frame_00186_inv_mask.png'
    test_error_metrics(path_to_est_flow, path_to_gt_flow, occ_path=path_to_occ_mask, inv_path=path_to_inv_mask)
