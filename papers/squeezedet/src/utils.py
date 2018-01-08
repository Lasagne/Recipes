"""
Copyright (c) 2017 Corvidim (corvidim.net)
Licensed under the BSD 2-Clause License (see LICENSE for details)
Authors: V. Ablavsky, A. J. Fox
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from pdb import set_trace as keyboard

import os, platform

import numpy as np

import theano, theano.tensor as T

######################################################################################
#                  is_headless()
######################################################################################
def is_headless():
    sysname = platform.system()
    if sysname == 'Linux':
        if os.getenv('DISPLAY') is None:
            return True
    else:
        return False

######################################################################################
#                   av_elu
######################################################################################
def av_elu(thn_x,t):
    """
    analog of SqueezDet src/utils/util.py:safe_exp()
    and inspired by lasagne.layers.elu()
    https://github.com/Lasagne/Lasagne/commit/c4e3f81d6b1e6f7518b3efa4681e548f87b2fd72         
    which, in turn, is a one-liner
      theano.tensor.switch(x > 0, x, theano.tensor.exp(x) - 1)

    x (theano tensor): tensor to be transformed
    t (floatX_t):      threshold
    """
    return T.switch(thn_x > t, T.exp(t)*(thn_x - t + 1),T.exp(thn_x))

######################################################################################
#                 bbox_transform()
######################################################################################
def bbox_transform(bbox):
  """
  [analog of SqueezeDet src/utils/util.py:bbox_transform()]
    convert a bbox of form [cx, cy, w, h] to [xmin, ymin, xmax, ymax].
    Works for numpy array or list of tensors.
  """
  cx, cy, w, h = bbox
  out_box = [[]]*4
  out_box[0] = cx-w/2
  out_box[1] = cy-h/2
  out_box[2] = cx+w/2
  out_box[3] = cy+h/2

  return out_box

######################################################################################
#                 bbox_transform_inv()
######################################################################################
def bbox_transform_inv(bbox):
  """
  [analog of SqueezeDet src/utils/util.py:bbox_transform()]
    convert a bbox of form [xmin, ymin, xmax, ymax] to [cx, cy, w, h]. 
    Works for numpy array or list of tensors.
  """
  xmin, ymin, xmax, ymax = bbox
  out_box = [[]]*4

  #NB SqueezeDet assumed OpenCV ==> origin pixel (0,0) at center of upper-left pixel; PIL uses upper-left
  width       = xmax - xmin + 1.0
  height      = ymax - ymin + 1.0
  out_box[0]  = xmin + 0.5*width
  out_box[1]  = ymin + 0.5*height
  out_box[2]  = width
  out_box[3]  = height

  return out_box

######################################################################################
#                  filter_prediction()
######################################################################################
def filter_prediction(mc, boxes, probs, cls_idx):
    """
    [from SqueezeDet src/nn_skeleton.py]

    Filter bounding box predictions with probability threshold and
    non-maximum supression.

    Args:
      boxes: array of [cx, cy, w, h].
      probs: array of probabilities
      cls_idx: array of class indices
    Returns:
      final_boxes: array of filtered bounding boxes.
      final_probs: array of filtered probabilities
      final_cls_idx: array of filtered class indices
    """
    if mc.TOP_N_DETECTION < len(probs) and mc.TOP_N_DETECTION > 0:
      order = probs.argsort()[:-mc.TOP_N_DETECTION-1:-1]
      probs = probs[order]
      boxes = boxes[order]
      cls_idx = cls_idx[order]
    else:
      filtered_idx = np.nonzero(probs>mc.PROB_THRESH)[0]
      probs = probs[filtered_idx]
      boxes = boxes[filtered_idx]
      cls_idx = cls_idx[filtered_idx]

    final_boxes = []
    final_probs = []
    final_cls_idx = []

    for c in range(mc.CLASSES):
      idx_per_class = [i for i in range(len(probs)) if cls_idx[i] == c]
      keep = sqzdet_nms(boxes[idx_per_class], probs[idx_per_class], mc.NMS_THRESH)
      for i in range(len(keep)):
        if keep[i]:
          final_boxes.append(boxes[idx_per_class[i]])
          final_probs.append(probs[idx_per_class[i]])
          final_cls_idx.append(c)
    return final_boxes, final_probs, final_cls_idx

######################################################################################
#                  sqzdet_nms()
######################################################################################
def sqzdet_nms(boxes, probs, threshold):
  """
  [from SqueezeDet src/utils/util.py]

  Non-Maximum supression.
  Args:
    boxes: array of [cx, cy, w, h] (center format)
    probs: array of probabilities
    threshold: two boxes are considered overlapping if their IOU is largher than
        this threshold
    form: 'center' or 'diagonal'
  Returns:
    keep: array of True or False.
  """

  order = probs.argsort()[::-1]
  keep = [True]*len(order)

  for i in range(len(order)-1):
    ovps = batch_iou(boxes[order[i+1:]], boxes[order[i]])
    for j, ov in enumerate(ovps):
      if ov > threshold:
        keep[order[j+i+1]] = False
  return keep

######################################################################################
#                  batch_iou()
######################################################################################
def batch_iou(boxes, box):
  """
  [from SqueezeDet src/utils/util.py]

  Compute the Intersection-Over-Union of a batch of boxes with another
  box.

  Args:
    box1: 2D array of [cx, cy, width, height].
    box2: a single array of [cx, cy, width, height]
  Returns:
    ious: array of a float number in range [0, 1].
  """
  lr = np.maximum(
      np.minimum(boxes[:,0]+0.5*boxes[:,2], box[0]+0.5*box[2]) - \
      np.maximum(boxes[:,0]-0.5*boxes[:,2], box[0]-0.5*box[2]),
      0
  )
  tb = np.maximum(
      np.minimum(boxes[:,1]+0.5*boxes[:,3], box[1]+0.5*box[3]) - \
      np.maximum(boxes[:,1]-0.5*boxes[:,3], box[1]-0.5*box[3]),
      0
  )
  inter = lr*tb
  union = boxes[:,2]*boxes[:,3] + box[2]*box[3] - inter
  return inter/union

######################################################################################
#                  set_anchors()
######################################################################################
def set_anchors(img_h,img_w,H=22,W=76):
    """
    generalized from kitti_squeezeDet_config, to support arbitrary-size input image
    """
    B = 9
    anchor_shapes = np.reshape(
        [np.array(
            [[  36.,  37.], [ 366., 174.], [ 115.,  59.],
             [ 162.,  87.], [  38.,  90.], [ 258., 173.],
             [ 224., 108.], [  78., 170.], [  72.,  43.]])] * H * W,
        (H, W, B, 2)
    )
    center_x = np.reshape(
        np.transpose(
            np.reshape(
                np.array([np.arange(1, W+1)*float(img_w)/(W+1)]*H*B),
                (B, H, W)
            ),
            (1, 2, 0)
        ),
        (H, W, B, 1)
    )
    center_y = np.reshape(
        np.transpose(
            np.reshape(
                np.array([np.arange(1, H+1)*float(img_h)/(H+1)]*W*B),
                (B, W, H)
            ),
            (2, 1, 0)
        ),
        (H, W, B, 1)
    )
    anchors = np.reshape(
        np.concatenate((center_x, center_y, anchor_shapes), axis=3),
        (-1, 4)
    )

    return anchors


######################################################################################
#                  get_det_roi()
######################################################################################
def get_det_roi(cfg_mc, img_shape, net_out, par):
    """
    net_out: a numpy tensor from lasagne-net final layer,  
             e.g., net_out.shape == (1, 72, 22, 76)

             The second axis, i.e., net_out[0,:,r,c]
             contains (a) class-conditional and marginal probabilities,
             (b) confidence scores, and (c) anchor-deformations for all the
             anchors at anchor-site (i,j).
             
    The code in this function parses the second axis in order (a), (b), and (c)
    to extract these probabilities and confidence scores, and compute
    the ROIs given anchor-deformations. It then runs non-max-suppression,
    and then returns the most confidence ROIs that are left.


    NB: much of the code below follows SqueezeDet src/nn_skeleton.py
        and retains tf (tensorflow) order, as opposed to th (Theano/Lasagne):

         tf: (rows, columns, channels, filters) aka (h, w, channels, filters)
         thn: (filters, channels, rows, columns) aka (filters, channels, h, w)
    """
    
    n_class_probs = cfg_mc.ANCHOR_PER_GRID * cfg_mc.CLASSES         # 27

    # for each anchor, the network predicts "confidence score" (a scalar)
    # n_conf_scores = n_anchor_per_grid * (1+n_class)
    #               = n_anchor_per_grid + n_class_prob
    n_conf_scores = cfg_mc.ANCHOR_PER_GRID

    # analog of pred_class_probs from nn_skeleton.py:147
    """
      self.pred_class_probs = tf.reshape(
          tf.nn.softmax(
              tf.reshape(
                  preds[:, :, :, :num_class_probs],
                  [-1, mc.CLASSES]
              )
          ),
          [mc.BATCH_SIZE, mc.ANCHORS, mc.CLASSES],
          name='pred_class_probs'
      )
    """

    idx_begin_probs = 0
    idx_end_probs = n_class_probs
    idx_begin_conf = idx_end_probs
    idx_end_conf = idx_begin_conf + n_conf_scores
    idx_begin_pred_box_delta = idx_end_conf
    # end index will equal net_out.shape[1] 
    idx_end_pred_box_delta = idx_begin_pred_box_delta + 4*cfg_mc.ANCHOR_PER_GRID

    net_out_tf_order = np.transpose(net_out,(2,3,0,1))  # (22, 76, 1, 72
    class_probs0 = net_out_tf_order[:, :, :, idx_begin_probs:idx_end_probs]  # (22, 76, 1, 27)
    class_probs1 = np.reshape(class_probs0, (-1, cfg_mc.CLASSES)) # (15048, 3)

    thn_x = T.dmatrix('thn_x')
    thn_y1 = T.nnet.softmax(thn_x)
    f_softmax = theano.function([thn_x],thn_y1)
    class_probs2 = f_softmax(class_probs1)  # (15048, 3)
    n_anchors = net_out.shape[2] * net_out.shape[3] * n_conf_scores   # net_out.shape (1, 72, 22, 76)
    class_probs3 = np.reshape(class_probs2,
                              (1, n_anchors, cfg_mc.CLASSES))

    #analog of 158: #confidence
    """
      # confidence
      num_confidence_scores = mc.ANCHOR_PER_GRID+num_class_probs
      self.pred_conf = tf.sigmoid(
          tf.reshape(
              preds[:, :, :, num_class_probs:num_confidence_scores],
              [mc.BATCH_SIZE, mc.ANCHORS]
          ),
          name='pred_confidence_score'
      )
    """

    conf_scores0 = net_out_tf_order[:, :, :, idx_begin_conf:idx_end_conf]  # (22, 76, 1, 9)
    conf_scores1 = np.reshape(conf_scores0, (1, n_anchors))  # (1, 15048)
    thn_y2 = T.nnet.sigmoid(thn_x)
    f_sigmoid = theano.function([thn_x],thn_y2)
    conf_scores2 = f_sigmoid(conf_scores1) # (1, 15048)

    # analog of 267: with tf.variable_scope('probability') as scope:
    """
     probs = tf.multiply(
          self.pred_class_probs,
          tf.reshape(self.pred_conf, [mc.BATCH_SIZE, mc.ANCHORS, 1]),
          name='final_class_prob'
      )

    278: self.det_probs = tf.reduce_max(probs, 2, name='score')

    279: self.det_class = tf.argmax(probs, 2, name='class_idx')
    """
    

    conf_scores3 = np.reshape(conf_scores2, (1, n_anchors, 1))  # (1, 15048, 1)
    class_probs4 = np.multiply(class_probs3, conf_scores3)  # (1, 15048, 3)
    class_probs_max = np.max(class_probs4, axis=2).squeeze() # (15048,)
    class_decisions = np.argmax(class_probs4, axis=2).squeeze() # (1, 15048)

    #analog of pred_box_delta from nn_skeleton.py:169 ("bbox_delta")
    net_out_slice_box = net_out_tf_order[:, :, :, idx_end_conf:] # (22, 76, 1, 36)
    pred_box_delta = np.reshape(net_out_slice_box, (1, n_anchors, 4)) # (1, 15048, 4)

    # analog of box_center_* from nn_skeleton.py:180ff
    # conceptually, unpack pred_box_delta of shape (1, 15048, 4)
    # into 4-tuple, each of shape (1,15048) delta_x, delta_y, delta_w, delta_h 

    img_h = img_shape[0]
    img_w = img_shape[1]
    anchor_box = set_anchors(img_h,img_w,H=net_out.shape[2],W=net_out.shape[3])

    anchor_x = anchor_box[:, 0]   # (15048,)
    anchor_y = anchor_box[:, 1]
    anchor_w = anchor_box[:, 2]
    anchor_h = anchor_box[:, 3]

    # analog of 179: with tf.variable_scope('stretching'):
    delta_x = pred_box_delta.squeeze()[:,0]
    delta_y = pred_box_delta.squeeze()[:,1]    
    delta_w = pred_box_delta.squeeze()[:,2]
    delta_h = pred_box_delta.squeeze()[:,3]
    box_center_x = anchor_x + delta_x * anchor_w
    box_center_y = anchor_y + delta_y * anchor_h
    box_width = anchor_w * av_elu(delta_w, cfg_mc.EXP_THRESH).eval()
    box_height = anchor_h * av_elu(delta_h, cfg_mc.EXP_THRESH).eval()

    # analog of 209: with tf.variable_scope('trimming'):
    xmins, ymins, xmaxs, ymaxs = bbox_transform([box_center_x, box_center_y, box_width, box_height])

    xmins = np.minimum(np.maximum(0.0, xmins), img_w-1.0)
    ymins = np.minimum(np.maximum(0.0, ymins), img_h-1.0)
    xmax = np.maximum(np.minimum(img_w-1.0, xmaxs), 0.0)
    ymax = np.maximum(np.minimum(img_w-1.0, ymaxs), 0.0)

    box_centers_ = bbox_transform_inv([xmins, ymins, xmaxs, ymaxs]) # tf: each is TensorShape([Dimension(1), Dimension(15048)])
    box_centers_stacked = np.stack(box_centers_)  # (4, 15048)   tf: TensorShape([Dimension(4), Dimension(1), Dimension(15048)])
    det_roi_ = np.transpose(box_centers_stacked) # (15048, 4)  tf: transpose(...,(1, 2, 0))  => tf: (1, 15048, 4)


    t_ = filter_prediction(cfg_mc, det_roi_, class_probs_max, class_decisions)
    (filtered_roi, filtered_probs, filtered_class) = t_
    
    keep_idx    = [idx for idx in range(len(filtered_probs)) \
                   if filtered_probs[idx] > cfg_mc.PLOT_PROB_THRESH]    # list len 16
    filtered_roi = [filtered_roi[idx] for idx in keep_idx]  # list len 16
    filtered_probs = [filtered_probs[idx] for idx in keep_idx]
    filtered_class = [filtered_class[idx] for idx in keep_idx]
    
    detection_info = [cfg_mc.CLASS_NAMES[idx]+': {:.2f}'.format(prob) \
                      for idx, prob in zip(filtered_class, filtered_probs)]

    if par['verbose']:
        print('len(filtered_roi): {0} len(filtered_probs): '
              '{1} cfg_mc.PLOT_PROB_THRESH: {2} cfg_mc.NMS_THRESH: {3}'.format(
                  len(filtered_roi),
                  len(filtered_probs),
                  cfg_mc.PLOT_PROB_THRESH,
                  cfg_mc.NMS_THRESH))

    return (filtered_roi, detection_info)
