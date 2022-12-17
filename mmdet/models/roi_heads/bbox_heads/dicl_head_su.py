from itertools import accumulate
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from mmcv.cnn import Scale, normal_init
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init

from mmdet.core import (auto_fp16, build_bbox_coder, force_fp32, multi_apply,
                        multiclass_nms, multiclass_nms_aug)
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy
from mmdet.models.utils import HybridMemoryMultiFocalPercent, Quaduplet2Loss
from mmcv.ops import DeformConv2dPack
from .labeled_matching_layer_queue import LabeledMatchingLayerQueue
from .unlabeled_matching_layer import UnlabeledMatchingLayer
from .triplet_loss import TripletLossFilter

@HEADS.register_module()
class DICLHeadsu(nn.Module):
    '''for person search, output reid features'''
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively."""

    def __init__(self,
                 with_avg_pool=False,
                 with_cls=True,
                 with_reg=True,
                 roi_feat_size=7,
                 in_channels=256,
                 num_classes=80,
                 bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     target_means=[0., 0., 0., 0.],
                     target_stds=[0.1, 0.1, 0.2, 0.2]),
                 reg_class_agnostic=False,
                 reg_decoded_bbox=False,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 rcnn_bbox_bn=False,
                 no_bg=False,
                 no_bg_triplet=False,
                 triplet_weight=1,
                 loss_add=True,
                 use_sim_loss=True,
                 use_kl_loss=True,
                 test_shuffle=False,
                 coefficient_sim=1,
                 coefficient_kl=0.1,
                 stacked_convs=2,
                 flag_reid_fc=True,
                 feature_h=14,
                 feature_w=6,
                 momentum = 0.2,
                 ):
        super(DICLHeadsu, self).__init__()
        assert with_cls or with_reg
        self.with_avg_pool = with_avg_pool
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.reg_class_agnostic = reg_class_agnostic
        self.reg_decoded_bbox = reg_decoded_bbox
        self.fp16_enabled = False
        self.momentum = momentum


        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.no_bg = no_bg
        self.no_bg_triplet = no_bg_triplet
        self.triplet_weight = triplet_weight
        self.loss_add = loss_add
        self.use_sim_loss = use_sim_loss
        self.use_kl_loss = use_kl_loss
        self.test_shuffle = test_shuffle
        self.coefficient_sim = coefficient_sim
        self.coefficient_kl = coefficient_kl
        self.stacked_convs = stacked_convs
        self.feat_channels = 512
        self.flag_reid_fc = flag_reid_fc
        self.deform_conv = DeformConv2dPack(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,)

        self.loss_tri = TripletLossFilter()

        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        else:
            in_channels *= self.roi_feat_area
        # if self.with_cls:
        #     # need to add background class
        #     self.fc_cls = nn.Linear(in_channels, num_classes + 1)
        self.rcnn_bbox_bn = rcnn_bbox_bn
        if self.with_reg:
            out_dim_reg = 4 if reg_class_agnostic else 4 * num_classes
            # if self.rcnn_bbox_bn:
            #     self.fc_reg = nn.Sequential(nn.Linear(in_channels, out_dim_reg),
            #     nn.BatchNorm1d(out_dim_reg)
            #     )
            # self.fc_reg = nn.Linear(in_channels, out_dim_reg)
            self.reg_convs = nn.ModuleList()
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                self.reg_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=None,
                        norm_cfg=dict(type='BN', requires_grad=True),
                        act_cfg=dict(type='ReLU'), #None, #
                        bias='auto'),)
            if self.rcnn_bbox_bn:
                self.fc_reg = nn.Sequential(nn.Linear(self.feat_channels, out_dim_reg),
                nn.BatchNorm1d(out_dim_reg)
                )
            else:
                self.fc_reg = nn.Linear(self.feat_channels, out_dim_reg)


            self.cls_convs = nn.ModuleList()
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                self.cls_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=None,
                        norm_cfg=dict(type='BN', requires_grad=True),
                        act_cfg=dict(type='ReLU'), #None,#
                        bias='auto'),)
            self.fc_cls = nn.Linear(self.feat_channels, num_classes + 1)

        self.id_feature = nn.Linear(in_channels, 256)
        self.gt_id_feature = nn.Linear(in_channels, 256)
        #for reid loss
        self.debug_imgs = None
        #set all proposal score to 1, for enquery inference
        self.proposal_score_max = False
        self.feature_h = feature_h
        self.feature_w = feature_w
        self.fc_reid = nn.Linear(in_channels * self.feature_h * self.feature_w, 256)  ###

        num_person = 483
        # num_person = 5532
        queue_size = 500
        # queue_size = 5000
        #self.classifier_reid = nn.Linear(self.feat_channels, num_person)
        self.labeled_matching_layer = LabeledMatchingLayerQueue(num_persons=num_person, feat_len=256) # for mot17half
        self.unlabeled_matching_layer = UnlabeledMatchingLayer(queue_size=queue_size, feat_len=256)


    def init_weights(self):
        # conv layers are already initialized by ConvModule
        if self.with_cls:
            nn.init.normal_(self.fc_cls.weight, 0, 0.01)
            nn.init.constant_(self.fc_cls.bias, 0)
            for m in self.cls_convs:
                if isinstance(m.conv, nn.Conv2d):
                    normal_init(m.conv, std=0.01)
        if self.with_reg:
            nn.init.normal_(self.fc_reg.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg.bias, 0)
            for m in self.reg_convs:
                if isinstance(m.conv, nn.Conv2d):
                    normal_init(m.conv, std=0.01)
        nn.init.normal_(self.id_feature.weight, 0, 0.001)
        nn.init.constant_(self.id_feature.bias, 0)


    @auto_fp16()
    def forward(self, x, gt_x=None, sampling_results=None):  ##(train 256*2048*7*7) 2*2048*7*7, search_gt_x=None

        x = self.deform_conv(x)
        gt_x = self.deform_conv(gt_x)
        
        cls_feat = x
        reg_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_feat = F.adaptive_avg_pool2d(cls_feat, (1, 1)).view(cls_feat.size(0), -1)  #######adaptive_avg_pool2d
        cls_score = self.fc_cls(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        reg_feat = F.adaptive_avg_pool2d(reg_feat, (1, 1)).view(reg_feat.size(0), -1)
        bbox_pred = self.fc_reg(reg_feat)

        x_reid = x
        if self.flag_reid_fc:
            if gt_x is not None:
                id_pred = F.normalize(self.fc_reid(x_reid.view(x_reid.size(0), -1)))
                gt_id_pred = F.normalize(self.fc_reid(gt_x.view(gt_x.size(0), -1)))
            else:
                id_pred = F.normalize(self.fc_reid(x_reid.view(x_reid.size(0), -1)))
                gt_id_pred = None
        else:
            if gt_x is not None:
                x_reid = F.adaptive_max_pool2d(x_reid, (1, 1)).view(x_reid.size(0), -1)#adaptive_avg_pool2d
                x_reid = self.id_feature(x_reid)
                gt_x = F.adaptive_max_pool2d(gt_x, (1, 1)).view(gt_x.size(0), -1)
                gt_x = self.id_feature(gt_x)
                id_pred = F.normalize(x_reid)
                gt_id_pred = F.normalize(gt_x)
            else:
                x_reid = F.adaptive_max_pool2d(x_reid, (1, 1)).view(x_reid.size(0), -1)
                x_reid = self.id_feature(x_reid)
                id_pred = F.normalize(x_reid)
                gt_id_pred = None
        if not self.training:
            if gt_id_pred is not None:
                id_pred = (id_pred + gt_id_pred) / 2

        return cls_score, bbox_pred, id_pred, gt_id_pred

    def _get_target_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes,
                           pos_gt_labels, cfg):
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        #labels = pos_bboxes.new_full((num_samples, 2),
        #                             self.num_classes,
        #                             dtype=torch.long)
        labels = pos_bboxes.new_full((num_samples, 3),
                                     self.num_classes,
                                     dtype=torch.long)
        #background id is -2
        labels[:, 1] = -2
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes)
            else:
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights

    def get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rcnn_train_cfg,
                    concat=True):
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            cfg=rcnn_train_cfg)

        if concat:
            labels = torch.cat(labels, 0) ###256*3
            label_weights = torch.cat(label_weights, 0)  ###256
            bbox_targets = torch.cat(bbox_targets, 0)  ###256*4
            bbox_weights = torch.cat(bbox_weights, 0)  ###256*4
        return labels, label_weights, bbox_targets, bbox_weights

    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'id_pred', 'gt_id_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             id_pred,
             gt_id_pred,
             sampling_results,
             gt_labels,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        id_labels = labels[:, 1]
        labels = labels[:, 0]
        losses = dict()

        # pos_id_pred = id_pred[id_labels != -2]
        # mean_id_pred = (pos_id_pred + gt_id_pred) / 2
        batch_size = len(sampling_results)
        l_nums_pos = list(len(sam.pos_bboxes) for sam in sampling_results)
        acc_nums_sam = list(accumulate((len(sam.pos_bboxes) + len(sam.neg_bboxes)) for sam in sampling_results))
        acc_nums_sam.append(0)
        acc_nums_gt = list(accumulate(sam.num_gts for sam in sampling_results))
        acc_nums_gt.append(0)
        batch_gt_id_pred = list(gt_id_pred[acc_nums_gt[i-1]:acc_nums_gt[i], :] for i in range(batch_size))

        mean_id_pred = []
        gt_list_as_pos = []
        pos_id_pred = []
        new_id_pred = id_pred.clone()
        for i in range(batch_size):
            _gt_list_as_pos = batch_gt_id_pred[i][sampling_results[i].pos_assigned_gt_inds]
            gt_list_as_pos.append(_gt_list_as_pos)

            _pos_id_pred = id_pred[acc_nums_sam[i - 1]: acc_nums_sam[i - 1] + l_nums_pos[i], :]
            pos_id_pred.append(_pos_id_pred)

            _mean_id_pred = (_pos_id_pred + _gt_list_as_pos) / 2
            mean_id_pred.append(_mean_id_pred)

            new_id_pred[acc_nums_sam[i - 1]: acc_nums_sam[i - 1] + l_nums_pos[i]] = _mean_id_pred

        mean_id_pred = torch.cat(mean_id_pred, dim=0)
        gt_list_as_pos = torch.cat(gt_list_as_pos, dim=0)
        assert pos_id_pred != id_pred[id_labels != -2]
        pos_id_pred = torch.cat(pos_id_pred, dim=0)

        labeled_matching_scores, labeled_matching_reid, labeled_matching_ids = self.labeled_matching_layer(mean_id_pred, id_labels[id_labels != -2])
        labeled_matching_scores *= 10

        unlabeled_matching_scores = self.unlabeled_matching_layer(mean_id_pred, id_labels[id_labels != -2])
        unlabeled_matching_scores *= 10
        matching_scores = torch.cat((labeled_matching_scores, unlabeled_matching_scores), dim=1)
        pid_labels = id_labels[id_labels != -2].clone()
        pid_labels[pid_labels == -2] = -1

        p_i = F.softmax(matching_scores, dim=1)
        focal_p_i = (1 - p_i) ** 2 * p_i.log()
        losses['loss_oim'] = F.nll_loss(focal_p_i, pid_labels, ignore_index=-1)

        pos_reid1 = torch.cat((mean_id_pred, labeled_matching_reid), dim=0)
        pid_labels1 = torch.cat((pid_labels, labeled_matching_ids), dim=0)
        losses['loss_tri'] = self.loss_tri(pos_reid1, pid_labels1)

        losses['loss_sim'] = self.coefficient_sim / len(pos_id_pred) * sum(
            1 - pos_id_pred[i].unsqueeze(dim=0) @ gt_list_as_pos[i].unsqueeze(dim=1) for i in
            range(len(mean_id_pred)))
        sim_pred = pos_id_pred @ pos_id_pred.transpose(0, 1)
        sim_gt = gt_list_as_pos @ gt_list_as_pos.transpose(0, 1)
        sim_pred = F.log_softmax(sim_pred, dim=-1)
        sim_gt = F.log_softmax(sim_gt, dim=-1)
        losses['loss_kl'] = self.coefficient_kl * (
                F.kl_div(sim_pred, sim_gt, log_target=True)
                + F.kl_div(sim_gt, sim_pred, log_target=True)
        )  # 'reduction=batchmean'


        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score.contiguous(),
                    labels.contiguous(),
                    label_weights.contiguous(),
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred.sum() * 0



        return losses

    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'id_pred'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   id_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[:, 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            if isinstance(scale_factor, float):
                bboxes /= scale_factor
            else:
                scale_factor = bboxes.new_tensor(scale_factor)
                bboxes = (bboxes.view(bboxes.size(0), -1, 4) /
                          scale_factor).view(bboxes.size()[0], -1)

        if cfg is None:
            return bboxes, scores
        else:
            if self.proposal_score_max:
                scores[:, 0] = 1
                scores[:, 1] = 0
            det_bboxes, det_labels, det_ids = multiclass_nms_aug(bboxes, scores, [id_pred, ],
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)
            if det_ids is None:
                det_ids = det_bboxes.new_zeros((0, id_pred.shape[1]))
            else:
                det_ids = det_ids[0]
            det_bboxes = torch.cat([det_bboxes, det_ids], dim=1)

            return det_bboxes, det_labels

    @force_fp32(apply_to=('bbox_preds', ))
    def refine_bboxes(self, rois, labels, bbox_preds, pos_is_gts, img_metas):
        """Refine bboxes during training.

        Args:
            rois (Tensor): Shape (n*bs, 5), where n is image number per GPU,
                and bs is the sampled RoIs per image. The first column is
                the image id and the next 4 columns are x1, y1, x2, y2.
            labels (Tensor): Shape (n*bs, ).
            bbox_preds (Tensor): Shape (n*bs, 4) or (n*bs, 4*#class).
            pos_is_gts (list[Tensor]): Flags indicating if each positive bbox
                is a gt bbox.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Refined bboxes of each image in a mini-batch.

        Example:
            >>> # xdoctest: +REQUIRES(module:kwarray)
            >>> import kwarray
            >>> import numpy as np
            >>> from mmdet.core.bbox.demodata import random_boxes
            >>> self = BBoxHead(reg_class_agnostic=True)
            >>> n_roi = 2
            >>> n_img = 4
            >>> scale = 512
            >>> rng = np.random.RandomState(0)
            >>> img_metas = [{'img_shape': (scale, scale)}
            ...              for _ in range(n_img)]
            >>> # Create rois in the expected format
            >>> roi_boxes = random_boxes(n_roi, scale=scale, rng=rng)
            >>> img_ids = torch.randint(0, n_img, (n_roi,))
            >>> img_ids = img_ids.float()
            >>> rois = torch.cat([img_ids[:, None], roi_boxes], dim=1)
            >>> # Create other args
            >>> labels = torch.randint(0, 2, (n_roi,)).long()
            >>> bbox_preds = random_boxes(n_roi, scale=scale, rng=rng)
            >>> # For each image, pretend random positive boxes are gts
            >>> is_label_pos = (labels.numpy() > 0).astype(np.int)
            >>> lbl_per_img = kwarray.group_items(is_label_pos,
            ...                                   img_ids.numpy())
            >>> pos_per_img = [sum(lbl_per_img.get(gid, []))
            ...                for gid in range(n_img)]
            >>> pos_is_gts = [
            >>>     torch.randint(0, 2, (npos,)).byte().sort(
            >>>         descending=True)[0]
            >>>     for npos in pos_per_img
            >>> ]
            >>> bboxes_list = self.refine_bboxes(rois, labels, bbox_preds,
            >>>                    pos_is_gts, img_metas)
            >>> print(bboxes_list)
        """
        img_ids = rois[:, 0].long().unique(sorted=True)
        assert img_ids.numel() <= len(img_metas)

        bboxes_list = []
        for i in range(len(img_metas)):
            inds = torch.nonzero(
                rois[:, 0] == i, as_tuple=False).squeeze(dim=1)
            num_rois = inds.numel()

            bboxes_ = rois[inds, 1:]
            label_ = labels[inds]
            bbox_pred_ = bbox_preds[inds]
            img_meta_ = img_metas[i]
            pos_is_gts_ = pos_is_gts[i]

            bboxes = self.regress_by_class(bboxes_, label_, bbox_pred_,
                                           img_meta_)

            # filter gt bboxes
            pos_keep = 1 - pos_is_gts_
            keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[:len(pos_is_gts_)] = pos_keep

            bboxes_list.append(bboxes[keep_inds.type(torch.bool)])

        return bboxes_list

    @force_fp32(apply_to=('bbox_pred', ))
    def regress_by_class(self, rois, label, bbox_pred, img_meta):
        """Regress the bbox for the predicted class. Used in Cascade R-CNN.

        Args:
            rois (Tensor): shape (n, 4) or (n, 5)
            label (Tensor): shape (n, )
            bbox_pred (Tensor): shape (n, 4*(#class)) or (n, 4)
            img_meta (dict): Image meta info.

        Returns:
            Tensor: Regressed bboxes, the same shape as input rois.
        """
        assert rois.size(1) == 4 or rois.size(1) == 5, repr(rois.shape)

        if not self.reg_class_agnostic:
            label = label * 4
            inds = torch.stack((label, label + 1, label + 2, label + 3), 1)
            bbox_pred = torch.gather(bbox_pred, 1, inds)
        assert bbox_pred.size(1) == 4

        if rois.size(1) == 4:
            new_rois = self.bbox_coder.decode(
                rois, bbox_pred, max_shape=img_meta['img_shape'])
        else:
            bboxes = self.bbox_coder.decode(
                rois[:, 1:], bbox_pred, max_shape=img_meta['img_shape'])
            new_rois = torch.cat((rois[:, [0]], bboxes), dim=1)

        return new_rois
