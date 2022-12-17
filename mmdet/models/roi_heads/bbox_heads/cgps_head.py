from itertools import accumulate
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from mmdet.core import (auto_fp16, build_bbox_coder, force_fp32, multi_apply,
                        multiclass_nms, multiclass_nms_aug)
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy
from mmdet.models.utils import HybridMemoryMultiFocalPercent, Quaduplet2Loss, LinearAverage, NCEAverage, NCECriterion, CircleLoss, convert_label_to_similarity


@HEADS.register_module()
class CGPSHead(nn.Module):
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
                 loss_reid=dict(loss_weight=1.0),
                 rcnn_bbox_bn=False,
                 id_num = 55272,
                 no_bg=False,
                 no_bg_triplet=False,
                 top_percent=0.1,
                 use_quaduplet_loss=True,
                 triplet_weight=1,
                 triplet_bg_weight=0.25,
                 loss_add=True,
                 use_sim_loss=True,
                 use_kl_loss=True,
                 test_shuffle=False,
                 coefficient_sim=1,
                 coefficient_kl=0.1,
                 loss_ir=dict(
                     type='CrossEntropyLoss',
                     loss_weight=1.0),
                 use_ir=False,
                 use_cir_loss=False,
                 flag_reid_fc=False,
                 ):
        super(CGPSHead, self).__init__()
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

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_ir = build_loss(loss_ir)
        self.loss_reid = HybridMemoryMultiFocalPercent(256, id_num, top_percent=top_percent)
        self.loss_triplet = Quaduplet2Loss(bg_weight=triplet_bg_weight)
        self.use_quaduplet_loss = use_quaduplet_loss
        self.reid_loss_weight = loss_reid['loss_weight']
        self.no_bg = no_bg
        self.no_bg_triplet = no_bg_triplet
        self.triplet_weight = triplet_weight
        self.loss_add = loss_add
        self.use_sim_loss = use_sim_loss
        self.use_kl_loss = use_kl_loss
        self.test_shuffle = test_shuffle
        self.coefficient_sim = coefficient_sim
        self.coefficient_kl = coefficient_kl
        self.IR = LinearAverage(256, id_num).cuda()
        self.IR_NCE = NCEAverage(256, id_num, 4096).cuda()
        self.NCECriterion = NCECriterion(id_num)
        self.use_ir = use_ir
        self.cir_criterion = CircleLoss(m=0.25, gamma=16)
        self.use_cir_loss = use_cir_loss
        self.flag_reid_fc = flag_reid_fc


        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        else:
            in_channels *= self.roi_feat_area
        if self.with_cls:
            # need to add background class
            self.fc_cls = nn.Linear(in_channels, num_classes + 1)
        self.rcnn_bbox_bn = rcnn_bbox_bn
        if self.with_reg:
            out_dim_reg = 4 if reg_class_agnostic else 4 * num_classes
            if self.rcnn_bbox_bn:
                self.fc_reg = nn.Sequential(nn.Linear(in_channels, out_dim_reg),
                nn.BatchNorm1d(out_dim_reg)
                )
            self.fc_reg = nn.Linear(in_channels, out_dim_reg)
        self.id_feature = nn.Linear(in_channels, 256)
        self.gt_id_feature = nn.Linear(in_channels, 256)
        #for reid loss
        self.debug_imgs = None
        #set all proposal score to 1, for enquery inference
        self.proposal_score_max = False
        self.fc_reid = nn.Linear(in_channels * 7 * 3, 256)

    def init_weights(self):
        # conv layers are already initialized by ConvModule
        if self.with_cls:
            nn.init.normal_(self.fc_cls.weight, 0, 0.01)
            nn.init.constant_(self.fc_cls.bias, 0)
        if self.with_reg:
            nn.init.normal_(self.fc_reg.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg.bias, 0)
        nn.init.normal_(self.id_feature.weight, 0, 0.001)
        nn.init.constant_(self.id_feature.bias, 0)


    @auto_fp16()
    def forward(self, x, gt_x=None): ##(train 256*2048*7*7) 2*2048*7*7, search_gt_x=None
        if gt_x is not None:
            mean_value = torch.mean(torch.cat((x, gt_x), dim=0), dim=0, keepdim=True)
        else:
            mean_value = torch.mean(x, dim=0, keepdim=True)
        x_reid = x - mean_value


        if self.with_avg_pool:
            # x = self.avg_pool(x)  ##2*2048*1*1
            x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1) ##2*2048
        cls_score = self.fc_cls(x) if self.with_cls else None  ##2*2
        bbox_pred = self.fc_reg(x) if self.with_reg else None  ##2*4
        if self.flag_reid_fc:
            x_reid = x_reid.view(x_reid.size(0), -1)
            id_pred = F.normalize(self.fc_reid(x_reid))
            if gt_x is not None:
                gt_x = gt_x - mean_value
                gt_x = gt_x.view(gt_x.size(0), -1)
                gt_id_pred = F.normalize(self.fc_reid(gt_x))
            else:
                gt_id_pred = None
        else:
            x_reid = F.adaptive_avg_pool2d(x_reid, (1, 1)).view(x_reid.size(0), -1)
            id_pred = F.normalize(self.id_feature(x_reid))  ##256*256
            if gt_x is not None:
                gt_x = gt_x - mean_value
                gt_x = F.adaptive_avg_pool2d(gt_x, (1, 1)).view(gt_x.size(0), -1)
                gt_id_pred = F.normalize(self.id_feature(gt_x))
            else:
                gt_id_pred = None

        return cls_score, bbox_pred, id_pred, gt_id_pred #, search_gt_id_pred

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

        losses['loss_id'] = self.loss_reid(mean_id_pred, id_labels[id_labels != -2]) * self.reid_loss_weight
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
            ) #'reduction=batchmean'
        cluster_id_labels = self.loss_reid.get_cluster_ids(id_labels[id_labels != -2])
        new_id_labels = id_labels.clone()
        new_id_labels[id_labels != -2] = cluster_id_labels
        if self.use_quaduplet_loss:
            losses['loss_triplet'] = self.loss_triplet(new_id_pred, new_id_labels) * self.triplet_weight
        if self.use_cir_loss:
            inp_sp, inp_sn = convert_label_to_similarity(new_id_pred[id_labels != -2], new_id_labels[id_labels != -2])
            losses['loss_cir'] = self.cir_criterion(inp_sp, inp_sn)

        # mean_id_pred = torch.cat(mean_id_pred, dim=0)
        # add_gt_as_proposal_pred = torch.cat(add_gt_as_proposal_pred, dim=0)
        # gt_rid_labels = torch.cat(list(gt[:, 1] for gt in gt_labels), dim=0)
        # losses['loss_id_gt'] = self.loss_reid(mean_id_pred, gt_rid_labels) * self.reid_loss_weight
        # losses['loss_sim'] = self.coefficient_sim / len(mean_id_pred) * sum(
        #     1 - mean_id_pred[i].unsqueeze(dim=0) @ gt_id_pred[i].unsqueeze(dim=1) for i in
        #     range(len(mean_id_pred)))
        # sim_pred = add_gt_as_proposal_pred @ add_gt_as_proposal_pred.transpose(0, 1)
        # sim_gt = gt_id_pred @ gt_id_pred.transpose(0, 1)
        # sim_pred = F.log_softmax(sim_pred, dim=-1)
        # sim_gt = F.log_softmax(sim_gt, dim=-1)
        # losses['loss_kl'] = self.coefficient_kl * (
        #         F.kl_div(sim_pred, sim_gt, log_target=True)
        #         + F.kl_div(sim_gt, sim_pred, log_target=True)
        #     ) #'reduction=batchmean'
        # if self.use_quaduplet_loss:
        #     cluster_id_labels = self.loss_reid.get_cluster_ids(id_labels[id_labels != -2])
        #     new_id_labels = id_labels.clone()
        #     new_id_labels[id_labels != -2] = cluster_id_labels
        #     losses['loss_triplet'] = self.loss_triplet(new_id_pred, new_id_labels) * self.triplet_weight
        #     inp_sp, inp_sn = convert_label_to_similarity(new_id_pred[id_labels != -2], new_id_labels[id_labels != -2])
        #     losses['loss_cir'] = self.cir_criterion(inp_sp, inp_sn)

            # if self.loss_add:
            #     if gt_id_pred is not None:
            #         batch_size = len(sampling_results)
            #         l_nums_pos = list(len(sam.pos_bboxes) for sam in sampling_results)
            #         acc_nums_sam = list(
            #             accumulate((len(sam.pos_bboxes) + len(sam.neg_bboxes)) for sam in sampling_results))
            #         acc_nums_sam.append(0)
            #         acc_nums_gt = list(accumulate(sam.num_gts for sam in sampling_results))
            #         acc_nums_gt.append(0)
            #         batch_gt_id_pred = list(gt_id_pred[acc_nums_gt[i - 1]:acc_nums_gt[i], :] for i in range(batch_size))
            #         # print(acc_nums_sam, sampling_results)
            #         # batch_pos = []
            #         # start = 0
            #         # for i in range(batch_size):
            #         #     end = start + l_nums_pos[i]
            #         #     batch_pos.append(id_pred[start: end, :])
            #         #     start = start + acc_nums_sam[i]
            #         #     print('id_pred[start: end, :]', id_pred[start: end, :].shape)
            #         # print('sampling_results', sampling_results,)
            #         mean_id_pred = []
            #         new_id_pred = id_pred.clone()
            #         add_gt_as_proposal_pred = []
            #         for i in range(batch_size):
            #             _add_gt_as_proposal_pred = id_pred[acc_nums_sam[i - 1]: acc_nums_sam[i - 1] + sampling_results[
            #                 i].num_gts, :]
            #             add_gt_as_proposal_pred.append(_add_gt_as_proposal_pred)
            #             _mean_id_pred = (_add_gt_as_proposal_pred + batch_gt_id_pred[i]) / 2
            #             mean_id_pred.append(_mean_id_pred)
            #             new_id_pred[acc_nums_sam[i - 1]: acc_nums_sam[i - 1] + sampling_results[i].num_gts,
            #             :] = _mean_id_pred
            #
            #         mean_id_pred = torch.cat(mean_id_pred, dim=0)
            #         add_gt_as_proposal_pred = torch.cat(add_gt_as_proposal_pred, dim=0)
            #         gt_rid_labels = torch.cat(list(gt[:, 1] for gt in gt_labels), dim=0)

                # batch_size = len(sampling_results)
                # num_channel = 256
                # num_total_gt = len(gt_id_pred)
                # acc_nums_gt = list(accumulate(sam.num_gts for sam in sampling_results))
                # acc_nums_pos = list(len(sam.pos_bboxes) for sam in sampling_results)
                # l_nums_gt = list(sam.num_gts for sam in sampling_results)
                # gt_start_ids = [0] + acc_nums_gt[:-1]
                # gt_end_ids = acc_nums_gt[:-1] + [num_total_gt]
                # pos_id_pred = id_pred.reshape(batch_size, -1, num_channel)
                # pred_gt_mean = []
                #
                # for i, (_pred, _sample, gt_start, gt_end) in enumerate(zip(
                #         pos_id_pred, sampling_results, gt_start_ids, gt_end_ids)):
                #     num_positive = _sample.pos_inds.shape[0]
                #     num_gt = _sample.num_gts
                #     if num_positive == 0 or num_gt == 0:
                #         continue
                #     _gt = gt_id_pred[gt_start:gt_end]
                #
                #     gt_inds = torch.arange(num_gt, device=_pred.device).unsqueeze(0)  # shape: 1, #gt
                #     pred = _pred[:num_positive]  # shape: #pos, 256
                #     assigned_gt_inds = _sample.pos_assigned_gt_inds.unsqueeze(-1)  # shape: #pos, 1
                #     pos_to_gt_mask = assigned_gt_inds == gt_inds  # shape: #pos, #gt
                #     coefficiecy_inner = 1. / pos_to_gt_mask.float().sum(dim=1, keepdim=True) # shape: #pos, 1
                #     coefficiecy_outter = 1. / pos_to_gt_mask.any(dim=0).float().sum()  # scalar
                #
                #     # shape: #gt, 256
                #     pred_as_gt_order = (coefficiecy_inner * pos_to_gt_mask.float()).transpose(0, 1) @ pred
                #
                #     # similarity shape: #gt, #gt
                #     # similarity = pred_as_gt_order @ _gt.transpose(0, 1)
                #     # _loss_sim = 1 - coefficiecy_outter * similarity.diag().sum()  # scalar
                #     if self.test_shuffle:
                #         shuffle_gt = _gt[torch.randperm(_gt.size(0))]
                #         shuffle_similarity = pred_as_gt_order @ shuffle_gt.transpose(0, 1)
                #         _loss_sim = 1 - coefficiecy_outter * shuffle_similarity.diag().sum()
                #     else:
                #         similarity = pred_as_gt_order @ _gt.transpose(0, 1)
                #         _loss_sim = 1 - coefficiecy_outter * similarity.diag().sum()  # scalar
                #
                #     assert torch.isfinite(_loss_sim).all()
                #     losses_sim.append(_loss_sim)
                #
                #     # shape: #gt, #gt
                #     sim_pred = pred_as_gt_order @ pred_as_gt_order.transpose(0, 1)
                #     sim_gt = _gt @ _gt.transpose(0, 1)
                #     # norm
                #     # sim_pred = F.log_softmax(sim_pred.reshape(1, -1), dim=-1)
                #     # sim_gt = F.log_softmax(sim_gt.reshape(1, -1), dim=-1)
                #     sim_pred = F.log_softmax(sim_pred, dim=-1)
                #     sim_gt = F.log_softmax(sim_gt, dim=-1)
                #     # sim_pred = F.log_softmax(pred_as_gt_order, dim=-1)
                #     # sim_gt = F.log_softmax(_gt, dim=-1)
                #     _loss_kl = (
                #             F.kl_div(sim_pred, sim_gt, log_target=True)
                #             + F.kl_div(sim_gt, sim_pred, log_target=True)
                #     ) #'reduction=batchmean'
                #     assert torch.isfinite(_loss_kl).all()
                #     # losses_kl.append(_loss_kl)
                #
                #     pred = (pred + _gt[_sample.pos_assigned_gt_inds]) / 2
                #     pred_gt_mean.append(pred)
                # for i in range(batch_size):
                #     pos_id_pred[i][:acc_nums_pos[i]] = pred_gt_mean[i]
                # id_pred = pos_id_pred.reshape(-1, num_channel)

                # gt_rid_labels = []
                # start = 0
                # for i in range(batch_size):
                #     end = start + l_nums_gt[i]
                #     gt_rid_labels.append(rid_labels[start: end])
                #     start = start + acc_nums_pos[i]
                # gt_rid_labels = torch.cat(gt_rid_labels, dim=0)
                # losses['loss_id_gt'] = self.loss_reid(gt_id_pred, gt_rid_labels) * self.reid_loss_weight



                # if self.use_kl_loss:
                #     losses['loss_kl'] = 0.1 * torch.stack(losses_kl).mean() \
                #         if losses_kl else gt_id_pred.new_zeros([])
                # if self.use_sim_loss:
                #     losses['loss_sim'] = 0.1 * torch.stack(losses_sim).mean() \
                #         if losses_sim else gt_id_pred.new_zeros([])

                # losses = {
                #     'loss_sim': 0. * torch.stack(losses_sim).mean()
                #     if losses_sim else gt_id_pred.new_zeros([]),
                #     'loss_kl': 0.1 * torch.stack(losses_kl).mean()
                #     if losses_kl else gt_id_pred.new_zeros([]),
                # }

        #     mean_id_pred = []
        #     _id_pred = id_pred.reshape(batch_size, -1, 256)
        #     for i in range(batch_size):
        #         for j in range(sampling_results[i].num_gts):
        #             inds = sampling_results[i].pos_assigned_gt_inds == j
        #             mean_id_pred.append(torch.mean(_id_pred[i][:len(sampling_results[i].pos_bboxes)][inds], dim=0))
        #     mean_id_pred = torch.stack(mean_id_pred, dim=0)
        #
        #     S_all = mean_id_pred @ mean_id_pred.transpose(-2, -1)
        #     S_person = gt_id_pred @ gt_id_pred.transpose(-2, -1)
        #
        #     if len(mean_id_pred) > 0:
        #         loss_sim = 1 / len(mean_id_pred) * sum(1 - mean_id_pred[i].unsqueeze(dim=0) @ gt_id_pred[i].unsqueeze(dim=1) for i in range(len(mean_id_pred)))
        #         print(loss_sim.item())
        #         assert not torch.isnan(loss_sim).any()
        #         loss_sim = mean_id_pred.new_zeros((1,))
        #         loss_kl = F.kl_div(S_all, S_person) + F.kl_div(S_person, S_all)
        #         print(loss_kl.item())
        #         assert not torch.isnan(loss_kl).any()
        #         loss_kl = mean_id_pred.new_zeros((1,))
        #
        #     else:
        #         loss_sim = 0
        #         loss_kl = 0
        #     losses['loss_sim'] = loss_sim
        #     losses['loss_kl'] = loss_kl
        # # F.log_softmax(pred_T, dim=1)

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
        #reid loss
        # print(id_labels)
        # labeled_matching_scores = self.labeled_matching_layer(id_pred, id_labels)
        # labeled_matching_scores *= 10
        # unlabeled_matching_scores = self.unlabeled_matching_layer(id_pred, id_labels)
        # unlabeled_matching_scores *= 10
        # matching_scores = torch.cat((labeled_matching_scores, unlabeled_matching_scores), dim=1)
        # pid_labels = id_labels.clone()
        # pid_labels[pid_labels == -2] = -1
        # # print(labels, pid_labels)
        # loss_oim = F.cross_entropy(matching_scores, pid_labels, ignore_index=-1)

        # rid_pred = id_pred[id_labels != -2]
        # rid_labels = id_labels[id_labels != -2]
        # losses['loss_id'] = self.loss_reid(rid_pred, rid_labels) * self.reid_loss_weight



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
                det_ids = det_bboxes.new_zeros((0, 256))
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
