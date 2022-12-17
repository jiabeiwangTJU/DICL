import torch
import torch.nn as nn
# from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import DETECTORS, build_backbone, build_head, build_neck, build_roi_extractor
from .base import BaseDetector
import torchvision
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
import torch.nn.functional as F

@DETECTORS.register_module()
class TwoStageDetectorsiamese(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 gt_assigner=dict(
                     type='MaxIoUAssigner',
                     pos_iou_thr=0.6,  ##0.7
                     neg_iou_thr=0.1,  # 0.5,
                     min_pos_iou=0.5,
                     match_low_quality=False,
                     ignore_iof_thr=-1),
                 gt_roi_extractor=dict(
                     type='SingleRoIExtractor',
                     roi_layer=dict(type='RoIAlign', output_size=tuple([224, 96]), sampling_ratio=0),
                     # 256  h, w[192, 64]
                     out_channels=3,
                     featmap_strides=[1]),
                 mask_ratio=0.2,
                 pixel_mask=True,
                 num_mask_patch=1,
                 pro_mask=0.5,
                 use_mask=False,
                 mask_up=True,
                 mask_down=False,
                 ):
        super(TwoStageDetectorsiamese, self).__init__()
        self.backbone = build_backbone(backbone)
        self.mask_ratio = mask_ratio
        self.pixel_mask = pixel_mask 
        self.num_mask_patch = num_mask_patch
        self.pro_mask = pro_mask
        self.use_mask = use_mask
        self.mask_up = mask_up
        self.mask_down = mask_down

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = build_head(roi_head)

        gt_sampler=dict(
            type='RandomSampler',
            num=128, #512,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=True)

        ######## gt-gt : ensure every gt included in samples ###########
        # gt_sampler=dict(
        #     type='RandomSampler_allgt',
        #     num=128, #512,
        #     pos_fraction=0.5,
        #     neg_pos_ub=-1,
        #     add_gt_as_proposals=True)

        self.init_gt(gt_roi_extractor)

        self.init_assigner_sampler(gt_assigner, gt_sampler)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    def init_assigner_sampler(self, gt_assigner, gt_sampler):
        """Initialize assigner and sampler."""

        self.gt_assigner = build_assigner(gt_assigner)
        self.gt_sampler = build_sampler(gt_sampler)

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(TwoStageDetectorsiamese, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_roi_head:
            self.roi_head.init_weights(pretrained)

        self.gt_roi_extractor.init_weights()

    def init_gt(self, gt_roi_extractor):
        """Initialize ``bbox_head``"""
        self.gt_roi_extractor = build_roi_extractor(gt_roi_extractor)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def gt_align(self, img, gt_bboxes):
        gt_rois=[]
        for i in range(len(gt_bboxes)):
            gt_rois.append(torch.cat((gt_bboxes[i].new_full((len(gt_bboxes[i]), 1), i).cuda(), gt_bboxes[i]), dim=1))
        gt_rois = torch.cat(gt_rois, dim=0)
        gt_bbox_feats = self.gt_roi_extractor(
            [img], gt_rois)
        return gt_bbox_feats

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        # proposals = torch.randn(1000, 4).to(img.device)
        #for faster rcnn person search, only 300 proposals
        proposals = torch.randn(300, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        ########Add mask on person region
        if self.use_mask:
            with torch.no_grad():
                ori_img = img.clone()
                C_means = F.adaptive_avg_pool2d(img, (1, 1))
                for i in range(img.shape[0]):
                    C_mean = C_means[i]
                    for j in range(gt_bboxes[i].shape[0]):
                        x1, y1, x2, y2 = (gt_bboxes[i][j] + 1).to(torch.int).cpu().numpy()
                        if (x2 - x1) < 4 or (y2 - y1) < 4:
                            break
                        img_gtbbox = img[i, :, y1: y2, x1: x2]
                        if torch.rand(1) < self.pro_mask:
                            if self.pixel_mask:
                                mask = torch.rand(img_gtbbox.shape[-2:]).type_as(img_gtbbox) <= self.mask_ratio
                                img_gtbbox[:, mask] = C_mean.squeeze(-1)
                            else:
                                h = int((y2 - y1) / 14)
                                w = int((x2 - x1) / 6)
                                center_x = torch.randint(6, (self.num_mask_patch,)) * w
                                center_y = torch.randint(14, (self.num_mask_patch,)) * h
                                mask = torch.zeros(img_gtbbox.shape[-2:]).type_as(img_gtbbox)
                                for n in range(self.num_mask_patch):
                                    mask[center_y[n]: center_y[n] + h, center_x[n]: center_x[n] + w] = 1
                                img_gtbbox[:, mask.bool()] = C_mean.squeeze(-1)

            img = img.detach()

        kwargs.pop('epoch')   ####use epoch for training (try)
        if self.use_mask:
            if self.mask_up:
                x = self.extract_feat(img)
            else:
                x = self.extract_feat(ori_img)
        else:
            x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.gt_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.gt_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)
                # print('proposal', len(proposal_list[i]), 'gt', assign_result.num_gts, 'ass_pos', sum(assign_result.gt_inds >= 0), 'ass_neg', sum(assign_result.gt_inds < 0))

        ############# pos-pos #############
        # pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        # proposal_img = self.gt_align(img, pos_bboxes_list)
        # gt_x = self.extract_feat(proposal_img)

        if self.use_mask:
            if self.mask_down:
                gt_img = self.gt_align(img, gt_bboxes)
            else:
                gt_img = self.gt_align(ori_img, gt_bboxes)
        else:
            gt_img = self.gt_align(img, gt_bboxes)
        gt_x = self.extract_feat(gt_img)

        roi_losses = self.roi_head.forward_train(x, gt_x, img_metas, sampling_results,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)

        losses.update(roi_losses)

        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        ##############Add mask on test set

        # num_mask_patch = 40
        # pixel_mask = False #False
        # mask_ratio = 0.2
        # C_means = F.adaptive_avg_pool2d(img, (1, 1))  # torch.mean(img[i], dim=1, keepdim=True)
        # for i in range(img.shape[0]):
        #     C_mean = C_means[i]
        #     img_gtbbox = img[i]
        #     if pixel_mask:
        #         mask = torch.rand(img_gtbbox.shape[-2:]).type_as(img_gtbbox) <= mask_ratio
        #         img_gtbbox[:, mask] = C_mean.squeeze(-1)
        #     else:
        #         h = int(img.shape[2] / 14)
        #         w = int(img.shape[3] / 14)
        #         center_x = torch.randint(14, (num_mask_patch,)) * w
        #         center_y = torch.randint(14, (num_mask_patch,)) * h
        #         mask = torch.zeros(img_gtbbox.shape[-2:]).type_as(img_gtbbox)
        #
        #         for n in range(num_mask_patch):
        #             mask[center_y[n]: center_y[n] + h, center_x[n]: center_x[n] + w] = 1
        #         img_gtbbox[:, mask.bool()] = C_mean.squeeze(-1)

        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals
        ############################# concate cropped proposals feature  else gt_x=None
        if proposal_list[0].shape[1] == 5:
            proposal_img = self.gt_align(img, [proposal_list[0][:, :-1]])
        if proposal_list[0].shape[1] == 4:
            proposal_img = self.gt_align(img, [proposal_list[0]])
        gt_x = self.extract_feat(proposal_img)

        return self.roi_head.simple_test(
            x, gt_x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)
