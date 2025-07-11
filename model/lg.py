import copy
import warnings
from typing import List, Tuple, Union
from collections import OrderedDict

import torch
from torch import Tensor
from torch.nn import Embedding
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torch.nn.utils.rnn import pad_sequence

from mmdet.registry import MODELS
from mmengine.structures import BaseDataElement, InstanceData
from mmdet.structures import SampleList, OptSampleList
from mmdet.structures.bbox import bbox2roi, roi2bbox, scale_boxes
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.detectors.base import BaseDetector
from mmengine.model.base_module import Sequential
from .predictor_heads.reconstruction import ReconstructionHead
from .predictor_heads.modules.loss import ReconstructionLoss
from .predictor_heads.modules.layers import build_mlp
from .predictor_heads.graph import GraphHead
from .predictor_heads.ds import DSHead
from .predictor_heads.modules.utils import dense_mask_to_polygon_mask
from .roi_extractors.sg_single_level_roi_extractor import SgSingleRoIExtractor
from mmdet.models.layers.transformer.utils import coordinate_to_encoding

@MODELS.register_module()
class LGDetector(BaseDetector):
    """Detector that also outputs a scene graph, reconstructed image, and any downstream predictions.

    Args:
        detector (ConfigType): underlying object detector config
        reconstruction_head (ConfigType): reconstruction head config
        reconstruction_loss (ConfigType): reconstruction loss config
        reconstruction_img_stats (ConfigType): reconstructed image mean and std
    """

    def __init__(self, detector: ConfigType, #Given from lg_mask_rcnn.py and $MMDETECTION/configs/_base_/models/mask-rcnn_r50_fpn.py
                num_classes: int, # 6: Responds to object classes in Endoscapes2023
                viz_feat_size: int, #256
                semantic_feat_size: int, #512 
                sem_feat_hidden_dim: int = 2048,
                semantic_feat_projector_layers: int = 3,
                perturb_factor: float = 0.0,
                use_pred_boxes_recon_loss: bool = False, 
                reconstruction_head: ConfigType = None,
                reconstruction_loss: ConfigType = None,
                reconstruction_img_stats: ConfigType = None,
                graph_head: ConfigType = None,
                ds_head: ConfigType = None,
                roi_extractor: ConfigType = None, 
                use_gt_dets: bool = False,
                trainable_detector_cfg: OptConfigType = None,
                trainable_backbone_cfg: OptConfigType = None, 
                force_train_graph_head: bool = False,
                sem_feat_use_class_logits: bool = True, 
                sem_feat_use_bboxes: bool = True,
                sem_feat_use_masks: bool = True, 
                mask_polygon_num_points: int = 16,
                mask_augment: bool = True, 
                force_encode_semantics: bool = False,
                trainable_neck_cfg: OptConfigType = None, **kwargs):
        super().__init__(**kwargs)
        
        self.num_classes = num_classes
        self.detector = MODELS.build(detector)
        self.roi_extractor = MODELS.build(roi_extractor) if roi_extractor is not None else None
        self.use_gt_dets = use_gt_dets
        self.perturb_factor = perturb_factor if not use_gt_dets else 0
        self.viz_feat_size = viz_feat_size

        # if trainable detector cfg is defined, that is used for trainable backbone
        if trainable_detector_cfg is not None:
            self.trainable_backbone = MODELS.build(trainable_detector_cfg)
        elif trainable_backbone_cfg is not None:
            bb = MODELS.build(trainable_backbone_cfg)
            if trainable_neck_cfg is not None:
                neck = MODELS.build(trainable_neck_cfg)
                self.trainable_backbone = Sequential(OrderedDict([
                        ('backbone', bb),
                        ('neck', neck),
                    ])
                )

            else:
                self.trainable_backbone = Sequential(OrderedDict([
                        ('backbone', bb),
                        ('neck', torch.nn.Identity()),
                    ])
                )
        else:
            self.trainable_backbone = None

        # add obj feat size to recon cfg
        if reconstruction_head is not None:
            reconstruction_head.viz_feat_size = viz_feat_size
            reconstruction_head.semantic_feat_size = semantic_feat_size
            self.reconstruction_head = MODELS.build(reconstruction_head)
        else:
            self.reconstruction_head = None

        self.reconstruction_loss = MODELS.build(reconstruction_loss) if reconstruction_loss is not None else None
        self.reconstruction_img_stats = reconstruction_img_stats if reconstruction_img_stats is not None else None

        # add roi extractor to graph head
        if graph_head is not None:
            graph_head.viz_feat_size = viz_feat_size
            graph_head.roi_extractor = self.roi_extractor
            self.graph_head = MODELS.build(graph_head)
            self.num_edge_classes = graph_head.num_edge_classes

        else:
            self.graph_head = None
            self.num_edge_classes = 0

        self.ds_head = MODELS.build(ds_head) if ds_head is not None else None

        # whether to force graph head training when detector is frozen
        self.force_train_graph_head = force_train_graph_head

        # use pred boxes or gt boxes for recon loss
        self.use_pred_boxes_recon_loss = use_pred_boxes_recon_loss

        # only encode semantics if needed
        self.encode_semantics = force_encode_semantics or ((self.ds_head is not None or \
                self.reconstruction_head is not None) and semantic_feat_size > 0)
        if self.encode_semantics:
            # build semantic feat projector (input feat size is classes+box coords+score)
            self.sem_feat_use_bboxes = sem_feat_use_bboxes
            self.sem_feat_use_class_logits = sem_feat_use_class_logits
            self.sem_feat_use_masks = sem_feat_use_masks
            self.mask_augment = mask_augment
            self.mask_polygon_num_points = mask_polygon_num_points
            self.semantic_feat_size = semantic_feat_size

            # compute sem_input_dim
            sem_input_dim = 1 # scores
            edge_sem_input_dim = 0 # don't factor in score for edge
            if self.sem_feat_use_bboxes:
                # boxes and score
                sem_input_dim += 4
                edge_sem_input_dim += 4

            if self.sem_feat_use_class_logits:
                # class logits
                sem_input_dim += num_classes
                edge_sem_input_dim += self.num_edge_classes

            if self.sem_feat_use_masks:
                sem_input_dim += self.mask_polygon_num_points * 2 # number of points, x and y coord

            dim_list = [sem_input_dim] + [sem_feat_hidden_dim] * (semantic_feat_projector_layers - 1) + [semantic_feat_size]
            self.semantic_feat_projector = build_mlp(dim_list, batch_norm='batch')

            edge_dim_list = [edge_sem_input_dim] + dim_list[1:]
            self.edge_semantic_feat_projector = build_mlp(edge_dim_list, batch_norm='batch')

    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList):
        
        #Batch inputs: [B, C, H, W]
        #Batch data samples: List with B elements, each element is a BaseDataElement
        
        if self.detector.training:
            losses = self.detector.loss(batch_inputs, batch_data_samples)
            #breakpoint()
        else:
            losses = {} #When training downstream heads, we don't want to use detector losses

        #breakpoint()
        # extract LG (gets non-differentiable predicted detections, and differentiable predicted graph)
        feats, graph, results, gt_edges, losses = self.extract_lg(batch_inputs,
                batch_data_samples, losses=losses)

        # use feats and detections to reconstruct img
        if self.reconstruction_head is not None:
            reconstructed_imgs, img_targets, rescaled_results = self.reconstruction_head.predict(
                    results, feats, graph, batch_inputs)

            recon_boxes = []
            for r in rescaled_results:
                if r.is_det_keyframe and not self.use_pred_boxes_recon_loss:
                    recon_boxes.append(r.gt_instances.bboxes)
                else:
                    recon_boxes.append(r.pred_instances.bboxes)

            reconstruction_losses = self.reconstruction_loss(reconstructed_imgs,
                    img_targets, recon_boxes)

            # update losses
            losses.update(reconstruction_losses)

        if self.ds_head is not None:
            try:
                ds_losses = self.ds_head.loss(graph, feats, batch_data_samples)
                losses.update(ds_losses)
            except AttributeError as e:
                print(e)
                raise NotImplementedError("Must have graph head in order to do downstream prediction")

        # breakpoint()
        
        return losses

    def predict(self, batch_inputs: Tensor, batch_data_samples: SampleList,
            rescale: bool = True) -> SampleList:
        # extract LG
        feats, graph, results, gt_edges, losses = self.extract_lg(batch_inputs,
                batch_data_samples)

        if graph is not None:
            # add latent graph to results
            results = self.add_lg_to_results(results, feats, graph)

            # add scene graph to result
            results = self.add_scene_graph_to_results(results, gt_edges, graph)

        # use feats and detections to reconstruct img
        if self.reconstruction_head is not None:
            reconstructed_imgs, _, _ = self.reconstruction_head.predict(results,
                    feats, graph, batch_inputs)

            for r, r_img in zip(results, reconstructed_imgs):
                # renormalize img
                norm_r_img = r_img * Tensor(self.reconstruction_img_stats.std).view(-1, 1, 1).to(r_img.device) / 255 + \
                        Tensor(self.reconstruction_img_stats.mean).view(-1, 1, 1).to(r_img.device) / 255
                r.reconstruction = torch.clamp(norm_r_img, 0, 1)

        if self.ds_head is not None:
            try:
                ds_preds, _ = self.ds_head.predict(graph, feats)
            except AttributeError:
                raise NotImplementedError("Must have graph head in order to do downstream prediction")

            for r, dp in zip(results, ds_preds):
                r.pred_ds = dp

        # rescale results if needed
        if rescale:
            scale_factor = 1 / torch.Tensor(results[0].scale_factor)
            for r in results:
                r.pred_instances.bboxes = scale_boxes(r.pred_instances.bboxes,
                        scale_factor.tolist())
                if 'masks' in r.pred_instances:
                    if r.pred_instances.masks.shape[0] == 0:
                        r.pred_instances.masks = torch.zeros((0, *r.ori_shape)).to(r.pred_instances.masks)
                    else:
                        r.pred_instances.masks = F.interpolate(r.pred_instances.masks.unsqueeze(0).float(),
                                size=r.ori_shape).bool().squeeze(0)

        return results

    def add_scene_graph_to_results(self, results: SampleList, gt_edges: BaseDataElement,
            graph: BaseDataElement) -> SampleList:
        for ind, r in enumerate(results):
            # GT
            r.gt_edges = InstanceData()
            r.gt_edges.edge_flats = gt_edges.edge_flats[ind]
            r.gt_edges.edge_boxes = gt_edges.edge_boxes[ind]
            r.gt_edges.relations = gt_edges.edge_relations[ind]

            # PRED
            r.pred_edges = InstanceData()

            # select correct batch
            batch_inds = graph.edges.edge_flats[:, 0] == ind
            r.pred_edges.edge_flats = graph.edges.edge_flats[batch_inds][:, 1:] # remove batch id
            r.pred_edges.edge_boxes = graph.edges.boxes[ind] # already a list
            r.pred_edges.relations = graph.edges.class_logits[batch_inds]

        return results

    def add_lg_to_results(self, results: SampleList, feats: BaseDataElement,
            graph: BaseDataElement) -> SampleList:
        for batch_ind, r in enumerate(results):
            # extract graph for frame i, add to result
            g = BaseDataElement()
            g.nodes = BaseDataElement()
            g.edges = BaseDataElement()
            g.nodes.viz_feats = feats.instance_feats[batch_ind]
            g.nodes.gnn_viz_feats = graph.nodes.gnn_viz_feats[batch_ind]
            if 'semantic_feats' in feats:
                g.nodes.semantic_feats = feats.semantic_feats[batch_ind]

            g.nodes.nodes_per_img = graph.nodes.nodes_per_img[batch_ind]
            g.nodes.bboxes = r.pred_instances.bboxes
            g.nodes.scores = r.pred_instances.scores
            g.nodes.labels = r.pred_instances.labels
            if 'masks' in r.pred_instances:
                g.nodes.masks = r.pred_instances.masks

            # split edge quantities and add
            epi = graph.edges.edges_per_img.tolist()
            for k in graph.edges.keys():
                if k in ['batch_index', 'presence_logits', 'edges_per_img']:
                    continue

                elif k == 'edge_flats':
                    val = graph.edges.get(k).split(epi)[batch_ind][:, 1:]

                elif not isinstance(graph.edges.get(k), Tensor):
                    # no need to split, just index
                    val = graph.edges.get(k)[batch_ind]

                else:
                    val = graph.edges.get(k).split(epi)[batch_ind]

                g.edges.set_data({k: val})

            # pool img feats and add to graph
            g.img_feats = F.adaptive_avg_pool2d(feats.bb_feats[-1][batch_ind], 1).squeeze()

            # add img shape to results
            g.ori_shape = r.ori_shape
            g.batch_input_shape = r.batch_input_shape

            # add LG to results
            r.lg = g

        return results

    def extract_lg(self, batch_inputs: Tensor, batch_data_samples: SampleList,
            force_perturb: bool = False, losses: dict = None,
            clip_size: int = -1) -> Tuple[BaseDataElement]:

        
        # run detector to get detections
        with torch.no_grad():
            results = self.detector.predict(batch_inputs, batch_data_samples, rescale=False)

        #breakpoint()
        # get bb and fpn features
        feats = self.extract_feat(batch_inputs, results, force_perturb, clip_size)

        # update feat of each pred instance
        for ind, r in enumerate(results):
            r.pred_instances.feats = feats.instance_feats[ind][:r.pred_instances.bboxes.shape[0]]

        # run graph head
        if self.graph_head is not None:
            if self.detector.training:
                # train graph with gt boxes (only when detector is training)
                graph_losses, graph = self.graph_head.loss_and_predict(
                        results, feats)
                losses.update(graph_losses)
                gt_edges = None

            else:
                if self.force_train_graph_head and self.training:
                    graph_losses, graph = self.graph_head.loss_and_predict(
                            results, feats)
                    losses.update(graph_losses)
                    gt_edges = None

                else:
                    graph, gt_edges = self.graph_head.predict(results, feats)

            if self.encode_semantics:
                # compute semantic feats
                self.compute_semantic_feat(results, feats, graph)

            # update feat of each pred instance
            for ind, r in enumerate(results):
                r.pred_instances.graph_feats = graph.nodes.gnn_viz_feats[ind, :r.pred_instances.bboxes.shape[0]]

        else:
            graph = None
            gt_edges = None

        # breakpoint()

        return feats, graph, results, gt_edges, losses

    def detach_results(self, results: SampleList) -> SampleList:
        for i in range(len(results)):
            results[i].pred_instances.bboxes = results[i].pred_instances.bboxes.detach()
            results[i].pred_instances.labels = results[i].pred_instances.labels.detach()

        return results

    def extract_feat(self, batch_inputs: Tensor, results: SampleList, force_perturb: bool,
            clip_size: int) -> BaseDataElement:
        feats = BaseDataElement()

        # load pred/gt dense labels
        use_masks = 'masks' in results[0].pred_instances
        if self.use_gt_dets and self.training: # only use gt dets when training
            # breakpoint()
            boxes = [r.gt_instances.bboxes.to(batch_inputs.device) \
                    if (r.is_det_keyframe and len(r.gt_instances) > 0) \
                    else Tensor([]).to(batch_inputs.device) if r.is_det_keyframe \
                    else r.pred_instances.bboxes for r in results]
            classes = [r.gt_instances.labels.to(batch_inputs.device) \
                    if (r.is_det_keyframe and len(r.gt_instances) > 0) \
                    else Tensor([]).to(batch_inputs.device) if r.is_det_keyframe \
                    else r.pred_instances.labels for r in results]
            scores = [torch.ones_like(r.pred_instances.labels) \
                    if (r.is_det_keyframe and len(r.gt_instances) > 0) \
                    else Tensor([]).to(batch_inputs.device) if r.is_det_keyframe \
                    else r.pred_instances.scores for r in results]
            masks = None
            if use_masks:
                masks = [r.gt_instances.masks.to(batch_inputs.device) \
                        if (r.is_det_keyframe and len(r.gt_instances) > 0) \
                        else torch.zeros([0, *r.ori_shape]).to(batch_inputs.device) \
                        if r.is_det_keyframe else r.pred_instances.masks for r in results]

        else:
            boxes = [r.pred_instances.bboxes for r in results]
            classes = [r.pred_instances.labels for r in results]
            scores = [r.pred_instances.scores for r in results]
            masks = None
            if use_masks:
                masks = [r.pred_instances.masks.to(batch_inputs.device) for r in results]

        # apply box perturbation
        if (self.training or force_perturb) and self.perturb_factor > 0:
            boxes = self.box_perturbation(boxes, results[0].ori_shape, clip_size)

        # run bbox feat extractor and add instance feats to feats
        if self.roi_extractor is not None:
            if self.trainable_backbone is not None:
                backbone = self.trainable_backbone.backbone #ResNet
                neck = self.trainable_backbone.neck #FPN 
            else:
                backbone = self.detector.backbone
                neck = self.detector.neck if self.detector.with_neck else torch.nn.Identity()

            bb_feats = backbone(batch_inputs) #[B, ft:256, 120, 214]
            neck_feats = neck(bb_feats)

            feats.bb_feats = bb_feats
            feats.neck_feats = neck_feats

            # convert bboxes to roi and extract roi feats
            rois = bbox2roi(boxes)
            roi_input_feats = feats.neck_feats if feats.neck_feats is not None else feats.bb_feats
            if isinstance(self.roi_extractor, SgSingleRoIExtractor) and 'masks' in results[0].pred_instances:
                roi_feats = self.roi_extractor(
                    roi_input_feats[:self.roi_extractor.num_inputs], rois, masks=masks,
                )

            else:
                roi_feats = self.roi_extractor(
                    roi_input_feats[:self.roi_extractor.num_inputs], rois
                )

            # pool feats and split into list
            boxes_per_img = [len(b) for b in boxes]
            feats.instance_feats = pad_sequence(roi_feats.squeeze(-1).squeeze(-1).split(boxes_per_img),
                    batch_first=True)

        else:
            # instance feats are just queries (run detector.get_queries to get)
            if self.trainable_backbone is not None:
                if 'SAM' in type(self.trainable_backbone).__name__:
                    # extract selected indices from results
                    selected_inds = [r.pred_instances['instance_ids'] for r in results]

                    # use trainable backbone to get both img and instance feats
                    feats.bb_feats, instance_feats = self.trainable_backbone.extract_feat(
                            batch_inputs, results, compute_instance_feats=True,
                            selected_inds=selected_inds)
                    feats.neck_feats = feats.bb_feats
                    feats.instance_feats = pad_sequence([i[:, 0] for i in instance_feats],
                            batch_first=True)

                else:
                    feats.bb_feats, feats.neck_feats, feats.instance_feats = \
                            self.trainable_backbone.get_queries(batch_inputs, results)

            else:
                if 'SAM' in type(self.detector).__name__:
                    if 'img_feats' in results[0]:
                        feats.bb_feats = torch.cat([r.img_feats for r in results])
                    else:
                        feats.bb_feats, _ = self.detector.extract_feat(batch_inputs, results)

                    feats.neck_feats = feats.bb_feats
                    feats.instance_feats = pad_sequence([r.pred_instances['feats'] \
                            for r in results], batch_first=True)

                else:
                    feats.bb_feats, feats.neck_feats, feats.instance_feats = \
                            self.detector.get_queries(batch_inputs, results)

        # breakpoint()
        
        return feats

    def compute_semantic_feat(self, results: SampleList, feats: BaseDataElement,
            graph: BaseDataElement) -> Tensor:
        device = feats.instance_feats.device
        if self.use_gt_dets and self.training:
            boxes = [r.gt_instances.bboxes.to(device) if r.is_det_keyframe \
                    else r.pred_instances.bboxes for r in results]
            classes = [r.gt_instances.labels.to(device) if r.is_det_keyframe \
                    else r.pred_instances.labels for r in results]
            scores = [torch.ones_like(r.gt_instances.classes) if r.is_det_keyframe \
                    else r.pred_instances.scores for r in results]
            masks = None
            if 'masks' in results[0].gt_instances:
                masks = [r.gt_instances.masks.to(device) \
                        if r.is_det_keyframe and 'masks' in r.gt_instances \
                        else r.pred_instances.masks for r in results]

        else:
            boxes = [r.pred_instances.bboxes for r in results]
            classes = [r.pred_instances.labels for r in results]
            scores = [r.pred_instances.scores for r in results]
            masks = None
            if 'masks' in results[0].pred_instances:
                masks = [r.pred_instances.masks for r in results]

        # compute semantic feat
        c = pad_sequence(classes, batch_first=True)
        b = pad_sequence(boxes, batch_first=True)
        s = pad_sequence(scores, batch_first=True)
        b_norm = b / Tensor(results[0].ori_shape).flip(0).repeat(2).to(b.device)
        c_one_hot = F.one_hot(c, num_classes=self.num_classes)

        sem_feat_input = []
        if self.sem_feat_use_bboxes:
            sem_feat_input.append(b_norm)

        if self.sem_feat_use_class_logits:
            sem_feat_input.append(c_one_hot)

        # process masks
        if self.sem_feat_use_masks:
            # iterate through masks and convert to polygon mask
            polygon_masks = self.masks_to_polygons(masks)

            # process masks
            polygon_masks = pad_sequence(polygon_masks, batch_first=True) # B x N x P x 2
            polygon_masks_norm = polygon_masks / Tensor(results[0].ori_shape).flip(0).to(polygon_masks.device)

            sem_feat_input.append(polygon_masks_norm.flatten(start_dim=-2))

        sem_feat_input.append(s.unsqueeze(-1))

        sem_feat_input = torch.cat(sem_feat_input, -1).flatten(end_dim=1)
        if sem_feat_input.shape[0] == 1:
            s = self.semantic_feat_projector(torch.cat([sem_feat_input, sem_feat_input]))[0]
        else:
            s = self.semantic_feat_projector(sem_feat_input)

        feats.semantic_feats = s.view(b_norm.shape[0], b_norm.shape[1], s.shape[-1])

        if graph is not None:
            # compute edge semantic feats
            eb_norm = torch.cat(graph.edges.boxes) / Tensor(results[0].batch_input_shape).flip(0).repeat(2).to(graph.edges.boxes[0].device) # make 0-1
            edge_sem_input = torch.cat([eb_norm, graph.edges.class_logits.detach()], -1) # detach class logits to prevent backprop
            if edge_sem_input.shape[1] == 1:
                edge_sem_feats = self.edge_semantic_feat_projector(edge_sem_input.repeat(2, 1))[0].unsqueeze(0)
            else:
                edge_sem_feats = self.edge_semantic_feat_projector(edge_sem_input)

            graph.edges.semantic_feats = edge_sem_feats

    def masks_to_polygons(self, masks: List[Tensor]) -> List[Tensor]:
        polygon_masks = []
        for m in masks:
            p_m = []
            if m.shape[0] > 0:
                for i in m:
                    p_m_i = dense_mask_to_polygon_mask(i, self.mask_polygon_num_points)
                    if self.training and self.mask_augment: # only augment mask at train time
                        p_m_i = torch.roll(p_m_i, torch.randint(p_m_i.shape[0], (1,)).item(), 0)

                    p_m.append(p_m_i)

                p_m = torch.stack(p_m) # N x P x 2

            else:
                p_m = torch.zeros(0, self.mask_polygon_num_points, 2).to(m.device)

            polygon_masks.append(p_m)

        return polygon_masks

    def box_perturbation(self, boxes: List[Tensor], image_shape: Tuple, clip_size: int):
        boxes_per_img = [len(b) for b in boxes]
        perturb_factor = min(self.perturb_factor, 1)
        xmin, ymin, xmax, ymax = torch.cat(boxes).unbind(1)

        # compute x and y perturbation ranges
        h = xmax - xmin
        w = ymax - ymin

        # if clip size is given (e.g. not -1), then apply same perturbation to all imgs in clip
        if clip_size != -1:
            num_clips = int(len(boxes) / clip_size)
            boxes_per_clip = torch.stack(boxes_per_img).view(-1, clip_size).sum(-1)
            perturb_amount = torch.rand(num_clips, 4).to(boxes[0]).repeat_interleave(boxes_per_clip)
            perturb = perturb_factor * (perturb_amount * torch.stack([h, w], dim=1).repeat(1, 2) - \
                    torch.stack([h, w], dim=1).repeat(1, 2))

        else:
            # generate random numbers drawn from (-h, h), (-w, w), multiply by perturb factor
            perturb = perturb_factor * (torch.rand(h.shape[0], 4).to(boxes[0]) * \
                    torch.stack([h, w], dim=1).repeat(1, 2) - \
                    torch.stack([h, w], dim=1).repeat(1, 2))
            perturbed_boxes = torch.cat(boxes) + perturb

        # ensure boxes are valid (clamp from 0 to img shape)
        perturbed_boxes = torch.maximum(torch.zeros_like(perturbed_boxes), perturbed_boxes)
        stacked_img_shapes = Tensor(image_shape).flip(0).unsqueeze(0).repeat(perturbed_boxes.shape[0],
                2).to(perturbed_boxes)
        perturbed_boxes = torch.minimum(stacked_img_shapes, perturbed_boxes)

        return perturbed_boxes.split(boxes_per_img)

    def _forward(self, batch_inputs: Tensor, batch_data_samples: OptSampleList = None):
        raise NotImplementedError

    def to(self, *args, **kwargs) -> torch.nn.Module:
        self.detector = self.detector.to(*args, **kwargs)

        return super().to(*args, **kwargs)
