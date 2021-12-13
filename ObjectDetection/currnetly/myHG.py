#from typing_extensions import Unpack
import torch
import torch.nn as nn
#from cf_utils import *
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Residual, self).__init__()
        
        self.residual_function = nn.Sequential(
            nn.ReLU(), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(), nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.ReLU(), nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False))
        
        self.skip_layer = nn.Sequential(
            nn.ReLU(), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False))

        #if in_channels == out_channels:self.need_skip = False
        #else: self.need_skip = True
    
    def forward(self, x):
        #if self.need_skip: residual = self.skip_layer(x)
        #else:residual = x
        out = self.residual_function(x)
        residual = self.skip_layer(x)
        out += residual
        return out

class Top_Left_Corner(nn.Module):
    def __init__(self, in_channels):
        super(Top_Left_Corner, self).__init__()
        self.in_channels = in_channels
        
    def _top_pool(self,x):
        for c in range(x.size(dim=0)):
            for i in range(x.size(dim=1)):
                for k in range(x.size(dim=3)):
                    for j in range(x.size(dim=2)-1, 0, -1):
                        if x[c][i][j][k] > x[c][i][j-1][k]:
                            x[c][i][j-1][k] = x[c][i][j][k]
        return x

    def _left_pool(self,x):
        for c in range(x.size(dim=0)):
            for i in range(x.size(dim=1)):
                for j in range(x.size(dim=2)):
                    for k in range(x.size(dim=3)-1, 0, -1):
                        if x[c][i][j][k] > x[c][i][j][k-1]:
                            x[c][i][j][k-1] = x[c][i][j][k]
        return x

    def forward(self, x):
        top = self._top_pool(x)
        left = self._left_pool(x)
        out = top + left
        return out

class Bottom_Right_Corner(nn.Module):
    def __init__(self, in_channels):
        super(Bottom_Right_Corner, self).__init__()
        self.in_channels = in_channels

    def _bottom_pool(self,x):
        for c in range(x.size(dim=0)):
            for i in range(x.size(dim=1)):
                for k in range(x.size(dim=3)):
                    for j in range(x.size(dim=2)-1):
                        if x[c][i][j][k] > x[c][i][j+1][k]:
                            x[c][i][j+1][k] = x[c][i][j][k]
        return x

    def _right_pool(self,x):
        for c in range(x.size(dim=0)):
            for i in range(x.size(dim=1)):
                for j in range(x.size(dim=2)):
                    for k in range(x.size(dim=3)-1):
                        if x[c][i][j][k] > x[c][i][j][k+1]:
                            x[c][i][j][k+1] = x[c][i][j][k]
        return x        
        
    def forward(self, x):
        bottom = self._bottom_pool(x)
        right = self._right_pool(x)
        out = bottom + right
        return out

class convolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn   = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn   = self.bn(conv)
        relu = self.relu(bn)
        return relu

def make_kp_layer(cnv_dim, curr_dim, out_dim):
    return nn.Sequential(
        convolution(3, cnv_dim, curr_dim, with_bn=False),
        nn.Conv2d(curr_dim, out_dim, 1)
    )

class HourglassModule(nn.Module):
    def __init__(self, num_feats, num_blocks, num_classes):
        super(HourglassModule, self).__init__()
        self.num_feats = num_feats
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.channels = [256,256,384,384,384,512]
        self.upsample_layer = nn.Upsample(scale_factor=2, mode='nearest') #'bilinear'
        
    #def _conv_layer(self, in_feats, out_feats):
    #    conv = nn.Conv2d(in_feats, out_feats, kernel_size=1, bias=False)
    #    return conv

    def forward(self, x):
        downsample = []
        
        for i in range(self.num_blocks):            
            hg_block = Residual(self.channels[i], self.channels[i+1], stride=2)(x)
            downsample.append(hg_block)
            x = hg_block
        
        hg_block = Residual(self.channels[-2], self.channels[-1], stride=2)(x)
        x = hg_block

        # mid-convolution layer (not yet ~ )
        # x = self._conv_layer(x,self.channels[-1], self.channels[-1], stride=1)

        # upsample
        for i in range(self.num_blocks+1, 1, -1):
            upsample = self.upsample_layer(x)
            upsample = nn.Conv2d(self.channels[i], self.channels[i-1], kernel_size=1, bias=False)(upsample)
            hg_out = downsample[i-2] + upsample
            x = hg_out
        
        hg_out = self.upsample_layer(x)
        
        return hg_out

class HourglassNet(nn.Module):
    def __init__(self, block=Residual, num_stacks=2, num_blocks=4, num_classes=1):
        super(HourglassNet, self).__init__()
        self.in_channels = 64
        self.num_feats = 256
        self.num_stacks = num_stacks
        
        # Initial processing of the image (gpu 사용량이 높아서 HG 들어가기 전에 줄여줘)
        self.conv1 = nn.Conv2d(1, self.in_channels, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = block(self.in_channels, int(self.num_feats/2))
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.layer2 = block(int(self.num_feats/2), int(self.num_feats/2))
        self.layer3 = block(int(self.num_feats/2), self.num_feats)

        # hourglass block
        self.hg_block = HourglassModule(self.num_feats, num_blocks=num_blocks, num_classes=num_classes) 
        
        # tl layer, br layer
        self.tl_layer = Top_Left_Corner(self.num_feats)
        self.br_layer = Bottom_Right_Corner(self.num_feats)

        # torch->2,256,128,128일듯

        ## keypoint heatmaps
        self.tl_heats = make_kp_layer(self.num_feats, self.num_feats, 2)
        self.br_heats = make_kp_layer(self.num_feats, self.num_feats, 2)

        ## tags
        self.tl_tags  = make_kp_layer(self.num_feats, self.num_feats, 1)
        self.br_tags  = make_kp_layer(self.num_feats, self.num_feats, 1)

        #for tl_heat, br_heat in zip(self.tl_heats, self.br_heats):
        #    tl_heat[-1].bias.data.fill_(-2.19)
        #    br_heat[-1].bias.data.fill_(-2.19)

        self.tl_regrs = make_kp_layer(self.num_feats, self.num_feats, 2)
        self.br_regrs = make_kp_layer(self.num_feats, self.num_feats, 2)


        # ??????????? dont use it
        #self.fc_loc = nn.Conv2d(256, 4, kernel_size=1, stride=1)
        #self.fc_scr = nn.Conv2d(256, 2, kernel_size=1, stride=1)


        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.hg_block(x)
        
        # x.size() = torch.size[2,256,128,128]

        tl_cnv = self.tl_layer(x)
        br_cnv = self.br_layer(x)

        tl_heat, br_heat = self.tl_heats(tl_cnv), self.br_heats(br_cnv)
        tl_tag,  br_tag  = self.tl_tags(tl_cnv),  self.br_tags(br_cnv)
        tl_regr, br_regr = self.tl_regrs(tl_cnv), self.br_regrs(br_cnv)

        #tl_tag  = _tranpose_and_gather_feat(tl_tag, tl_inds)
        #br_tag  = _tranpose_and_gather_feat(br_tag, br_inds)
        #tl_regr = _tranpose_and_gather_feat(tl_regr, tl_inds)
        #br_regr = _tranpose_and_gather_feat(br_regr, br_inds)

        outs = [tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr]

        # predicted_scores = self.fc_loc(x)
        # predicted_locs = self.fc_scr(x)
        
        return outs

from torchsummary import summary
device = torch.device('cuda:0')
net = HourglassNet()
summary(net,input_size=(1,512,512),device='cpu')



def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def _neg_loss(preds, gt):
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)

    neg_weights = torch.pow(1 - gt[neg_inds], 4)

    loss = 0
    for pred in preds:
        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]

        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

def _sigmoid(x):
    x = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    return x

def _ae_loss(tag0, tag1, mask):
    num  = mask.sum(dim=1, keepdim=True).float()
    tag0 = tag0.squeeze()
    tag1 = tag1.squeeze()

    tag_mean = (tag0 + tag1) / 2

    tag0 = torch.pow(tag0 - tag_mean, 2) / (num + 1e-4)
    tag0 = tag0[mask].sum()
    tag1 = torch.pow(tag1 - tag_mean, 2) / (num + 1e-4)
    tag1 = tag1[mask].sum()
    pull = tag0 + tag1

    mask = mask.unsqueeze(1) + mask.unsqueeze(2)
    mask = mask.eq(2)
    num  = num.unsqueeze(2)
    num2 = (num - 1) * num
    dist = tag_mean.unsqueeze(1) - tag_mean.unsqueeze(2)
    dist = 1 - torch.abs(dist)
    dist = nn.functional.relu(dist, inplace=True)
    dist = dist - 1 / (num + 1e-4)
    dist = dist / (num2 + 1e-4)
    dist = dist[mask]
    push = dist.sum()
    return pull, push

def _regr_loss(regr, gt_regr, mask):
    num  = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)

    regr    = regr[mask]
    gt_regr = gt_regr[mask]
    
    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss

class AELoss(nn.Module):
    def __init__(self, pull_weight=1, push_weight=1, regr_weight=1, focal_loss=_neg_loss):
        super(AELoss, self).__init__()

        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.regr_weight = regr_weight
        self.focal_loss  = focal_loss
        self.ae_loss     = _ae_loss
        self.regr_loss   = _regr_loss

    def forward(self, outs, targets):

        tl_heats = outs[0]
        br_heats = outs[1]
        tl_tags  = outs[2]
        br_tags  = outs[3]
        tl_regrs = outs[4]
        br_regrs = outs[5]

        gt_tl_heat = targets[0]
        gt_br_heat = targets[1]
        gt_mask    = targets[2]
        gt_tl_regr = targets[3]
        gt_br_regr = targets[4]

        # focal loss
        focal_loss = 0

        tl_heats = _sigmoid(tl_heats)
        br_heats = _sigmoid(br_heats)

        focal_loss += self.focal_loss(tl_heats, gt_tl_heat)
        focal_loss += self.focal_loss(br_heats, gt_br_heat)

        # tag loss
        pull_loss = 0
        push_loss = 0

        for tl_tag, br_tag in zip(tl_tags, br_tags):
            pull, push = self.ae_loss(tl_tag, br_tag, gt_mask)
            pull_loss += pull
            push_loss += push
        pull_loss = self.pull_weight * pull_loss
        push_loss = self.push_weight * push_loss

        regr_loss = 0
        for tl_regr, br_regr in zip(tl_regrs, br_regrs):
            regr_loss += self.regr_loss(tl_regr, gt_tl_regr, gt_mask)
            regr_loss += self.regr_loss(br_regr, gt_br_regr, gt_mask)
        regr_loss = self.regr_weight * regr_loss

        loss = (focal_loss + pull_loss + push_loss + regr_loss) / len(tl_heats)

        
        return loss.unsqueeze(0)


# used
##################################################################################
# un-used

"""
    def forward(self, x):
        out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)

        for i in range(self.num_stacks):
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            out.append(score)
            if i < self.num_stacks-1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_

        return out
"""

# channels, height, width
# CornerPooling (tlpool, brpool)


# heatmaps, embedding, offset 
class Prediction_module(nn.Module):
    """
    Convolutions to predict class scores and bounding boxes using lower and higher-level feature maps.

    The bounding boxes (locations) are predicted as encoded offsets w.r.t each of the 8732 prior (default) boxes.
    See 'cxcy_to_gcxgcy' in utils.py for the encoding definition.

    The class scores represent the scores of each object class in each of the 8732 bounding boxes located.
    A high score for 'background' = no object.
    """

    def __init__(self, n_classes):
        """
        :param n_classes: number of different types of objects
        """
        super(Prediction_module, self).__init__()

        self.n_classes = n_classes

        # Number of prior-boxes we are considering per position in each feature map
        n_boxes = {'conv11_2': 4}
                   
        # Class prediction convolutions (predict classes in localization boxes)
        self.cl_conv11_2 = nn.Conv2d(256, 4 * n_classes, kernel_size=3, padding=1)


    def forward(self, conv4_3_feats, conv11_2_feats):
        """
        Forward propagation.
        :param conv11_2_feats: conv11_2 feature map, a tensor of dimensions (N, 256, 1, 1)
        :return: 8732 locations and class scores (i.e. w.r.t each prior box) for each image
        """
        batch_size = conv4_3_feats.size(0)

        c_conv11_2 = self.cl_conv11_2(conv11_2_feats)  # (N, 4 * n_classes, 1, 1)
        c_conv11_2 = c_conv11_2.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 4 * n_classes)
        c_conv11_2 = c_conv11_2.view(batch_size, -1, self.n_classes)  # (N, 4, n_classes)

        # A total of 8732 boxes
        # Concatenate in this specific order (i.e. must match the order of the prior-boxes)
        locs = torch.cat([l_conv4_3, l_conv7, l_conv8_2, l_conv9_2, l_conv10_2, l_conv11_2], dim=1)  # (N, 8732, 4)
        classes_scores = torch.cat([c_conv11_2], dim=1)  # (N, 8732, n_classes)

        return locs, classes_scores

# generating bounding boxes


class AELoss(nn.Module):
    def __init__(self, pull_weight=1, push_weight=1, regr_weight=1, focal_loss=_neg_loss):
        super(AELoss, self).__init__()

        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.regr_weight = regr_weight
        self.focal_loss  = focal_loss
        self.ae_loss     = _ae_loss
        self.regr_loss   = _regr_loss

    def forward(self, outs, targets):
        stride = 6

        tl_heats = outs[0::stride]
        br_heats = outs[1::stride]
        tl_tags  = outs[2::stride]
        br_tags  = outs[3::stride]
        tl_regrs = outs[4::stride]
        br_regrs = outs[5::stride]

        gt_tl_heat = targets[0]
        gt_br_heat = targets[1]
        gt_mask    = targets[2]
        gt_tl_regr = targets[3]
        gt_br_regr = targets[4]

        # focal loss
        focal_loss = 0

        tl_heats = [_sigmoid(t) for t in tl_heats]
        br_heats = [_sigmoid(b) for b in br_heats]

        focal_loss += self.focal_loss(tl_heats, gt_tl_heat)
        focal_loss += self.focal_loss(br_heats, gt_br_heat)

        # tag loss
        pull_loss = 0
        push_loss = 0

        for tl_tag, br_tag in zip(tl_tags, br_tags):
            pull, push = self.ae_loss(tl_tag, br_tag, gt_mask)
            pull_loss += pull
            push_loss += push
        pull_loss = self.pull_weight * pull_loss
        push_loss = self.push_weight * push_loss

        regr_loss = 0
        for tl_regr, br_regr in zip(tl_regrs, br_regrs):
            regr_loss += self.regr_loss(tl_regr, gt_tl_regr, gt_mask)
            regr_loss += self.regr_loss(br_regr, gt_br_regr, gt_mask)
        regr_loss = self.regr_weight * regr_loss

        loss = (focal_loss + pull_loss + push_loss + regr_loss) / len(tl_heats)

        
        return loss.unsqueeze(0)


class MultiBoxLoss(nn.Module):
    """
    The MultiBox loss, a loss function for object detection.

    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes, and
    (2) a confidence loss for the predicted class scores.
    """

    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy #model.priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold #0.5
        self.neg_pos_ratio = neg_pos_ratio #3
        self.alpha = alpha #1

        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """
        Forward propagation.
        :predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :boxes: true  object bounding boxes in boundary coordinates, a list of N tensors
        :labels: true object labels, a list of N tensors
        :return: multibox loss, a scalar
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)  # (N, 8732, 4)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)  # (N, 8732)

        # For each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            overlap = find_jaccard_overlap(boxes[i],
                                           self.priors_xy)  # (n_objects, 8732)

            # For each prior, find the object that has the maximum overlap
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (8732)

            # We don't want a situation where an object is not represented in our positive (non-background) priors -
            # 1. An object might not be the best object for all priors, and is therefore not in object_for_each_prior.
            # 2. All priors with the object may be assigned as background based on the threshold (0.5).

            # To remedy this -
            # First, find the prior that has the maximum overlap for each object.
            _, prior_for_each_object = overlap.max(dim=1)  # (N_o)

            # Then, assign each object to the corresponding maximum-overlap-prior. (This fixes 1.)
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)

            # To ensure these priors qualify, artificially give them an overlap of greater than 0.5. (This fixes 2.)
            overlap_for_each_prior[prior_for_each_object] = 1.

            # Labels for each prior
            label_for_each_prior = labels[i][object_for_each_prior]  # (8732)
            # Set priors whose overlaps with objects are less than the threshold to be background (no object)
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0  # (8732)

            # Store
            true_classes[i] = label_for_each_prior

            # Encode center-size object coordinates into the form we regressed predicted boxes to
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)  # (8732, 4)

        # Identify priors that are positive (object/non-background)
        positive_priors = true_classes != 0  # (N, 8732)

        # LOCALIZATION LOSS

        # Localization loss is computed only over positive (non-background) priors
        loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors])  # (), scalar

        # Note: indexing with a torch.uint8 (byte) tensor flattens the tensor when indexing is across multiple dimensions (N & 8732)
        # So, if predicted_locs has the shape (N, 8732, 4), predicted_locs[positive_priors] will have (total positives, 4)

        # CONFIDENCE LOSS

        # Confidence loss is computed over positive priors and the most difficult (hardest) negative priors in each image
        # That is, FOR EACH IMAGE,
        # we will take the hardest (neg_pos_ratio * n_positives) negative priors, i.e where there is maximum loss
        # This is called Hard Negative Mining - it concentrates on hardest negatives in each image, and also minimizes pos/neg imbalance

        # Number of positive and hard-negative priors per image
        n_positives = positive_priors.sum(dim=1)  # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

        # First, find the loss for all priors
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))  # (N * 8732)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 8732)

        # We already know which priors are positive
        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

        # Next, find which priors are hard-negative
        # To do this, sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
        conf_loss_neg = conf_loss_all.clone()  # (N, 8732)
        conf_loss_neg[positive_priors] = 0.  # (N, 8732), positive priors are ignored (never in top n_hard_negatives)
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 8732), sorted by decreasing hardness
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)  # (N, 8732)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 8732)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), scalar

        # TOTAL LOSS

        return conf_loss + self.alpha * loc_loss


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
