import ever as er
import numpy as np
import torch
import torch.nn as nn
from core import loss, field
from core.head import get_detector

MAX_TIMES = 50


def generate_target(x1, y):
    # x [N, C * 1, H, W]
    # y dict(mask1=tensor[N, H, W], ...)
    mask1 = y[field.MASK1]
    N = x1.size(0)
    org_inds = np.arange(N)
    t = 0
    while True and t <= MAX_TIMES:
        t += 1
        shuffle_inds = org_inds.copy()
        np.random.shuffle(shuffle_inds)

        ok = org_inds == shuffle_inds
        if all(~ok):
            break
    virtual_x2 = x1[shuffle_inds, :, :, :]
    virtual_mask2 = mask1[shuffle_inds, ...]
    x = torch.cat([x1, virtual_x2], dim=1)

    y[field.VMASK2] = virtual_mask2
    return x, y


class ChangeMixin(nn.Module):
    def __init__(self,
                 feature_extractor,
                 classifier,
                 detector_config,
                 loss_config):
        super(ChangeMixin, self).__init__()
        self.features = feature_extractor
        self.classifier = classifier
        self.detector_config = detector_config
        self.change_detector = get_detector(**detector_config)
        self.loss_config = er.config.AttrDict.from_dict(loss_config)

    def extract_feature(self, x):
        return self.features(x)

    def classify(self, x):
        return self.classifier(x)

    def forward(self, x, y=None):
        if self.training:
            if x.size(1) == 3:
                x, y = generate_target(x, y)

            x1 = x[:, :3, :, :]
            vx2 = x[:, 3:, :, :]

            y1_feature = self.extract_feature(x1)
            vy2_feature = self.extract_feature(vx2)

            y1_pred = self.classify(y1_feature)

            # extract positive feature
            if self.detector_config.get('t1t2', True):
                change_y1vy2_logit = self.change_detector(torch.cat([y1_feature, vy2_feature], dim=1))
            else:
                change_y1vy2_logit = None
            if self.detector_config.get('t2t1', True):
                change_y2vy1_logit = self.change_detector(torch.cat([vy2_feature, y1_feature], dim=1))
            else:
                change_y2vy1_logit = None

            y1_true = y[field.MASK1]
            vy2_true = y[field.VMASK2]

            loss_dict = dict()
            loss_dict.update(loss.misc_info(y1_pred.device))

            if self.detector_config.get('symmetry_loss', False):
                loss_dict.update(
                    loss.semantic_and_symmetry_loss(y1_true,
                                                    vy2_true,
                                                    y1_pred,
                                                    change_y1vy2_logit,
                                                    change_y2vy1_logit,
                                                    self.loss_config))
            else:
                raise ValueError()

            return loss_dict

        x1 = x[:, :3, :, :]
        x2 = x[:, 3:, :, :]

        y1_feature = self.extract_feature(x1)
        y2_feature = self.extract_feature(x2)

        y1_pred = self.classify(y1_feature)

        change_y1y2_logit = self.change_detector(torch.cat([y1_feature, y2_feature], dim=1))

        y2_pred = self.classify(y2_feature)

        logit1 = y1_pred
        logit2 = y2_pred
        change_logit = change_y1y2_logit
        return torch.cat([logit1, logit2, change_logit], dim=1)
