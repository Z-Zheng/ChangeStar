import ever as er
import torch.nn as nn

from core.mixin import ChangeMixin
from module.segmentation import Segmentation


@er.registry.MODEL.register()
class ChangeStar(er.ERModule):
    def __init__(self, config):
        super().__init__(config)

        segmentation = Segmentation(self.config.segmenation)

        layers = [nn.Conv2d(self.config.classifier.in_channels, self.config.classifier.out_channels, 3, 1, 1),
                  nn.UpsamplingBilinear2d(scale_factor=self.config.classifier.scale)]
        classifier = nn.Sequential(*layers)

        self.changemixin = ChangeMixin(segmentation, classifier, self.config.detector, self.config.loss_config)

    def forward(self, x, y=None):
        if self.training or x.size(1) == 6:
            # segmentation + change detection
            return self.changemixin(x, y)

        if x.size(1) == 3:
            # only segmentation
            seg_logit = self.changemixin.classify(self.changemixin.extract_feature(x))
            return seg_logit.sigmoid()

    def set_default_config(self):
        self.config.update(dict(
            segmenation=dict(),
            classifier=dict(
                in_channels=256,
                out_channels=1,
                scale=4.0
            ),
            detector=dict(
                name='convs',
                in_channels=256 * 2,
                inner_channels=16,
                out_channels=1,
                num_convs=4,
            ),
            loss_config=dict(
                semantic=dict(ignore_index=-1),
                change=dict(ignore_index=-1)
            )
        ))

    def log_info(self):
        return dict(
            cfg=self.config
        )
