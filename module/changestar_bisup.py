import ever as er
import torch
from core import loss
from core.mixin import get_detector
from module.segmentation import Segmentation


@er.registry.MODEL.register()
class ChangeStarBiSup(er.ERModule):
    def __init__(self, config):
        super().__init__(config)

        self.segmentation = Segmentation(self.config.segmenation)

        self.detector = get_detector(**self.config.detector)

        self.init_from_weight_file()

    def forward(self, x, y=None):
        if x.size(1) == 6:
            # segmentation + change detection
            x1 = x[:, :3, :, :]
            x2 = x[:, 3:, :, :]
            t1_feature = self.segmentation(x1)
            t2_feature = self.segmentation(x2)

            change_logit = self.detector(torch.cat([t1_feature, t2_feature], dim=1))

            if self.training:
                loss_dict = dict()

                loss_dict.update(loss.misc_info(change_logit.device))

                loss_dict.update(
                    loss.bce_dice_loss((y['change'] > 0).float(), change_logit, self.config.loss_config))

                return loss_dict

            return change_logit

        raise ValueError('')

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
                ignore_index=-1
            )
        ))

    def log_info(self):
        return dict(
            cfg=self.config
        )
