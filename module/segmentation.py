import ever as er
import torch.nn as nn
from module.model_list import Models


@er.registry.MODEL.register()
class Segmentation(er.ERModule):
    def __init__(self, config):
        super(Segmentation, self).__init__(config)

        odict = Models[self.config.model_type]
        for k, v in odict.items():
            if isinstance(v, nn.Module):
                odict[k] = v
            elif issubclass(v, er.ERModule):
                odict[k] = v(self.config[k])
            elif issubclass(v, nn.Module):
                odict[k] = v(**self.config[k])

        self.features = nn.Sequential(odict)

    def forward(self, x, y=None):
        logit = self.features(x)
        return logit

    def set_default_config(self):
        self.config.update(dict())
