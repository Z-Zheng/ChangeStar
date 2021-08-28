from collections import OrderedDict

import ever.module as M

Models = dict(
    deeplabv3=OrderedDict([
        ('backbone', M.ResNetEncoder),
        ('neck', M.ListIndex(-1)),
        ('head', M.AtrousSpatialPyramidPool),
    ]),
    deeplabv3p=OrderedDict([
        ('backbone', M.ResNetEncoder),
        ('neck', M.ListIndex(0, -1)),
        ('head', M.Deeplabv3pDecoder),
    ]),
    pspnet=OrderedDict([
        ('backbone', M.ResNetEncoder),
        ('neck', M.ListIndex(-1)),
        ('head', M.PyramidPoolModule),
    ]),
    semantic_fpn=OrderedDict([
        ('backbone', M.ResNetEncoder),
        ('neck', M.FPN),
        ('head', M.AssymetricDecoder)
    ]),
    farseg=OrderedDict([
        ('backbone', M.ResNetEncoder),
        ('neck', M.ListIndex(0, 1, 2, 3)),
        ('head', M.FarSegHead),
    ]),
)
