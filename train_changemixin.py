import random

import ever as er
import numpy as np
import torch
from tqdm import tqdm

er.registry.register_all()


def register_leviscd_evaluate_fn(launcher):
    launcher.override_evaluate(evaluate_levircd)


def evaluate_levircd(self, test_dataloader, config=None):
    self.model.eval()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    det_metric_op = er.metric.PixelMetric(2,
                                          self.model_dir,
                                          logger=self.logger)

    with torch.no_grad():
        for img, ret_gt in tqdm(test_dataloader):
            img = img.to(device)

            y1y2change = self.model.module(img).sigmoid() > 0.5

            pr_change = y1y2change[:, 2, :, :].cpu()
            pr_change = pr_change.numpy().astype(np.uint8)
            gt_change = ret_gt['change']
            gt_change = gt_change.numpy()
            y_true = gt_change.ravel()
            y_pred = pr_change.ravel()

            y_true = np.where(y_true > 0, np.ones_like(y_true), np.zeros_like(y_true))

            det_metric_op.forward(y_true, y_pred)

    split = [s.replace('./LEVIR-CD/', '') for s in test_dataloader.config.root_dir]
    split_str = ','.join(split)
    self.logger.info(f'det -[LEVIRCD {split_str}]')
    det_metric_op.summary_all()

    torch.cuda.empty_cache()


if __name__ == '__main__':
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    SEED = 2333
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.set_rng_state(torch.manual_seed(SEED).get_state())

    trainer = er.trainer.get_trainer('th_amp_ddp')()
    trainer.run(after_construct_launcher_callbacks=[register_leviscd_evaluate_fn])
