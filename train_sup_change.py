import ever as er
import numpy as np
import torch
from tqdm import tqdm

er.registry.register_all()


def register_evaluate_fn(launcher):
    launcher.override_evaluate(evaluate_levircd)


def evaluate_levircd(self, test_dataloader, config=None):
    self.model.eval()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    metric_op = er.metric.PixelMetric(2,
                                      self.model_dir,
                                      logger=self.logger)

    with torch.no_grad():
        for img, ret_gt in tqdm(test_dataloader):
            img = img.to(device)

            change = self.model.module(img).sigmoid() > 0.5

            pr_change = change.cpu().numpy().astype(np.uint8)
            gt_change = ret_gt['change']
            gt_change = gt_change.numpy()
            y_true = gt_change.ravel()
            y_pred = pr_change.ravel()

            y_true = np.where(y_true > 0, np.ones_like(y_true), np.zeros_like(y_true))

            metric_op.forward(y_true, y_pred)

    metric_op.summary_all()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    SEED = 2333
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    trainer = er.trainer.get_trainer('th_amp_ddp')()
    blob = trainer.run(after_construct_launcher_callbacks=[register_evaluate_fn])
