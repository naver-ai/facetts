from .lrs3_dataset import LRS3Dataset, TextMelVideoBatchCollate
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class LRS3DataModule(LightningDataModule):
    def __init__(self, _config):
        super().__init__()

        self.config = _config
        self.per_gpu_batchsize = _config["per_gpu_batchsize"]
        self.num_workers = _config["num_workers"]
        self.prefetch_factor = _config["prefetch_factor"]

        self.setup_flag = False

    @property
    def dataset_cls(self):
        return LRS3Dataset

    @property
    def dataset_name(self):
        return "LRS3"

    def set_train_dataset(self):
        self.train_dataset = self.dataset_cls(split="train", config=self.config)

    def set_val_dataset(self):
        self.val_dataset = self.dataset_cls(split="val", config=self.config)

    def set_test_dataset(self):
        self.test_dataset = self.dataset_cls(split="test", config=self.config)

    def setup(self, stage):
        if not self.setup_flag:
            self.set_train_dataset()
            self.set_val_dataset()
            self.set_test_dataset()

            self.setup_flag = True

    def load_dataloader(self, dataset, shuffle=True, drop_last=False):
        batch_collate = TextMelVideoBatchCollate()

        loader = DataLoader(
            dataset,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            batch_size=self.per_gpu_batchsize,
            pin_memory=True,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=batch_collate,
        )
        return loader

    def train_dataloader(self):
        loader = self.load_dataloader(self.train_dataset, True, False)
        return loader

    def val_dataloader(self):
        loader = self.load_dataloader(self.val_dataset, False, False)
        return loader

    def test_dataloader(self):
        loader = self.load_dataloader(self.test_dataset, False, False)
        return loader
