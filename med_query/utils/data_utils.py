# Copyright (c) DAMO Health
import torch


class DataPrefetcher(object):
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            self.batch["img"] = self.batch["img"].cuda(non_blocking=True)
            self.batch["msk"] = self.batch["msk"].cuda(non_blocking=True)
            self.batch["bboxes"] = [i.cuda(non_blocking=True) for i in self.batch["bboxes"]]
            self.batch["labels"] = [i.cuda(non_blocking=True) for i in self.batch["labels"]]
            self.batch["angles"] = [i.cuda(non_blocking=True) for i in self.batch["angles"]]
            self.batch["rotats"] = [i.cuda(non_blocking=True) for i in self.batch["rotats"]]

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        if batch is not None:
            return batch
        else:
            raise StopIteration

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return self
