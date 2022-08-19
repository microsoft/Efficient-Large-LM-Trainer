from fairseq.data import BaseWrapperDataset, data_utils


class PadDataset(BaseWrapperDataset):
    def __init__(self, dataset, pad_idx, left_pad):
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.left_pad = left_pad

    def collater(self, samples):
        return data_utils.collate_tokens(samples, self.pad_idx, left_pad=self.left_pad, pad_to_multiple=8)


class RightPadDataset(PadDataset):
    def __init__(self, dataset, pad_idx):
        super().__init__(dataset, pad_idx, left_pad=False)
