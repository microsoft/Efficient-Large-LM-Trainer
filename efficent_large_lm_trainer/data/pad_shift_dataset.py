from fairseq.data import BaseWrapperDataset, data_utils


class PadShiftDataset(BaseWrapperDataset):
    def __init__(self, dataset, pad_idx, start_idx):
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.start_idx = start_idx

    def collater(self, samples):
        return data_utils.collate_tokens(
            samples,
            self.pad_idx,
            eos_idx=self.start_idx,
            left_pad=False,
            move_eos_to_beginning=True
        )