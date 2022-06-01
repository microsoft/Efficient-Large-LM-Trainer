# Copyright (c) Microsoft. All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from fairseq.data import BaseWrapperDataset


class TableLookupDataset(BaseWrapperDataset):
    def __init__(self, dataset, table):
        super().__init__(dataset)
        self.table = [
            torch.tensor(item, dtype=torch.long)
            for item in table
        ]
        self._sizes = np.array([
            len(table[item[0].item()])
            for item in dataset
        ])

    def __getitem__(self, idx):
        item = self.dataset[idx]
        assert len(item) == 1
        item = item[0].item()
        return self.table[item]

    @property
    def sizes(self):
        return self._sizes

    def num_tokens(self, index):
        return self._sizes[index]

    def size(self, index):
        return self._sizes[index]