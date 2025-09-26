import torch
from utils.ssd_utils.ssd_utils import maybe_cuda, random_retrieve, SummarizeUpdate
import copy
from collections import defaultdict
import numpy as np

class DynamicBuffer(torch.nn.Module):
    def __init__(self, model, params):
        super().__init__()
        self.params = params
        self.model = model
        self.current_index = 0
        self.n_seen_so_far = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.images_per_class = self.params.images_per_class
        self.num_classes = 0

        # define buffer
        buffer_size = params.memory_size
        print('buffer has %d slots' % buffer_size)
        input_size = (20, 256)
        buffer_img = maybe_cuda(torch.FloatTensor(buffer_size, *input_size).fill_(0))
        buffer_label = maybe_cuda(torch.LongTensor(buffer_size).fill_(0))

        # registering as buffer allows us to save the object using `torch.save`
        self.register_buffer('buffer_img', buffer_img)
        self.register_buffer('buffer_img_rep', copy.deepcopy(buffer_img))
        self.register_buffer('buffer_label', buffer_label)
        self.condense_dict = defaultdict(list)
        self.labelsize = params.images_per_class
        self.avail_indices = list(np.arange(buffer_size))
        self.sum_up = SummarizeUpdate(self, self.params)

    def update(self, x, y,**kwargs):
        return self.sum_up.update(buffer=self, x=x, y=y, **kwargs)

    def retrieve(self, **kwargs):
        return random_retrieve(buffer=self, **kwargs)