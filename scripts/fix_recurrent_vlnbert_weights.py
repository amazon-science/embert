from collections import OrderedDict

import torch

original_path = "storage/models/pretrained/snap/VLNBERT-OSCAR-final/state_dict/best_val_unseen"

destination_path = "storage/models/pretrained/recurrent_vlnbert_oscar.pt"

# first of all we load the checkpoint
checkpoint = torch.load(original_path, map_location="cpu")

# we extract the actual state_dict
orig_state_dict = checkpoint["vln_bert"]["state_dict"]

# the weights stored by Recurrent VLN-BERT have a prefix named vln_bert.bert.*, we just get rid of vln_bert

new_state_dict = OrderedDict({
    k.replace("vln_bert.", ""): v for k, v in orig_state_dict.items()
})

torch.save(new_state_dict, destination_path)
