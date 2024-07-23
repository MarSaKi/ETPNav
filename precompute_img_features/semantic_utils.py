import torch
import numpy as np


"""
12 cat labels are ranging from 1 to 12.
/!\ Need to add +1 in the list below  /!\
(The keys of the below dict are already ranging from 1 to 40)
"""
mpcat40_to_12cat = {"31":"0",
                    "13":"1",
                    "11":"2",
                    "8":"3",
                    "27":"4",
                    "10":"5",
                    "5":"6",
                    "3":"7",
                    "7":"8",
                    "14":"9",
                    "26":"10",
                    "15":"11"
                   }
mpcat40_to_12cat = {int(k): int(v) for k,v in mpcat40_to_12cat.items()}


label_colours = [(0,0,0),
                 (106, 137, 204),   # shelving
                 (230, 126, 34),    # chest of drawers
                 (7, 153, 146),   # bed
                 (248, 194, 145),   # cushion
                 (76, 209, 55),     # fireplace
                 (255, 168, 1),   # sofa
                 (184, 233, 148),   # table
                 (39, 174, 96),    # chair
                 (229, 80, 57),  # cabinet
                 (30, 55, 153),  # plant
                 (24, 220, 255), #(56, 173, 169),    # counter
                 (234, 32, 39),  #sink
]


def color_label(label):
    is_tensor = False
    if torch.is_tensor(label):
        is_tensor = True
        label = label.clone().cpu().data.numpy()

    colored_label = np.vectorize(lambda x: label_colours[int(x)])

    colored = np.asarray(colored_label(label)).astype(np.float32)
    colored = colored.squeeze()

    if not is_tensor:
        return colored
    try:
        return torch.from_numpy(colored.transpose([1, 0, 2, 3]))
    except ValueError:
        return torch.from_numpy(colored[np.newaxis, ...])



def convert_mpcat40_to_12cat(im):
    """
    converts a 2D image of semantic labels from the mpcat40 to the 12cat list of
    objects.
    Accepts either 2D array or 2D torch tensor.

    Input:
        2D array/Tensor: im

    Outputs:
        2D array/Tensor
    """

    assert len(im.shape) == 2


    if isinstance(im, torch.Tensor):
        im = im.to(dtype=torch.int)
        new_im = torch.zeros(im.shape, dtype=torch.int)
        unique = torch.unique(im)
        unique = unique.data.cpu().numpy()
        for u in unique:
            u = u.item()
            if u in mpcat40_to_12cat:
                new_im[im == u] = mpcat40_to_12cat[u] + 1
        return new_im
    elif isinstance(im, np.ndarray):
        im = im.astype(np.int)
        new_im = np.zeros(im.shape, dtype=np.int)
        unique = np.unique(im)
        unique = unique.astype(np.int)
        for u in unique:
            if u in mpcat40_to_12cat:
                new_im[im == u] = mpcat40_to_12cat[u] + 1
        return new_im
    else:
        print('format not supported: ', type(im))
        return None



  


"""
MP3D
"""
use_fine = ['appliances', 'furniture']

object_whitelist = ['shelving', 'chest_of_drawers', 'bed', 'cushion', 'fireplace',
                    'sofa', 'table', 'chair', 'cabinet', 'plant', 'counter', 'sink']





"""
REPLICA
"""
replica_to_mp3d_12cat_mapping = {'chair': 'chair',
                                 'cushion':'cushion',
                                 'table':'table',
                                 'indoor-plant': 'plant',
                                 'pillow':'cushion',
                                 'plant-stand': 'plant',
                                 'cabinet':'cabinet',
                                 'shelf':'shelving',
                                 'rack':'chest_of_drawers',
                                 'sofa': 'sofa',
                                 'countertop':'counter',
                                 'sink': 'sink',
                                 'base-cabinet':'cabinet',
                                 'wall-cabinet':'cabinet',
                                 'bed':'bed',
                                 'comforter':'bed',
                                 'desk': 'table',
                                }
