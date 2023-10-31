import random
import math
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from .common import pad_tensors, gen_seq_masks

############### Masked Language Modeling ###############
def random_word(tokens, vocab_range, mask):
    """
    Masking some random tokens for Language Model task with probabilities as in
        the original BERT paper.
    :param tokens: list of int, tokenized sentence.
    :param vocab_range: for choosing a random word
    :return: (list of int, list of int), masked tokens and related labels for
        LM prediction
    """
    output_tokens, output_label = [], []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                output_tokens.append(mask)

            # 10% randomly change token to random token
            elif prob < 0.9:
                output_tokens.append(random.choice(list(range(*vocab_range))))

            # -> rest 10% randomly keep current token
            else:
                output_tokens.append(token)

            # append current token to output (we will predict these later)
            output_label.append(token)
        else:
            output_tokens.append(token)
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)
    
    if all(o == -1 for o in output_label):
        # at least mask 1
        output_label[0] = tokens[0]
        output_tokens[0] = mask

    return output_tokens, output_label    

class MlmDataset(Dataset):
    def __init__(self, nav_db, tok):
        self.nav_db = nav_db
        self.tok = tok

        self.vocab_range = [1996, 29611] #TODO: manually checked in bert-base-uncased
        self.cls_token_id = self.tok.cls_token_id   # 101
        self.sep_token_id = self.tok.sep_token_id   # 102
        self.mask_token_id = self.tok.mask_token_id # 103
        self.pad_token_id = self.tok.pad_token_id   # 0

    def __len__(self):
        return len(self.nav_db)

    def __getitem__(self, idx):
        inputs = self.nav_db.get_input(idx, 'pos')

        output = {}

        txt_ids, txt_labels = random_word(inputs['instr_encoding'], 
            self.vocab_range, self.mask_token_id)
        output['txt_ids'] = torch.LongTensor(txt_ids)
        output['txt_labels'] = torch.LongTensor(txt_labels)

        output['traj_view_img_fts'] = [torch.from_numpy(x) for x in inputs['traj_view_img_fts']]
        output['traj_view_dep_fts'] = [torch.from_numpy(x) for x in inputs['traj_view_dep_fts']]
        if 'traj_obj_img_fts' in inputs:
            output['traj_obj_img_fts'] = [torch.from_numpy(x) for x in inputs['traj_obj_img_fts']]
        output['traj_loc_fts'] = [torch.from_numpy(x) for x in inputs['traj_loc_fts']]
        output['traj_nav_types'] = [torch.LongTensor(x) for x in inputs['traj_nav_types']]
        output['traj_cand_vpids'] = inputs['traj_cand_vpids']
        output['traj_vpids'] = inputs['traj_vpids']

        output['gmap_vpids'] = inputs['gmap_vpids']
        output['gmap_step_ids'] = torch.LongTensor(inputs['gmap_step_ids'])
        output['gmap_visited_masks'] = torch.BoolTensor(inputs['gmap_visited_masks'])
        output['gmap_pos_fts'] = torch.from_numpy(inputs['gmap_pos_fts'])
        output['gmap_pair_dists'] = torch.from_numpy(inputs['gmap_pair_dists'])

        # output['vp_pos_fts'] = torch.from_numpy(inputs['vp_pos_fts'])
        # output['vp_angles'] = inputs['vp_angles']
        return output

def mlm_collate(inputs):
    batch = {
        k: [x[k] for x in inputs] for k in inputs[0].keys()
    }
    # text batches
    batch['txt_lens'] = torch.LongTensor([len(x) for x in batch['txt_ids']])
    batch['txt_ids'] = pad_sequence(batch['txt_ids'], batch_first=True, padding_value=0)
    batch['txt_labels'] = pad_sequence(batch['txt_labels'], batch_first=True, padding_value=-1)

    # trajectory batches: traj_cand_vpids, traj_vpids
    batch['traj_step_lens'] = [len(x) for x in batch['traj_view_img_fts']]
    batch['traj_vp_view_lens'] = torch.LongTensor(
        sum([[len(y) for y in x] for x in batch['traj_view_img_fts']], [])
    )
    batch['traj_view_img_fts'] = pad_tensors(sum(batch['traj_view_img_fts'], []))
    batch['traj_view_dep_fts'] = pad_tensors(sum(batch['traj_view_dep_fts'], []))
    if 'traj_obj_img_fts' in batch:
        batch['traj_vp_obj_lens'] = torch.LongTensor(
            sum([[len(y) for y in x] for x in batch['traj_obj_img_fts']], [])
        )
        batch['traj_obj_img_fts'] = pad_tensors(sum(batch['traj_obj_img_fts'], []))
    batch['traj_loc_fts'] = pad_tensors(sum(batch['traj_loc_fts'], []))
    batch['traj_nav_types'] = pad_sequence(sum(batch['traj_nav_types'], []), batch_first=True, padding_value=0)

    # gmap batches: gmap_vpids
    batch['gmap_lens'] = torch.LongTensor([len(x) for x in batch['gmap_step_ids']]) # included [stop]
    batch['gmap_step_ids'] = pad_sequence(batch['gmap_step_ids'], batch_first=True, padding_value=0)
    batch['gmap_visited_masks'] = pad_sequence(batch['gmap_visited_masks'], batch_first=True, padding_value=0)
    batch['gmap_pos_fts'] = pad_tensors(batch['gmap_pos_fts'])
    max_gmap_len = max(batch['gmap_lens'])
    batch_size = len(batch['gmap_lens'])
    gmap_pair_dists = torch.zeros(batch_size, max_gmap_len, max_gmap_len).float()
    for i in range(batch_size):
        gmap_pair_dists[i, :batch['gmap_lens'][i], :batch['gmap_lens'][i]] = batch['gmap_pair_dists'][i]
    batch['gmap_pair_dists'] = gmap_pair_dists

    # vp batches: vp_angles
    # batch['vp_lens'] = torch.LongTensor([len(x[-1]) for x in batch['vp_pos_fts']])  # included [stop]
    # batch['vp_pos_fts'] = pad_tensors(batch['vp_pos_fts'])

    return batch


############### Masked Region Modeling ###############
def _get_img_mask(mask_prob, num_images):
    img_mask = [np.random.rand() < mask_prob for _ in range(num_images)]
    if not any(img_mask):
        # at least mask 1
        img_mask[np.random.randint(num_images)] = True
    img_mask = torch.tensor(img_mask)
    return img_mask

def _mask_img_feat(img_feat, img_masks):
    img_masks_ext = img_masks.unsqueeze(-1).expand_as(img_feat)
    img_feat_masked = img_feat.data.masked_fill(img_masks_ext, 0)
    return img_feat_masked

def _get_targets(img_soft_label, img_masks):
    soft_label_dim = img_soft_label.size(-1)
    img_masks_ext_for_label = img_masks.unsqueeze(-1).expand_as(img_soft_label)
    label_targets = img_soft_label[img_masks_ext_for_label].contiguous().view(-1, soft_label_dim)
    return label_targets

class MrcDataset(Dataset):
    def __init__(self, nav_db, tok, mask_prob, end_vp_pos_ratio=1):
        self.nav_db = nav_db
        self.tok = tok
        self.mask_prob = mask_prob

        self.cls_token_id = self.tok.cls_token_id   # 101
        self.sep_token_id = self.tok.sep_token_id   # 102
        self.pad_token_id = self.tok.pad_token_id   # 0

        self.end_vp_pos_ratio = end_vp_pos_ratio
        

    def __len__(self):
        return len(self.nav_db.data)

    def __getitem__(self, idx):
        r = np.random.rand()
        if r < self.end_vp_pos_ratio:
            end_vp_type = 'pos'
        else:
            end_vp_type = 'neg_in_gt_path'
        inputs = self.nav_db.get_input(idx, end_vp_type, return_img_probs=True)

        output = {}

        output['txt_ids'] = torch.LongTensor(inputs['instr_encoding'])

        output['traj_view_img_fts'] = [torch.from_numpy(x) for x in inputs['traj_view_img_fts']]
        
        # mask image
        view_mrc_masks = _get_img_mask(self.mask_prob, len(output['traj_view_img_fts'][-1]))
        output['traj_view_img_fts'][-1] = _mask_img_feat(output['traj_view_img_fts'][-1], view_mrc_masks)
        output['vp_view_probs'] = torch.from_numpy(inputs['vp_view_probs']) # no [stop]
        output['vp_view_mrc_masks'] = view_mrc_masks
        output['traj_loc_fts'] = [torch.from_numpy(x) for x in inputs['traj_loc_fts']]
        output['traj_nav_types'] = [torch.LongTensor(x) for x in inputs['traj_nav_types']]
        output['traj_cand_vpids'] = inputs['traj_cand_vpids']
        output['traj_vpids'] = inputs['traj_vpids']

        output['gmap_vpids'] = inputs['gmap_vpids']
        output['gmap_step_ids'] = torch.LongTensor(inputs['gmap_step_ids'])
        output['gmap_visited_masks'] = torch.BoolTensor(inputs['gmap_visited_masks'])
        output['gmap_pos_fts'] = torch.from_numpy(inputs['gmap_pos_fts'])
        output['gmap_pair_dists'] = torch.from_numpy(inputs['gmap_pair_dists'])

        output['vp_pos_fts'] = torch.from_numpy(inputs['vp_pos_fts'])
        output['vp_angles'] = inputs['vp_angles']

        if 'traj_obj_img_fts' in inputs:
            output['traj_obj_img_fts'] = [torch.from_numpy(x) for x in inputs['traj_obj_img_fts']]
            if len(output['traj_obj_img_fts'][-1]) > 0:
                obj_mrc_masks = _get_img_mask(self.mask_prob, len(output['traj_obj_img_fts'][-1]))
                output['traj_obj_img_fts'][-1] = _mask_img_feat(output['traj_obj_img_fts'][-1], obj_mrc_masks)
            else:
                obj_mrc_masks = torch.zeros(0, ).bool()
            output['vp_obj_probs'] = torch.from_numpy(inputs['vp_obj_probs'])
            output['vp_obj_mrc_masks'] = obj_mrc_masks

        return output

def mrc_collate(inputs):
    batch = {
        k: [x[k] for x in inputs] for k in inputs[0].keys()
    }
    # text batches
    batch['txt_lens'] = torch.LongTensor([len(x) for x in batch['txt_ids']])
    batch['txt_ids'] = pad_sequence(batch['txt_ids'], batch_first=True, padding_value=0)

    # trajectory batches: traj_cand_vpids, traj_vpids
    batch['traj_step_lens'] = [len(x) for x in batch['traj_view_img_fts']]
    batch['traj_vp_view_lens'] = torch.LongTensor(
        sum([[len(y) for y in x] for x in batch['traj_view_img_fts']], [])
    )
    batch['traj_view_img_fts'] = pad_tensors(sum(batch['traj_view_img_fts'], []))
    batch['traj_loc_fts'] = pad_tensors(sum(batch['traj_loc_fts'], []))
    batch['traj_nav_types'] = pad_sequence(sum(batch['traj_nav_types'], []), batch_first=True, padding_value=0)

    # gmap batches: gmap_vpids
    batch['gmap_lens'] = torch.LongTensor([len(x) for x in batch['gmap_step_ids']]) # included [stop]
    batch['gmap_step_ids'] = pad_sequence(batch['gmap_step_ids'], batch_first=True, padding_value=0)
    batch['gmap_visited_masks'] = pad_sequence(batch['gmap_visited_masks'], batch_first=True, padding_value=0)
    batch['gmap_pos_fts'] = pad_tensors(batch['gmap_pos_fts'])
    max_gmap_len = max(batch['gmap_lens'])
    batch_size = len(batch['gmap_lens'])
    gmap_pair_dists = torch.zeros(batch_size, max_gmap_len, max_gmap_len).float()
    for i in range(batch_size):
        gmap_pair_dists[i, :batch['gmap_lens'][i], :batch['gmap_lens'][i]] = batch['gmap_pair_dists'][i]
    batch['gmap_pair_dists'] = gmap_pair_dists

    # vp batches: vp_angles
    batch['vp_lens'] = torch.LongTensor([len(x[-1]) for x in batch['vp_pos_fts']])  # included [stop]
    batch['vp_pos_fts'] = pad_tensors(batch['vp_pos_fts'])

    # vp labels
    batch['vp_view_mrc_masks'] = pad_sequence(batch['vp_view_mrc_masks'], batch_first=True, padding_value=0)
    batch['vp_view_probs'] = pad_tensors(batch['vp_view_probs'])

    if 'traj_obj_img_fts' in batch:
        batch['traj_vp_obj_lens'] = torch.LongTensor(
            sum([[len(y) for y in x] for x in batch['traj_obj_img_fts']], [])
        )
        batch['traj_obj_img_fts'] = pad_tensors(sum(batch['traj_obj_img_fts'], []))
        batch['vp_obj_mrc_masks'] = pad_sequence(batch['vp_obj_mrc_masks'], batch_first=True, padding_value=0)
        batch['vp_obj_probs'] = pad_tensors(batch['vp_obj_probs'])

    return batch


############### Single-step Action Prediction ###############
class SapDataset(Dataset):
    def __init__(self, nav_db, tok, end_vp_pos_ratio=0.2):
        '''Instruction Trajectory Matching'''
        self.nav_db = nav_db
        self.tok = tok

        self.cls_token_id = self.tok.cls_token_id   # 101
        self.sep_token_id = self.tok.sep_token_id   # 102
        self.pad_token_id = self.tok.pad_token_id   # 0

        self.end_vp_pos_ratio = end_vp_pos_ratio

    def __len__(self):
        return len(self.nav_db.data)

    def __getitem__(self, idx):
        r = np.random.rand()
        if r < self.end_vp_pos_ratio:
            end_vp_type = 'pos'
        elif r < 0.6:
            end_vp_type = 'neg_in_gt_path'
        else:
            end_vp_type = 'neg_others'
        inputs = self.nav_db.get_input(idx, end_vp_type, return_act_label=True)

        output = {}

        output['txt_ids'] = torch.LongTensor(inputs['instr_encoding'])

        output['traj_view_img_fts'] = [torch.from_numpy(x) for x in inputs['traj_view_img_fts']]
        output['traj_view_dep_fts'] = [torch.from_numpy(x) for x in inputs['traj_view_dep_fts']]
        if 'traj_obj_img_fts' in inputs:
            output['traj_obj_img_fts'] = [torch.from_numpy(x) for x in inputs['traj_obj_img_fts']]
        output['traj_loc_fts'] = [torch.from_numpy(x) for x in inputs['traj_loc_fts']]
        output['traj_nav_types'] = [torch.LongTensor(x) for x in inputs['traj_nav_types']]
        output['traj_cand_vpids'] = inputs['traj_cand_vpids']
        output['traj_vpids'] = inputs['traj_vpids']

        output['gmap_vpids'] = inputs['gmap_vpids']
        output['gmap_step_ids'] = torch.LongTensor(inputs['gmap_step_ids'])
        output['gmap_visited_masks'] = torch.BoolTensor(inputs['gmap_visited_masks'])
        output['gmap_pos_fts'] = torch.from_numpy(inputs['gmap_pos_fts'])
        output['gmap_pair_dists'] = torch.from_numpy(inputs['gmap_pair_dists'])

        # output['vp_pos_fts'] = torch.from_numpy(inputs['vp_pos_fts'])
        # output['vp_angles'] = inputs['vp_angles']

        output['local_act_labels'] = inputs['local_act_labels']
        output['global_act_labels'] = inputs['global_act_labels']
        return output

def sap_collate(inputs):
    batch = {
        k: [x[k] for x in inputs] for k in inputs[0].keys()
    }
    # text batches
    batch['txt_lens'] = torch.LongTensor([len(x) for x in batch['txt_ids']])
    batch['txt_ids'] = pad_sequence(batch['txt_ids'], batch_first=True, padding_value=0)

    # trajectory batches: traj_cand_vpids, traj_vpids
    batch['traj_step_lens'] = [len(x) for x in batch['traj_view_img_fts']]
    batch['traj_vp_view_lens'] = torch.LongTensor(
        sum([[len(y) for y in x] for x in batch['traj_view_img_fts']], [])
    )
    batch['traj_view_img_fts'] = pad_tensors(sum(batch['traj_view_img_fts'], []))
    batch['traj_view_dep_fts'] = pad_tensors(sum(batch['traj_view_dep_fts'], []))
    if 'traj_obj_img_fts' in batch:
        batch['traj_vp_obj_lens'] = torch.LongTensor(
            sum([[len(y) for y in x] for x in batch['traj_obj_img_fts']], [])
        )
        batch['traj_obj_img_fts'] = pad_tensors(sum(batch['traj_obj_img_fts'], []))
    batch['traj_loc_fts'] = pad_tensors(sum(batch['traj_loc_fts'], []))
    batch['traj_nav_types'] = pad_sequence(sum(batch['traj_nav_types'], []), batch_first=True, padding_value=0)

    # gmap batches: gmap_vpids
    batch['gmap_lens'] = torch.LongTensor([len(x) for x in batch['gmap_step_ids']]) # included [stop]
    batch['gmap_step_ids'] = pad_sequence(batch['gmap_step_ids'], batch_first=True, padding_value=0)
    batch['gmap_visited_masks'] = pad_sequence(batch['gmap_visited_masks'], batch_first=True, padding_value=0)
    batch['gmap_pos_fts'] = pad_tensors(batch['gmap_pos_fts'])
    max_gmap_len = max(batch['gmap_lens'])
    batch_size = len(batch['gmap_lens'])
    gmap_pair_dists = torch.zeros(batch_size, max_gmap_len, max_gmap_len).float()
    for i in range(batch_size):
        gmap_pair_dists[i, :batch['gmap_lens'][i], :batch['gmap_lens'][i]] = batch['gmap_pair_dists'][i]
    batch['gmap_pair_dists'] = gmap_pair_dists

    # vp batches: vp_angles
    # batch['vp_lens'] = torch.LongTensor([len(x[-1]) for x in batch['vp_pos_fts']])  # included [stop]
    # batch['vp_pos_fts'] = pad_tensors(batch['vp_pos_fts'])

    # action labels
    batch['local_act_labels'] = torch.LongTensor(batch['local_act_labels'])
    batch['global_act_labels'] = torch.LongTensor(batch['global_act_labels'])
    return batch


############### Object Grounding ###############
class OGDataset(Dataset):
    def __init__(self, nav_db, tok):
        self.nav_db = nav_db
        self.tok = tok

    def __len__(self):
        return len(self.nav_db.data)

    def __getitem__(self, idx):
        inputs = self.nav_db.get_input(idx, 'pos', return_obj_label=True)

        output = {}

        output['txt_ids'] = torch.LongTensor(inputs['instr_encoding'])

        output['traj_view_img_fts'] = [torch.from_numpy(x) for x in inputs['traj_view_img_fts']]
        output['traj_obj_img_fts'] = [torch.from_numpy(x) for x in inputs['traj_obj_img_fts']]
        output['traj_loc_fts'] = [torch.from_numpy(x) for x in inputs['traj_loc_fts']]
        output['traj_nav_types'] = [torch.LongTensor(x) for x in inputs['traj_nav_types']]
        output['traj_cand_vpids'] = inputs['traj_cand_vpids']
        output['traj_vpids'] = inputs['traj_vpids']

        output['gmap_vpids'] = inputs['gmap_vpids']
        output['gmap_step_ids'] = torch.LongTensor(inputs['gmap_step_ids'])
        output['gmap_visited_masks'] = torch.BoolTensor(inputs['gmap_visited_masks'])
        output['gmap_pos_fts'] = torch.from_numpy(inputs['gmap_pos_fts'])
        output['gmap_pair_dists'] = torch.from_numpy(inputs['gmap_pair_dists'])

        output['vp_pos_fts'] = torch.from_numpy(inputs['vp_pos_fts'])
        output['vp_angles'] = inputs['vp_angles']

        output['obj_labels'] = inputs['obj_labels']
        return output

def og_collate(inputs):
    batch = {
        k: [x[k] for x in inputs] for k in inputs[0].keys()
    }
    # text batches
    batch['txt_lens'] = torch.LongTensor([len(x) for x in batch['txt_ids']])
    batch['txt_ids'] = pad_sequence(batch['txt_ids'], batch_first=True, padding_value=0)

    # trajectory batches: traj_cand_vpids, traj_vpids
    batch['traj_step_lens'] = [len(x) for x in batch['traj_view_img_fts']]
    batch['traj_vp_view_lens'] = torch.LongTensor(
        sum([[len(y) for y in x] for x in batch['traj_view_img_fts']], [])
    )
    batch['traj_vp_obj_lens'] = torch.LongTensor(
        sum([[len(y) for y in x] for x in batch['traj_obj_img_fts']], [])
    )
    batch['traj_view_img_fts'] = pad_tensors(sum(batch['traj_view_img_fts'], []))
    batch['traj_obj_img_fts'] = pad_tensors(sum(batch['traj_obj_img_fts'], []))
    batch['traj_loc_fts'] = pad_tensors(sum(batch['traj_loc_fts'], []))
    batch['traj_nav_types'] = pad_sequence(sum(batch['traj_nav_types'], []), batch_first=True, padding_value=0)

    # gmap batches: gmap_vpids
    batch['gmap_lens'] = torch.LongTensor([len(x) for x in batch['gmap_step_ids']]) # included [stop]
    batch['gmap_step_ids'] = pad_sequence(batch['gmap_step_ids'], batch_first=True, padding_value=0)
    batch['gmap_visited_masks'] = pad_sequence(batch['gmap_visited_masks'], batch_first=True, padding_value=0)
    batch['gmap_pos_fts'] = pad_tensors(batch['gmap_pos_fts'])
    max_gmap_len = max(batch['gmap_lens'])
    batch_size = len(batch['gmap_lens'])
    gmap_pair_dists = torch.zeros(batch_size, max_gmap_len, max_gmap_len).float()
    for i in range(batch_size):
        gmap_pair_dists[i, :batch['gmap_lens'][i], :batch['gmap_lens'][i]] = batch['gmap_pair_dists'][i]
    batch['gmap_pair_dists'] = gmap_pair_dists

    # vp batches: vp_angles
    batch['vp_lens'] = torch.LongTensor([len(x[-1]) for x in batch['vp_pos_fts']])  # included [stop]
    batch['vp_pos_fts'] = pad_tensors(batch['vp_pos_fts'])

    # vp labels
    batch['obj_labels'] = torch.LongTensor(batch['obj_labels'])
    return batch
