import tokenizers as tk
from typing import Tuple, Sequence, Optional
from PIL import Image
from functools import partial
import torch
from torch import nn, Tensor
from torchvision import transforms
from src.model import ImgLinearBackbone, Encoder, Decoder, EncoderDecoder
from src.utils import subsequent_mask, pred_token_within_range, greedy_sampling

def load_vocab_and_model(vocab_path, patch_size, d_model, nhead, dropout, max_seq_len):
    vocab = tk.Tokenizer.from_file(vocab_path)
    
    Backbone = ImgLinearBackbone(d_model=d_model, patch_size=patch_size)
    encoder = Encoder(d_model=d_model, nhead=nhead, dropout = dropout, activation="gelu", norm_first=True, nlayer=12,ff_ratio=4)
    decoder = Decoder(d_model=d_model, nhead=nhead, dropout = dropout, activation="gelu", norm_first=True, nlayer=4, ff_ratio=4)
    
    model = EncoderDecoder(
        backbone=Backbone,
        encoder=encoder,
        decoder=decoder,
        vocab_size=vocab.get_vocab_size(),
        d_model=d_model,
        padding_idx=vocab.token_to_id("<pad>"),
        max_seq_len=max_seq_len,
        dropout=dropout,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
    return vocab, model

def autoregressive_decode(
    model: EncoderDecoder,
    image: Tensor,
    prefix: Sequence[int],
    max_decode_len: int,
    eos_id: int,
    device,
    token_whitelist: Optional[Sequence[int]] = None,
    token_blacklist: Optional[Sequence[int]] = None,
) -> Tensor:
    model.eval()
    with torch.no_grad():
        memory = model.encode(image)
        context = torch.tensor(prefix, dtype=torch.int32).repeat(image.shape[0], 1).to(device)

    for _ in range(max_decode_len):
        eos_flag = [eos_id in k for k in context]
        if all(eos_flag):
            break

        with torch.no_grad():
            causal_mask = subsequent_mask(context.shape[1]).to(device)
            logits = model.decode(
                memory, context, tgt_mask=causal_mask, tgt_padding_mask=None
            )
            logits = model.generator(logits)[:, -1, :]

        logits = pred_token_within_range(
            logits.detach(),
            white_list=token_whitelist,
            black_list=token_blacklist,
        )

        next_probs, next_tokens = greedy_sampling(logits)
        context = torch.cat([context, next_tokens], dim=1)
    return context

def image_to_tensor(image: Image, size: Tuple[int, int]) -> Tensor:
    T = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.86597056,0.88463002,0.87491087], std = [0.20686628,0.18201602,0.18485524])
    ])
    image_tensor = T(image)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor

def rescale_bbox(
    bbox: Sequence[Sequence[float]],
    src: Tuple[int, int],
    tgt: Tuple[int, int]
) -> Sequence[Sequence[float]]:
    assert len(src) == len(tgt) == 2
    ratio = [tgt[0] / src[0], tgt[1] / src[1]] * 2
    bbox = [[int(round(i * j)) for i, j in zip(entry, ratio)] for entry in bbox]
    return bbox

# Extract table header section(Column)
def structure_extract_thead_contents(data):
    return data[data.index('<thead>'):data.index('</thead>')+1]

# Count non-empty cells in each column based on structure
def structure_count_brackets(data):
    in_tr = False
    count = 0
    tr = []
    for item in data:
        if '<tr>' in item:
            in_tr = True
        elif '</tr>' in item:
            in_tr = False
            tr.append(count)
            count = 0
        
        if in_tr and '[]' in item:
            count += item.count('[]')
    return tr

# Count and identify when bboxes exceed the expected column count per row
def wrong_columns(pred_bboxes, len_colls):
    all_bbox_cols=[]
    count = 0
    for later_bbox, front_bbox in zip(pred_bboxes, [[0,0,0,0]]+pred_bboxes):
        if front_bbox[2] > later_bbox[0]:
            all_bbox_cols.append(count)
            count = 0
        if len(all_bbox_cols) == len(len_colls):
            break
        else:
            count += 1
    return all_bbox_cols

# Function to adjust bboxes when detected count doesn't match expected column count#
def correction_bbox(pred_bbox, structure_colls, all_bbox_cols):
    start = 1
    remo_idx = []
    none_first = False
    for structure_c, bbox_c in zip(structure_colls, all_bbox_cols):
        # When there are excess bboxes, examine adjacent ones
        if structure_c < bbox_c:
            for j in range(start, bbox_c+1):
                front = pred_bbox[j-1]
                back = pred_bbox[j]
                
                # Merge adjacent bboxes if they are within threshold distance
                if (back[0]-front[2] < 16) and (back[0]-front[2] > -5):
                    remo_idx.append(j-1)
                    back[0]=front[0]
                    
                    # Extend y-coordinates to maximum range of both bboxes
                    if front[1] > back[1]:
                        front[1] = back[1]
                    if front[3] < back[3]:
                        front[3] = back[3]

        # Fewer bboxes than expected - typically due to empty first row/column
        elif structure_c > bbox_c:
            none_first = True
                
        start += bbox_c

    for i in remo_idx[::-1]:
        pred_bbox.pop(i)
        
    return pred_bbox, none_first