import os
from pathlib import Path
from PIL import Image
import torch
from src.utils import bbox_str_to_token_list, html_str_to_token_list, build_table_from_html_and_cell, html_table_template
from src.trainer.utils import VALID_HTML_TOKEN, VALID_BBOX_TOKEN, INVALID_CELL_TOKEN
from modules import image_to_tensor, rescale_bbox, load_vocab_and_model, autoregressive_decode, structure_extract_thead_contents, structure_count_brackets, wrong_columns, correction_bbox
from ocrs import Use_ocr, Cell_content

def unitable(args): 
    ###################### Main function description ######################
    # Main function for table image processing and HTML conversion  
    # Load input image and perform preprocessing operations
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    image = Image.open(Path(args.image_path)).convert("RGB")
    image_size = image.size
    image_tensor = image_to_tensor(image, size=(448, 448)).to(device)

    VOCAB_DIR = Path(args.vocab_dir)
    VOCAB_FILE_NAME = ["vocab_html.json", "vocab_bbox.json", "vocab_cell_6k.json"]
    
    MODEL_DIR = Path(args.model_dir)
    MODEL_FILE_NAME = ["unitable_large_structure.pt", "unitable_large_bbox.pt", "unitable_large_content.pt"]
    
    ###################### Table structure extraction ######################
    # Table structure extraction: Recognize table layout and structural information
    structure_vocab_path = str(VOCAB_DIR / VOCAB_FILE_NAME[0])
    structure_model_weights = MODEL_DIR / MODEL_FILE_NAME[0]

    vocab, model = load_vocab_and_model(structure_vocab_path,
                        patch_size = args.patch_size,
                        d_model = args.d_model,
                        nhead = args.nhead,
                        dropout = args.dropout,
                        max_seq_len = args.structure_max_seq_len)

    model.load_state_dict(torch.load(structure_model_weights, map_location="cpu"))
    model = model.to(device)

    # Inference
    pred_html = autoregressive_decode(
        model = model,
        image = image_tensor,
        prefix = [vocab.token_to_id("[html]")],
        max_decode_len = args.structure_max_decode_len,
        eos_id = vocab.token_to_id("<eos>"),
        device = device,
        token_whitelist=[vocab.token_to_id(i) for i in VALID_HTML_TOKEN],
        token_blacklist = None)

    # Convert token id to token text
    pred_html = pred_html.detach().cpu().numpy()[0]
    pred_html = vocab.decode(pred_html, skip_special_tokens=False)
    pred_html = html_str_to_token_list(pred_html)
    
    ###################### Table cell bbox detection ######################
    # Cell region detection: Detect position and bounding boxes for each cell
    bbox_vocab_path = str(VOCAB_DIR / VOCAB_FILE_NAME[1])
    bbox_model_weights = MODEL_DIR / MODEL_FILE_NAME[1]

    vocab, model = load_vocab_and_model(bbox_vocab_path,
                        patch_size = args.patch_size,
                        d_model = args.d_model,
                        nhead = args.nhead,
                        dropout = args.dropout,
                        max_seq_len = args.bbox_max_seq_len)

    model.load_state_dict(torch.load(bbox_model_weights, map_location="cpu"))
    model = model.to(device)

    # Inference
    pred_bbox = autoregressive_decode(
        model = model,
        image = image_tensor,
        prefix = [vocab.token_to_id("[bbox]")],
        max_decode_len = args.bbox_max_decode_len,
        eos_id = vocab.token_to_id("<eos>"),
        device = device,
        token_whitelist=[vocab.token_to_id(i) for i in VALID_BBOX_TOKEN[: 449]],
        token_blacklist = None)
    
    # Convert token id to token text
    pred_bbox = pred_bbox.detach().cpu().numpy()[0]
    pred_bbox = vocab.decode(pred_bbox, skip_special_tokens=False)
    pred_bbox = bbox_str_to_token_list(pred_bbox)
    
    # Additional adjustments for Korean tables
    none_first = False
    if args.korean:  
        # Analyze column structure and validate bbox matching
        thead_count = structure_extract_thead_contents(pred_html)
        structure_cols = structure_count_brackets(thead_count)
        all_bbox_cols = wrong_columns(pred_bbox, structure_cols)
        
        # Adjust if structure and bbox counts don't match
        if structure_cols != all_bbox_cols:  
            pred_bbox, none_first = correction_bbox(pred_bbox, structure_cols, all_bbox_cols)
            
    pred_bbox = rescale_bbox(pred_bbox, src=(448, 448), tgt=image_size)
            
    ###################### Table cell content recognition ######################
    # Cell content recognition: Extract text content from each cell
    content_vocab_path = str(VOCAB_DIR / VOCAB_FILE_NAME[2])
    content_model_weights = MODEL_DIR / MODEL_FILE_NAME[2]
    
    vocab, model = load_vocab_and_model(content_vocab_path,
                        patch_size = args.patch_size,
                        d_model = args.d_model,
                        nhead = args.nhead,
                        dropout = args.dropout,
                        max_seq_len = args.content_max_seq_len)

    model.load_state_dict(torch.load(content_model_weights, map_location="cpu"))
    model = model.to(device)
    
    # Crop and transform cell images
    image_tensor = [image_to_tensor(image.crop(bbox), size=(112, 448)) for bbox in pred_bbox]
    image_tensors = torch.cat(image_tensor, dim=0).to(device)
    
    cells = [''] if none_first else []
    
    # Automatically apply OCR for Korean text detection in tables
    if args.korean:
        pred_cell, to_model_idx = Use_ocr(image, pred_bbox, args.ocr, cells, args.api_url, args.secret_key, Path(os.path.dirname(args.image_path))/"temporary.png")
        
        # Complement with model-based recognition when OCR fails
        for idx in (to_model_idx or []):
            img_tensor = image_tensors[idx].unsqueeze(0)
            cell_output = Cell_content(img_tensor, model, vocab, args.content_max_decode_len, device, INVALID_CELL_TOKEN)
            pred_cell[idx] = ''.join(cell_output)
    else:
        pred_cell = Cell_content(image_tensors, model, vocab, args.content_max_decode_len, device, INVALID_CELL_TOKEN)

    ### ------------ Result HTML code ------------ ###
    # Merge structural layout with cell contents to create final table HTML
    pred_code = build_table_from_html_and_cell(pred_html, pred_cell)
    pred_code = "".join(pred_code)
    pred_code = html_table_template(pred_code)
    
    return pred_html, pred_bbox, pred_code