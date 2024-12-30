from src.utils import cell_str_to_token_list
from modules import autoregressive_decode
import numpy as np
import re
import requests
import uuid
import time
import json
import os
from PIL import Image

# 1: EasyOCR
def UseEasyOCR(each_bboxes, OCR_engine, cells):
    to_model_idx = []
    for idx, one_bbox in enumerate(each_bboxes):
        output = OCR_engine.readtext(one_bbox, detail=0)
        output = '  '.join(output)
        # Mark for model processing if OCR result is empty or contains no Korean/numbers
        if (output=='') or not (re.search('[0-9\u1100-\u11FF\u3130-\u318F\uAC00-\uD7A3]', output)):
            to_model_idx.append(idx)

        cells.append(output)
    return cells, to_model_idx

# 2: PaddleOCR
def UsePaddleOCR(each_bboxes, OCR_engine, cells):
    to_model_idx = []
    for idx, one_bbox in enumerate(each_bboxes):
        output = OCR_engine.ocr(one_bbox, cls=False)[0]
        if output==None:
            output = ''
        else:
            output = [o[1][0] if o else '' for o in output]
            output = ''.join(output)
        if (output=='') or not (re.search('[0-9\u1100-\u11FF\u3130-\u318F\uAC00-\uD7A3]', output)):
            to_model_idx.append(idx)

        cells.append(output)
    return cells, to_model_idx

# 3: NaverClovaOCR
def UseNaverClovaOCR(each_bboxes, naverOCR_engine, cells, api_url, secret_key, file_path):
    assert api_url, "api_url is required when you use NaverClova OCR(ocr == 3)"
    assert secret_key, "secret key is required when you use NaverClova OCR(ocr == 3)"
    
    to_model_idx = []
    for idx, one_bbox in enumerate(each_bboxes):
        output = naverOCR_engine(one_bbox, api_url, secret_key, file_path)
        output = '  '.join(output)
        if (output=='') or not (re.search('[0-9\u1100-\u11FF\u3130-\u318F\uAC00-\uD7A3]', output)):
            to_model_idx.append(idx)
        
        cells.append(output)
    return cells, to_model_idx

def naverOCR_engine(one_bbox, api_url, secret_key, file_path):
    img_pil = Image.fromarray(one_bbox)
    img_pil.save(file_path)
    # Prepare and execute OCR request
    request_json = {
        'images': [{
                'format': 'jpg',
                'name': 'demo'
            }],
        'requestId': str(uuid.uuid4()),
        'version': 'V2',
        'timestamp': int(round(time.time() * 1000))
    }
    payload = {'message': json.dumps(request_json).encode('UTF-8')}
    files = [('file', open(file_path, 'rb'))]
    headers = {'X-OCR-SECRET': secret_key}
    
    # request API
    response = requests.request("POST", api_url, headers=headers, data=payload, files=files)
    # Parse the result
    result = response.text.encode('utf8')
    parsed_data = json.loads(result.decode('utf-8'))
    # Extract inferText
    output = []
    if "images" in parsed_data and parsed_data["images"]:
        for image in parsed_data["images"]:
            if "fields" in image and image["fields"]:
                for field in image["fields"]:
                    if "inferText" in field:
                        output.append(field["inferText"])
    output = '' if output == [] else output
    os.remove(file_path)
    return output

# Select and execute OCR engine
def Use_ocr(image, bboxes, ocr, cells, api_url=None, secret_key=None, file_path=None):
    image = np.array(image)
    npad = ((200,200),(200,200),(0,0))    
    each_bboxes = [np.pad(image[b[1]:b[3],b[0]:b[2],:], npad, 'constant', constant_values=255) for b in bboxes]
    
    try:
        if ocr == 1:
            import easyocr
            OCR_engine = easyocr.Reader(['ko', 'en'], gpu=True)
            ocr_output, to_model_idx = UseEasyOCR(each_bboxes, OCR_engine, cells)
        elif ocr == 2:
            from paddleocr import PaddleOCR
            OCR_engine = PaddleOCR(lang="korean", use_gpu=True, gpu_id=0)
            ocr_output, to_model_idx = UsePaddleOCR(each_bboxes, OCR_engine, cells)
        elif ocr == 3:
            if not api_url or not secret_key:
                raise ValueError("Both api_url and secret_key are required when using NaverClovaOCR (ocr:3)")
            ocr_output, to_model_idx = UseNaverClovaOCR(each_bboxes, naverOCR_engine, cells, api_url, secret_key, file_path)
    
    except Exception as e:
        print(f"Error occurred: {e}")
        ocr_output = None
        to_model_idx = None
        
    return ocr_output, to_model_idx

# Cell content recognition using model
def Cell_content(bbox, model, vocab, content_max_decode_len, device, INVALID_CELL_TOKEN):
    cell_output = autoregressive_decode(
                    model = model,
                    image = bbox,
                    prefix = [vocab.token_to_id("[cell]")],
                    max_decode_len = content_max_decode_len,
                    eos_id = vocab.token_to_id("<eos>"),
                    device = device,
                    token_whitelist=None,
                    token_blacklist = [vocab.token_to_id(i) for i in INVALID_CELL_TOKEN])
    cell_output = cell_output.detach().cpu().numpy()
    cell_output = vocab.decode_batch(cell_output, skip_special_tokens=False)
    cell_output = [cell_str_to_token_list(i) for i in cell_output]
    cell_output = [re.sub(r'(\d).\s+(\d)', r'\1.\2', i) for i in cell_output]
    return cell_output