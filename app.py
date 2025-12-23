import os
import base64
import collections
import io
import cv2
import numpy as np
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from paddleocr import PaddleOCR

app = Flask(__name__)
CORS(app)

# Initialize PaddleOCR
# use_textline_orientation=True allows detecting rotated text
# lang='en' for English support
# show_log argument is removed as it caused errors
print("Loading PaddleOCR...")
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
print("PaddleOCR Loaded!")

def group_boxes_into_lines(result, y_threshold=10):
    """
    Groups OCR bounding boxes into lines based on Y-coordinates.
    result structure: [[box, (text, conf)], ...]
    """
    if not result:
        return []

    boxes_with_info = []
    
    for line in result:
        # line is [box, (text, conf)]
        coords, (text, conf) = line
        
        # Coords might be list of lists or numpy array
        ys = [p[1] for p in coords]
        xs = [p[0] for p in coords]
        y_center = sum(ys) / len(ys)
        x_min = min(xs)
        boxes_with_info.append({
            "text": text,
            "y_center": y_center,
            "x_min": x_min,
            "height": max(ys) - min(ys)
        })

    # Sort by Y-center first
    boxes_with_info.sort(key=lambda b: b['y_center'])

    lines = []
    current_line = []
    
    for box in boxes_with_info:
        if not current_line:
            current_line.append(box)
            continue
            
        # Check if this box belongs to the current line (similar Y center)
        avg_height = sum(b['height'] for b in current_line) / len(current_line)
        last_y = current_line[-1]['y_center']
        
        # Adaptive threshold: 50% of box height is usually good
        if abs(box['y_center'] - last_y) < (avg_height * 0.5):
            current_line.append(box)
        else:
            lines.append(current_line)
            current_line = [box]
            
    if current_line:
        lines.append(current_line)

    # For each line, sort by X-min to read left-to-right
    final_text_lines = []
    for line in lines:
        line.sort(key=lambda b: b['x_min'])
        # Join text with large spacing to simulate "Item ... Price"
        joined_text = "   ".join([b['text'] for b in line])
        final_text_lines.append(joined_text)

    return final_text_lines

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "engine": "PaddleOCR"})

@app.route('/analyze', methods=['POST'])
def analyze_receipt():
    try:
        data = request.json
        if 'image' not in data:
            return jsonify({"error": "No image provided"}), 400
            
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(image_data)
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
             return jsonify({"error": "Failed to decode image"}), 400

        # Run OCR
        result = ocr.ocr(img)
        
        formatted_lines = []
        
        # PaddleOCR returns a list of results (one per image). We sent one image.
        if result and isinstance(result, list):
             img_result = result[0]
             
             if isinstance(img_result, dict):
                 # Extract from dict
                 boxes = img_result.get('rec_boxes', [])
                 texts = img_result.get('rec_texts', [])
                 scores = img_result.get('rec_scores', [])
                 
                 print(f"DEBUG - Boxes type: {type(boxes)}")
                 if len(boxes) > 0:
                     print(f"DEBUG - First box: {boxes[0]}")
                     print(f"DEBUG - First box type: {type(boxes[0])}")
                 
                 for box, text, score in zip(boxes, texts, scores):
                     # Ensure box is a list of points [[x,y], [x,y]...]
                     if isinstance(box, np.ndarray):
                         box = box.tolist()
                     
                     # Check if box is flattened [x1, y1, x2, y2, ...]
                     # If it is a list of numbers, reshape it
                     if isinstance(box, list) and len(box) > 0 and isinstance(box[0], (int, float)):
                         # Assuming 4 points -> 8 numbers
                         if len(box) % 2 == 0:
                             box = [[box[i], box[i+1]] for i in range(0, len(box), 2)]
                     
                     formatted_lines.append([box, (text, score)])
                     
             elif isinstance(img_result, list):
                 # Old format: It is a list of lines
                 formatted_lines = img_result
             else:
                 print(f"Unknown img_result type: {type(img_result)}")

        else:
             print(f"Unknown result type: {type(result)}")
             
        if not formatted_lines:
            return jsonify({"text": "", "lines": []})
        
        structured_lines = group_boxes_into_lines(formatted_lines)
        full_text = "\n".join(structured_lines)
        
        return jsonify({ 
            "text": full_text,
            "lines": structured_lines 
        })

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port)
