##### THIS IS FINAL SCRIPT. 


import cv2
import pytesseract
import numpy as np
from datetime import datetime
import csv
import os
from ultralytics import YOLO
import logging
import torch
import threading
import queue
from collections import deque
import time

# Configuration
VIDEO_PATH = "sample.mp4"  # Change to your video path
OUTPUT_CSV = "license_plate_logs.csv"
LOG_FILE = "processing.log"
MIN_PLATE_CONFIDENCE = 0.7
OCR_MIN_CONFIDENCE = 0.75

# Threading configuration
MAX_BUFFER_SIZE = 100  # Maximum frames in buffer
NUM_WORKER_THREADS = 3  # Number of processing threads
FRAME_SKIP = 1  # Process every N frames (1 = all frames)
DETECTION_BATCH_SIZE = 5  # Process frames in batches

# Duplicate filtering configuration
MIN_TIME_BETWEEN_SAME_PLATE = 2.0  # Minimum seconds between same plate detections
MIN_DISTANCE_THRESHOLD = 50  # Minimum pixel distance for same plate in different position
SIMILARITY_THRESHOLD = 0.8  # String similarity threshold for plate matching

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

# Thread-safe frame buffer and tracking
frame_buffer = queue.Queue(maxsize=MAX_BUFFER_SIZE)
results_queue = queue.Queue()
processing_complete = threading.Event()

# Global tracking for duplicate detection (thread-safe)
plate_tracking = {}  # {plate_text: {'last_time': datetime, 'last_position': (x,y), 'count': int}}
tracking_lock = threading.Lock()

# Initialize CSV file
def initialize_csv():
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Frame_Number', 'Plate', 'Confidence', 'ProcessingTime', 'X1', 'Y1', 'X2', 'Y2'])

def initialize_models():
    """Load and warm up models"""
    logging.info("Loading vehicle detection model...")
    vehicle_model = YOLO('models/yolov8n.pt')
    
    logging.info("Loading license plate model...")
    lp_model = YOLO('models/best.pt')
    
    # Warm up models with smaller dummy tensor for faster startup
    dummy = torch.zeros((1, 3, 320, 320))
    vehicle_model(dummy)
    lp_model(dummy)
    
    return vehicle_model, lp_model

def calculate_string_similarity(str1, str2):
    """Calculate similarity between two strings using Levenshtein distance"""
    if len(str1) == 0 or len(str2) == 0:
        return 0.0
    
    # Convert to uppercase for comparison
    str1, str2 = str1.upper(), str2.upper()
    
    # Handle exact match
    if str1 == str2:
        return 1.0
    
    # Calculate Levenshtein distance
    matrix = [[0] * (len(str2) + 1) for _ in range(len(str1) + 1)]
    
    for i in range(len(str1) + 1):
        matrix[i][0] = i
    for j in range(len(str2) + 1):
        matrix[0][j] = j
    
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i-1] == str2[j-1]:
                matrix[i][j] = matrix[i-1][j-1]
            else:
                matrix[i][j] = min(
                    matrix[i-1][j] + 1,    # deletion
                    matrix[i][j-1] + 1,    # insertion
                    matrix[i-1][j-1] + 1   # substitution
                )
    
    # Calculate similarity ratio
    max_len = max(len(str1), len(str2))
    distance = matrix[len(str1)][len(str2)]
    similarity = 1.0 - (distance / max_len)
    
    return similarity

def calculate_bbox_distance(bbox1, bbox2):
    """Calculate distance between centers of two bounding boxes"""
    x1_center = (bbox1[0] + bbox1[2]) / 2
    y1_center = (bbox1[1] + bbox1[3]) / 2
    x2_center = (bbox2[0] + bbox2[2]) / 2
    y2_center = (bbox2[1] + bbox2[3]) / 2
    
    distance = ((x1_center - x2_center) ** 2 + (y1_center - y2_center) ** 2) ** 0.5
    return distance

def is_duplicate_detection(plate_text, bbox, timestamp):
    """Check if this detection is a duplicate of a recent one"""
    with tracking_lock:
        current_time = timestamp
        center_pos = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        
        # Check for exact matches first
        if plate_text in plate_tracking:
            last_detection = plate_tracking[plate_text]
            time_diff = (current_time - last_detection['last_time']).total_seconds()
            
            # If same plate detected within time threshold
            if time_diff < MIN_TIME_BETWEEN_SAME_PLATE:
                # Check if position is similar (same vehicle moving)
                last_pos = last_detection['last_position']
                distance = ((center_pos[0] - last_pos[0]) ** 2 + (center_pos[1] - last_pos[1]) ** 2) ** 0.5
                
                if distance < MIN_DISTANCE_THRESHOLD:
                    return True  # Duplicate detection
        
        # Check for similar plates (OCR variations)
        for tracked_plate, tracking_data in plate_tracking.items():
            if tracked_plate == plate_text:
                continue
                
            # Calculate string similarity
            similarity = calculate_string_similarity(plate_text, tracked_plate)
            
            if similarity >= SIMILARITY_THRESHOLD:
                time_diff = (current_time - tracking_data['last_time']).total_seconds()
                
                if time_diff < MIN_TIME_BETWEEN_SAME_PLATE:
                    # Check position similarity
                    distance = calculate_bbox_distance(bbox, 
                        (tracking_data['last_position'][0] - 50, tracking_data['last_position'][1] - 25,
                         tracking_data['last_position'][0] + 50, tracking_data['last_position'][1] + 25))
                    
                    if distance < MIN_DISTANCE_THRESHOLD:
                        return True  # Similar plate in similar position
        
        # Update tracking for this plate
        plate_tracking[plate_text] = {
            'last_time': current_time,
            'last_position': center_pos,
            'count': plate_tracking.get(plate_text, {}).get('count', 0) + 1
        }
        
        return False
    """Enhanced OCR processing for license plates"""
    try:
        # Resize plate image if too small
        h, w = plate_img.shape[:2]
        if h < 30 or w < 80:
            scale_factor = max(30/h, 80/w)
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            plate_img = cv2.resize(plate_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale and enhance contrast
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # Apply advanced preprocessing
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Multiple thresholding approaches
        _, thresh1 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh2 = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Try OCR on both thresholded images
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        
        texts = []
        for thresh in [thresh1, thresh2]:
            try:
                text = pytesseract.image_to_string(thresh, config=custom_config).strip()
                if text:
                    texts.append(text)
            except:
                continue
        
        # Choose best result
        best_text = None
        max_len = 0
        
        for text in texts:
            clean_text = ''.join([c for c in text if c.isalnum()])
            if len(clean_text) > max_len and len(clean_text) >= 4:
                best_text = clean_text
                max_len = len(clean_text)
        
        if best_text:
            # Character normalization
            char_map = {'O':'0', 'I':'1', 'Z':'2', 'B':'8', 'S':'5'}
            normalized = ''.join([char_map.get(c.upper(), c.upper()) for c in best_text])
            return normalized
        
        return None
        
    except Exception as e:
        logging.error(f"OCR processing error: {str(e)}")
        return None

def calculate_string_similarity(str1, str2):
    """Calculate similarity between two strings using Levenshtein distance"""
    if len(str1) == 0 or len(str2) == 0:
        return 0.0
    
    # Convert to uppercase for comparison
    str1, str2 = str1.upper(), str2.upper()
    
    # Handle exact match
    if str1 == str2:
        return 1.0
    
    # Calculate Levenshtein distance
    matrix = [[0] * (len(str2) + 1) for _ in range(len(str1) + 1)]
    
    for i in range(len(str1) + 1):
        matrix[i][0] = i
    for j in range(len(str2) + 1):
        matrix[0][j] = j
    
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i-1] == str2[j-1]:
                matrix[i][j] = matrix[i-1][j-1]
            else:
                matrix[i][j] = min(
                    matrix[i-1][j] + 1,    # deletion
                    matrix[i][j-1] + 1,    # insertion
                    matrix[i-1][j-1] + 1   # substitution
                )
    
    # Calculate similarity ratio
    max_len = max(len(str1), len(str2))
    distance = matrix[len(str1)][len(str2)]
    similarity = 1.0 - (distance / max_len)
    
    return similarity

def calculate_bbox_distance(bbox1, bbox2):
    """Calculate distance between centers of two bounding boxes"""
    x1_center = (bbox1[0] + bbox1[2]) / 2
    y1_center = (bbox1[1] + bbox1[3]) / 2
    x2_center = (bbox2[0] + bbox2[2]) / 2
    y2_center = (bbox2[1] + bbox2[3]) / 2
    
    distance = ((x1_center - x2_center) ** 2 + (y1_center - y2_center) ** 2) ** 0.5
    return distance

def is_duplicate_detection(plate_text, bbox, timestamp):
    """Check if this detection is a duplicate of a recent one"""
    with tracking_lock:
        current_time = timestamp
        center_pos = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        
        # Check for exact matches first
        if plate_text in plate_tracking:
            last_detection = plate_tracking[plate_text]
            time_diff = (current_time - last_detection['last_time']).total_seconds()
            
            # If same plate detected within time threshold
            if time_diff < MIN_TIME_BETWEEN_SAME_PLATE:
                # Check if position is similar (same vehicle moving)
                last_pos = last_detection['last_position']
                distance = ((center_pos[0] - last_pos[0]) ** 2 + (center_pos[1] - last_pos[1]) ** 2) ** 0.5
                
                if distance < MIN_DISTANCE_THRESHOLD:
                    return True  # Duplicate detection
        
        # Check for similar plates (OCR variations)
        for tracked_plate, tracking_data in plate_tracking.items():
            if tracked_plate == plate_text:
                continue
                
            # Calculate string similarity
            similarity = calculate_string_similarity(plate_text, tracked_plate)
            
            if similarity >= SIMILARITY_THRESHOLD:
                time_diff = (current_time - tracking_data['last_time']).total_seconds()
                
                if time_diff < MIN_TIME_BETWEEN_SAME_PLATE:
                    # Check position similarity
                    distance = calculate_bbox_distance(bbox, 
                        (tracking_data['last_position'][0] - 50, tracking_data['last_position'][1] - 25,
                         tracking_data['last_position'][0] + 50, tracking_data['last_position'][1] + 25))
                    
                    if distance < MIN_DISTANCE_THRESHOLD:
                        return True  # Similar plate in similar position
        
        # Update tracking for this plate
        plate_tracking[plate_text] = {
            'last_time': current_time,
            'last_position': center_pos,
            'count': plate_tracking.get(plate_text, {}).get('count', 0) + 1
        }
        
        return False

def frame_capture_worker():
    """Worker thread for capturing frames"""
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        logging.error(f"Failed to open video: {VIDEO_PATH}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logging.info(f"Video FPS: {fps}, Total frames: {total_frames}")
    
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Skip frames if configured
        if frame_count % FRAME_SKIP != 0:
            continue
        
        # Calculate timestamp
        timestamp = datetime.now()
        
        try:
            # Add frame to buffer (non-blocking)
            frame_data = {
                'frame': frame.copy(),
                'frame_number': frame_count,
                'timestamp': timestamp
            }
            
            frame_buffer.put(frame_data, timeout=0.01)
            
            # Progress logging
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps_actual = frame_count / elapsed
                logging.info(f"Captured {frame_count}/{total_frames} frames, FPS: {fps_actual:.1f}, Buffer size: {frame_buffer.qsize()}")
                
        except queue.Full:
            logging.warning(f"Frame buffer full, dropping frame {frame_count}")
            continue
    
    cap.release()
    logging.info(f"Frame capture completed. Total frames captured: {frame_count}")



def process_plate_image(plate_img):
    """Enhanced OCR processing for license plates"""
    try:
        # Resize plate image if too small
        h, w = plate_img.shape[:2]
        if h < 30 or w < 80:
            scale_factor = max(30/h, 80/w)
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            plate_img = cv2.resize(plate_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale and enhance contrast
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # Apply advanced preprocessing
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Multiple thresholding approaches
        _, thresh1 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh2 = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Try OCR on both thresholded images
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        
        texts = []
        for thresh in [thresh1, thresh2]:
            try:
                text = pytesseract.image_to_string(thresh, config=custom_config).strip()
                if text:
                    texts.append(text)
            except:
                continue
        
        # Choose best result
        best_text = None
        max_len = 0
        
        for text in texts:
            clean_text = ''.join([c for c in text if c.isalnum()])
            if len(clean_text) > max_len and len(clean_text) >= 4:
                best_text = clean_text
                max_len = len(clean_text)
        
        if best_text:
            # Character normalization
            char_map = {'O':'0', 'I':'1', 'Z':'2', 'B':'8', 'S':'5'}
            normalized = ''.join([char_map.get(c.upper(), c.upper()) for c in best_text])
            return normalized
        
        return None
        
    except Exception as e:
        logging.error(f"OCR processing error: {str(e)}")
        return None


def detection_worker(worker_id, vehicle_model, lp_model):
    """Worker thread for processing frames"""
    logging.info(f"Detection worker {worker_id} started")
    processed_count = 0
    
    while not processing_complete.is_set() or not frame_buffer.empty():
        try:
            # Get frame from buffer
            frame_data = frame_buffer.get(timeout=1.0)
            
            frame = frame_data['frame']
            frame_number = frame_data['frame_number']
            timestamp = frame_data['timestamp']
            
            process_start_time = time.time()
            
            # Vehicle detection with optimized parameters
            vehicles = vehicle_model(frame, classes=[2, 3, 5, 7], conf=0.4, verbose=False)
            
            for vehicle in vehicles:
                if not hasattr(vehicle, 'boxes') or vehicle.boxes is None:
                    continue
                
                # License plate detection on vehicle region
                plates = lp_model(frame, conf=0.5, verbose=False)
                
                for plate in plates:
                    if hasattr(plate, 'boxes') and plate.boxes is not None:
                        for box in plate.boxes:
                            conf = float(box.conf.item())
                            if conf > MIN_PLATE_CONFIDENCE:
                                # Extract plate ROI
                                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                
                                # Validate bounding box
                                h, w = frame.shape[:2]
                                if x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h and x2 > x1 and y2 > y1:
                                    plate_roi = frame[y1:y2, x1:x2]
                                    
                                    # Perform OCR
                                    plate_text = process_plate_image(plate_roi)
                                    
                                    if plate_text:
                                        # Check for duplicate detection
                                        if not is_duplicate_detection(plate_text, (x1, y1, x2, y2), timestamp):
                                            # Calculate processing time
                                            proc_time = (time.time() - process_start_time) * 1000
                                            
                                            # Add result to queue
                                            result = {
                                                'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"),
                                                'frame_number': frame_number,
                                                'plate_text': plate_text,
                                                'confidence': conf,
                                                'processing_time': proc_time,
                                                'bbox': (x1, y1, x2, y2),
                                                'worker_id': worker_id
                                            }
                                            
                                            results_queue.put(result)
                                            logging.info(f"Worker {worker_id}: NEW PLATE {plate_text} (Conf: {conf:.2f}) in frame {frame_number}")
                                        else:
                                            logging.debug(f"Worker {worker_id}: DUPLICATE {plate_text} in frame {frame_number} - skipped")
            
            processed_count += 1
            frame_buffer.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"Worker {worker_id} processing error: {str(e)}")
            continue
    
    logging.info(f"Detection worker {worker_id} completed. Processed {processed_count} frames")

def results_writer():
    """Worker thread for writing results to CSV"""
    logging.info("Results writer started")
    results_count = 0
    
    while not processing_complete.is_set() or not results_queue.empty():
        try:
            result = results_queue.get(timeout=1.0)
            
            # Write to CSV
            with open(OUTPUT_CSV, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    result['timestamp'],
                    result['frame_number'],
                    result['plate_text'],
                    f"{result['confidence']:.2f}",
                    f"{result['processing_time']:.2f}ms",
                    result['bbox'][0],  # x1
                    result['bbox'][1],  # y1
                    result['bbox'][2],  # x2
                    result['bbox'][3]   # y2
                ])
            
            results_count += 1
            results_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"Results writer error: {str(e)}")
    
    logging.info(f"Results writer completed. Wrote {results_count} results")

def process_video_multithreaded():
    """Main processing function with multithreading"""
    # Initialize CSV
    initialize_csv()
    
    # Load models (only once, shared across threads)
    vehicle_model, lp_model = initialize_models()
    
    # Start frame capture thread
    capture_thread = threading.Thread(target=frame_capture_worker, name="FrameCapture")
    capture_thread.daemon = True
    capture_thread.start()
    
    # Start detection worker threads
    detection_threads = []
    for i in range(NUM_WORKER_THREADS):
        thread = threading.Thread(
            target=detection_worker, 
            args=(i, vehicle_model, lp_model),
            name=f"DetectionWorker-{i}"
        )
        thread.daemon = True
        thread.start()
        detection_threads.append(thread)
    
    # Start results writer thread
    writer_thread = threading.Thread(target=results_writer, name="ResultsWriter")
    writer_thread.daemon = True
    writer_thread.start()
    
    # Wait for frame capture to complete
    capture_thread.join()
    logging.info("Frame capture completed")
    
    # Wait for all frames to be processed
    frame_buffer.join()
    logging.info("Frame buffer empty")
    
    # Signal processing complete
    processing_complete.set()
    
    # Wait for all threads to complete
    for thread in detection_threads:
        thread.join()
    
    writer_thread.join()
    
    logging.info("All processing completed")

def monitor_progress():
    """Monitor processing progress"""
    start_time = time.time()
    last_buffer_size = 0
    
    while not processing_complete.is_set():
        buffer_size = frame_buffer.qsize()
        results_size = results_queue.qsize()
        elapsed = time.time() - start_time
        
        logging.info(f"Progress - Buffer: {buffer_size}, Results: {results_size}, Elapsed: {elapsed:.1f}s")
        
        # Check if processing is stalled
        if buffer_size > last_buffer_size and buffer_size >= MAX_BUFFER_SIZE * 0.9:
            logging.warning("Buffer nearly full - processing may be slower than capture")
        
        last_buffer_size = buffer_size
        time.sleep(5)  # Update every 5 seconds

if __name__ == "__main__":
    logging.info("Starting optimized license plate processing...")
    
    # Start progress monitor in separate thread
    monitor_thread = threading.Thread(target=monitor_progress, name="ProgressMonitor")
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Start main processing
    start_time = time.time()
    process_video_multithreaded()
    
    total_time = time.time() - start_time
    logging.info(f"Processing finished in {total_time:.2f} seconds")
    
    # Print final statistics
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, 'r') as f:
            line_count = sum(1 for line in f) - 1  # Subtract header
        logging.info(f"Total unique license plates detected: {line_count}")
        
        # Print tracking statistics
        with tracking_lock:
            total_detections = sum(data['count'] for data in plate_tracking.values())
            unique_plates = len(plate_tracking)
            logging.info(f"Total detections before filtering: {total_detections}")
            logging.info(f"Unique plates after filtering: {unique_plates}")
            logging.info(f"Duplicate detections filtered: {total_detections - line_count}")