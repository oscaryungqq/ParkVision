import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict
from statistics import median
import torch

OCCUPIED_CLASS_ID = 1
EMPTY_SPOT_CLASS_ID = 0
CAR_CLASS_ID = 2
DETECTION_FRAME_WIDTH = 640
DETECTION_FRAME_HEIGHT = 640

class ParkVisionProcessor:
    def __init__(self, occupied_model_path, spot_model_path, car_model_path):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.occupied_model = YOLO(occupied_model_path).to(self.device)
        self.parking_spot_model = YOLO(spot_model_path).to(self.device)
        self.car_model = YOLO(car_model_path).to(self.device)
        
        self.deepsort = DeepSort(max_age=100, n_init=3, nn_budget=50, embedder_gpu=(self.device != "cpu"))
        print("ParkVisionProcessor initialized and models loaded.")

    def process_video(self, input_path: str, output_path: str):
        cap, out, video_props = self._setup_video_io(input_path, output_path)
        if not cap:
            return None

        state = self._initialize_state()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            state['frame_count'] += 1
            
            frame_specialized = self._preprocess_frame_specialized(frame)
            frame_general = self._preprocess_frame_general(frame)
            
            parking_detections, car_detections, occupied_detections = self._run_inference(frame_specialized, frame_general)

            for spot_id, spot_bbox in state['occupied_spots'].items():
                if spot_id.startswith("direct_"):
                    continue
                x1, y1, x2, y2 = spot_bbox
                matched = False
                for det in occupied_detections:
                    det_bbox_tlwh = det[0]
                    det_x1, det_y1, det_w, det_h = det_bbox_tlwh
                    det_x2, det_y2 = det_x1 + det_w, det_y1 + det_h
                    if self._calculate_iou([x1, y1, x2, y2], [det_x1, det_y1, det_x2, det_y2]) > 0.5:
                        matched = True
                        break
                if not matched:
                    occupied_detections.append(([x1, y1, x2-x1, y2-y1], 0.7, "occupied"))

            all_detections = parking_detections + car_detections + occupied_detections
            tracks = self.deepsort.update_tracks(all_detections, frame=frame_general)

            self._update_and_draw(frame, tracks, state, video_props)
            
            out.write(frame)
            print(f"Processed frame {state['frame_count']}...")

        self._cleanup(cap, out)
        return output_path

    def _initialize_state(self):
        return {
            'empty_spots': {},   
            'cars': {},          
            'occupied_spots': {},
            'overlap_tracker': defaultdict(lambda: 0),
            'history': {'empty': [], 'occupied': []},
            'max_total_spots': 0,
            'frame_count': 0
        }

    def _setup_video_io(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened(): return None, None, None

        video_props = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(cap.get(cv2.CAP_PROP_FPS)),
            'scale_x': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) / DETECTION_FRAME_WIDTH,
            'scale_y': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) / DETECTION_FRAME_HEIGHT,
        }
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, video_props['fps'], (video_props['width'], video_props['height']))
        return cap, out, video_props

    def _run_inference(self, frame_specialized, frame_general):
        parking_detections = []
        for res in self.parking_spot_model(frame_specialized, conf=0.2, verbose=False):
            for box in res.boxes:
                if int(box.cls) == EMPTY_SPOT_CLASS_ID:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    parking_detections.append(([x1, y1, x2-x1, y2-y1], float(box.conf[0]), "empty_spot"))
        
        car_detections = []
        for res in self.car_model(frame_general, verbose=False):
            for box in res.boxes:
                if int(box.cls) == CAR_CLASS_ID:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    car_detections.append(([x1, y1, x2-x1, y2-y1], float(box.conf[0]), "car"))

        occupied_detections = []
        for res in self.occupied_model(frame_specialized, conf=0.1, verbose=False):
             for box in res.boxes:
                if int(box.cls) == OCCUPIED_CLASS_ID:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    occupied_detections.append(([x1, y1, x2-x1, y2-y1], float(box.conf[0]), "occupied"))
        
        return parking_detections, car_detections, occupied_detections

    def _update_and_draw(self, frame, tracks, state, video_props):
        current_empty_spots = {}
        current_cars = {}
        current_occupied = {}

        for track in tracks:
            if not track.is_confirmed(): continue
            
            tid = track.track_id
            bbox = track.to_tlbr() 
            label = track.det_class
            conf = track.get_det_conf() if track.get_det_conf() else 0.7

            if label == "empty_spot":
                matched = False
                for existing_bbox in current_empty_spots.values():
                    if self._calculate_iou(bbox, existing_bbox) > 0.5:
                        matched = True
                        break
                if not matched:
                    current_empty_spots[tid] = bbox
                if tid not in state['empty_spots']:
                    state['empty_spots'][tid] = bbox

            elif label == "car":
                current_cars[tid] = bbox
            elif label == "occupied":
                current_occupied[tid] = bbox

        for eid in list(state['empty_spots'].keys()):
            if eid not in current_empty_spots and eid not in state['occupied_spots']:
                del state['empty_spots'][eid]

        fps = video_props['fps']
        for eid, ebox in current_empty_spots.items():
            if eid in state['occupied_spots']: continue
            
            overlapping = False
            for cid, cbox in current_cars.items():
                if self._calculate_iou(ebox, cbox) > 0.5:
                    overlapping = True
                    state['overlap_tracker'][eid] += 1
                    if state['overlap_tracker'][eid] >= int(fps * 0.5):
                        state['occupied_spots'][eid] = cbox
                    break
            if not overlapping:
                state['overlap_tracker'][eid] = 0

        for tid, obox in current_occupied.items():
            matched = False
            for sid, sbox in state['occupied_spots'].items():
                if self._calculate_iou(obox, sbox) > 0.5:
                    matched = True; break
            if not matched:
                for eid, ebox in state['empty_spots'].items():
                    if self._calculate_iou(obox, ebox) > 0.5:
                        state['occupied_spots'][eid] = obox
                        break
                state['occupied_spots'][f"direct_{tid}"] = obox

        sx, sy = video_props['scale_x'], video_props['scale_y']
        
        for tid, bbox in state['empty_spots'].items():
            if tid in state['occupied_spots']: continue
            x1, y1, x2, y2 = map(int, [bbox[0]*sx, bbox[1]*sy, bbox[2]*sx, bbox[3]*sy])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Parking Spot", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for sid, bbox in state['occupied_spots'].items():
            x1, y1, x2, y2 = map(int, [bbox[0]*sx, bbox[1]*sy, bbox[2]*sx, bbox[3]*sy])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "Occupied", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        state['history']['empty'].append(len(current_empty_spots))
        state['history']['occupied'].append(len(state['occupied_spots']))
        for k in ['empty', 'occupied']:
            if len(state['history'][k]) > 30: state['history'][k].pop(0)

        smooth_e = int(median(state['history']['empty'])) if state['history']['empty'] else 0
        smooth_o = int(median(state['history']['occupied'])) if state['history']['occupied'] else 0
        state['max_total_spots'] = max(state['max_total_spots'], smooth_e + smooth_o)

        cv2.putText(frame, f"Empty Spots: {smooth_e}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Total Cars: {len(current_cars)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Occupied Spots: {smooth_o}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Total Capacity: {state['max_total_spots']}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def _cleanup(self, cap, out):
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    @staticmethod
    def _preprocess_frame_specialized(frame):
        f = cv2.resize(frame, (DETECTION_FRAME_WIDTH, DETECTION_FRAME_HEIGHT))
        f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        f = clahe.apply(f)
        return cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _preprocess_frame_general(frame):
        return cv2.resize(frame, (DETECTION_FRAME_WIDTH, DETECTION_FRAME_HEIGHT))

    @staticmethod
    def _calculate_iou(box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        xi1, yi1, xi2, yi2 = max(x1, x3), max(y1, y3), min(x2, x4), min(y2, y4)
        if xi2 < xi1 or yi2 < yi1: return 0.0
        inter = (xi2 - xi1) * (yi2 - yi1)
        union = (x2 - x1) * (y2 - y1) + (x4 - x3) * (y4 - y3) - inter
        return inter / union if union > 0 else 0.0