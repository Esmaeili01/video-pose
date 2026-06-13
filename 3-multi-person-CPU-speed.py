import sys
import cv2
import time
import threading
from queue import Queue, Empty

# --- FORCE REGISTRY UNIFICATION ---
from mmengine.registry import TRANSFORMS, init_default_scope
from mmdet.datasets.transforms import PackDetInputs
from mmpose.datasets.transforms import PackPoseInputs

TRANSFORMS.register_module(module=PackDetInputs)
TRANSFORMS.register_module(module=PackPoseInputs)

from mmdet.apis import inference_detector, init_detector
from mmpose.apis import inference_topdown, init_model
from mmpose.visualization import PoseLocalVisualizer
from mmpose.structures import merge_data_samples

from PyQt6.QtWidgets import (QApplication, QLabel, QVBoxLayout, QWidget, QComboBox, 
                             QPushButton, QSlider, QHBoxLayout, QCheckBox)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap

# --- CONFIG ---
DET_CONFIG = 'detection/rtmdet_nano.py'
DET_CHECKPOINT = 'detection/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth'
VIDEO_PATH = '60min_mb720p.mp4'

MODELS = {
    "RTMPose-T": {"config": "models/rtmpose-t_config.py", "checkpoint": "models/rtmpose-t_simcc-ucoco_dw-ucoco_270e-256x192-dcf277bf_20230728.pth"},
    "RTMPose-S": {"config": "models/rtmpose-s_config.py", "checkpoint": "models/rtmpose-s_simcc-ucoco_dw-ucoco_270e-256x192-3fd922c8_20230728.pth"},
    "RTMPose-M": {"config": "models/rtmpose-m_config.py", "checkpoint": "models/rtmpose-m_simcc-ucoco_dw-ucoco_270e-256x192-c8b76419_20230728.pth"}
}

class InferenceWorker(threading.Thread):
    def __init__(self, task_queue, result_queue, model_name):
        super().__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.model_name = model_name
        self.daemon = True
        self.detector = init_detector(DET_CONFIG, DET_CHECKPOINT, device='cpu')
        init_default_scope('mmpose')
        self.pose_model = init_model(MODELS[model_name]['config'], MODELS[model_name]['checkpoint'], device='cpu')

    def run(self):
        while True:
            frame = self.task_queue.get()
            start = time.time()
            
            init_default_scope('mmdet')
            det_result = inference_detector(self.detector, frame)
            init_default_scope('mmpose')
            
            pred_instances = det_result.pred_instances
            bboxes = pred_instances.bboxes[pred_instances.scores > 0.5].cpu().numpy()
            pose_results = inference_topdown(self.pose_model, frame, bboxes)
            
            latency = (time.time() - start) * 1000
            self.result_queue.put((pose_results, bboxes, latency))
            self.task_queue.task_done()

class PoseApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RTMPose Multi-Person Control")
        self.task_queue = Queue(maxsize=1)
        self.result_queue = Queue(maxsize=1)
        self.cap = cv2.VideoCapture(VIDEO_PATH)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        
        layout = QVBoxLayout()
        top_ctrl = QHBoxLayout()
        self.chk_bbox = QCheckBox("Show Detection Box"); self.chk_bbox.setChecked(True)
        self.chk_pose = QCheckBox("Show Pose"); self.chk_pose.setChecked(True)
        top_ctrl.addWidget(self.chk_bbox); top_ctrl.addWidget(self.chk_pose)
        layout.addLayout(top_ctrl)
        
        self.combo = QComboBox()
        self.combo.addItems(MODELS.keys())
        self.combo.currentTextChanged.connect(self.load_model)
        layout.addWidget(QLabel("Select Model:")); layout.addWidget(self.combo)
        self.video_label = QLabel("Initializing..."); layout.addWidget(self.video_label)
        
        ctrl_layout = QHBoxLayout()
        self.btn_play = QPushButton("Play/Pause"); self.btn_play.clicked.connect(self.toggle_play)
        self.btn_reset = QPushButton("Start Over"); self.btn_reset.clicked.connect(self.reset_video)
        ctrl_layout.addWidget(self.btn_play); ctrl_layout.addWidget(self.btn_reset)
        layout.addLayout(ctrl_layout)
        
        self.progress_bar = QSlider(Qt.Orientation.Horizontal); self.progress_bar.setRange(0, self.total_frames)
        layout.addWidget(self.progress_bar)
        self.time_label = QLabel("00:00 / 00:00"); layout.addWidget(self.time_label)
        self.stats_label = QLabel("Stats: ..."); layout.addWidget(self.stats_label)
        self.setLayout(layout)
        
        self.is_playing = True
        self.load_model(self.combo.currentText())
        self.prev_time = time.time()
        self.timer = QTimer(); self.timer.timeout.connect(self.process_frame); self.timer.start(10)
        self.last_frame_time = time.time()

    def load_model(self, name):
        self.worker = InferenceWorker(self.task_queue, self.result_queue, name)
        self.worker.start()
        self.visualizer = PoseLocalVisualizer(radius=2, line_width=1)
        # Load meta from temp model
        temp_model = init_model(MODELS[name]['config'], MODELS[name]['checkpoint'], device='cpu')
        self.visualizer.set_dataset_meta(temp_model.dataset_meta)

    def toggle_play(self): self.is_playing = not self.is_playing
    def reset_video(self): self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def process_frame(self):
        if not self.is_playing: return
        ret, frame = self.cap.read()
        if not ret: return
        
        if self.task_queue.empty(): self.task_queue.put(frame)
        
        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        try:
            pose_results, bboxes, latency = self.result_queue.get_nowait()
            data_to_draw = merge_data_samples(pose_results) if (self.chk_pose.isChecked() and len(pose_results) > 0) else None
            
            self.visualizer.set_image(frame)
            vis = self.visualizer.add_datasample('res', frame, data_sample=data_to_draw, draw_gt=False,
                                                 draw_bbox=self.chk_bbox.isChecked() if data_to_draw else self.chk_bbox.isChecked(),
                                                 draw_pred=self.chk_pose.isChecked(), show=False)
            
            if self.chk_bbox.isChecked() and not self.chk_pose.isChecked():
                for bbox in bboxes: cv2.rectangle(vis, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            
            vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            qt_img = QImage(vis.data, vis.shape[1], vis.shape[0], vis.shape[1]*3, QImage.Format.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_img).scaled(800, 600, Qt.AspectRatioMode.KeepAspectRatio))
            
            fps_val = 1.0 / (time.time() - self.prev_time + 0.0001)
            self.prev_time = time.time()
            self.progress_bar.setValue(current_frame)
            self.time_label.setText(f"{int(current_frame/(self.fps*60)):02d}:{int(current_frame/self.fps)%60:02d} / {int(self.total_frames/(self.fps*60)):02d}:{int(self.total_frames/self.fps)%60:02d}")
            self.stats_label.setText(f"People: {len(pose_results)} | Latency: {latency:.1f}ms | FPS: {fps_val:.1f}")
        except Empty: pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PoseApp()
    window.show()
    sys.exit(app.exec())