import sys
import cv2
import time
from PyQt6.QtWidgets import (QApplication, QLabel, QVBoxLayout, QWidget, QComboBox, 
                             QPushButton, QSlider, QHBoxLayout)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap

# --- FORCE REGISTRY UNIFICATION ---
from mmengine.registry import TRANSFORMS
from mmdet.datasets.transforms import PackDetInputs
from mmpose.datasets.transforms import PackPoseInputs
from mmengine.registry import init_default_scope

TRANSFORMS.register_module(module=PackDetInputs)
TRANSFORMS.register_module(module=PackPoseInputs)

from mmdet.apis import inference_detector, init_detector
from mmpose.apis import inference_topdown, init_model
from mmpose.visualization import PoseLocalVisualizer
from mmpose.structures import merge_data_samples

# --- CONFIG ---
DET_CONFIG = 'detection/rtmdet_nano.py'
DET_CHECKPOINT = 'detection/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth'
VIDEO_PATH = '60min_mb720p.mp4'

MODELS = {
    "RTMPose-T": {"config": "models/rtmpose-t_config.py", "checkpoint": "models/rtmpose-t_simcc-ucoco_dw-ucoco_270e-256x192-dcf277bf_20230728.pth"},
    "RTMPose-S": {"config": "models/rtmpose-s_config.py", "checkpoint": "models/rtmpose-s_simcc-ucoco_dw-ucoco_270e-256x192-3fd922c8_20230728.pth"},
    "RTMPose-M": {"config": "models/rtmpose-m_config.py", "checkpoint": "models/rtmpose-m_simcc-ucoco_dw-ucoco_270e-256x192-c8b76419_20230728.pth"}
}

class PoseApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RTMPose Multi-Person Control")
        
        # Detector
        self.detector = init_detector(DET_CONFIG, DET_CHECKPOINT, device='cpu')
        self.cap = cv2.VideoCapture(VIDEO_PATH)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        
        # UI Layout
        layout = QVBoxLayout()
        self.combo = QComboBox()
        self.combo.addItems(MODELS.keys())
        self.combo.currentTextChanged.connect(self.load_model)
        layout.addWidget(QLabel("Select Model:")); layout.addWidget(self.combo)
        
        self.video_label = QLabel("Loading..."); layout.addWidget(self.video_label)
        
        # Controls
        ctrl_layout = QHBoxLayout()
        self.btn_play = QPushButton("Play/Pause"); self.btn_play.clicked.connect(self.toggle_play)
        self.btn_reset = QPushButton("Start Over"); self.btn_reset.clicked.connect(self.reset_video)
        ctrl_layout.addWidget(self.btn_play); ctrl_layout.addWidget(self.btn_reset)
        layout.addLayout(ctrl_layout)
        
        self.progress_bar = QSlider(Qt.Orientation.Horizontal); self.progress_bar.setRange(0, self.total_frames)
        layout.addWidget(self.progress_bar)
        self.time_label = QLabel("00:00 / 00:00"); layout.addWidget(self.time_label)
        
        self.stats_label = QLabel("People: 0 | Latency: 0ms"); layout.addWidget(self.stats_label)
        self.setLayout(layout)
        
        self.is_playing = True
        self.load_model(self.combo.currentText())
        self.timer = QTimer(); self.timer.timeout.connect(self.process_frame); self.timer.start(1)

    def toggle_play(self): self.is_playing = not self.is_playing
    def reset_video(self): self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def load_model(self, name):
        init_default_scope('mmpose')
        self.pose_model = init_model(MODELS[name]['config'], MODELS[name]['checkpoint'], device='cpu')
        self.visualizer = PoseLocalVisualizer(radius=2, line_width=1)
        self.visualizer.set_dataset_meta(self.pose_model.dataset_meta)

    def process_frame(self):
        if not self.is_playing: return
        ret, frame = self.cap.read()
        if not ret: self.timer.stop(); return
        
        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        start = time.time()

        init_default_scope('mmdet'); det_result = inference_detector(self.detector, frame); init_default_scope('mmpose')
        bboxes = det_result.pred_instances.bboxes[det_result.pred_instances.scores > 0.5].cpu().numpy()
        pose_results = inference_topdown(self.pose_model, frame, bboxes)
        
        merged = merge_data_samples(pose_results) if len(pose_results) > 0 else frame
        vis = self.visualizer.add_datasample('res', frame, merged, draw_gt=False, show=False)
        
        # Display
        vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        self.video_label.setPixmap(QPixmap.fromImage(QImage(vis.data, vis.shape[1], vis.shape[0], vis.shape[1]*3, QImage.Format.Format_RGB888)).scaled(800, 600, Qt.AspectRatioMode.KeepAspectRatio))
        
        # Update Controls
        self.progress_bar.setValue(current_frame)
        def format_time(f): return f"{int(f/self.fps)//60:02d}:{int(f/self.fps)%60:02d}"
        self.time_label.setText(f"{format_time(current_frame)} / {format_time(self.total_frames)}")
        self.stats_label.setText(f"People Detected: {len(pose_results)} | Processing: {(time.time()-start)*1000:.1f}ms")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PoseApp()
    window.show()
    sys.exit(app.exec())