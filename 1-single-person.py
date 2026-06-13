import sys
import cv2
import time

# --- FORCE REGISTRY UNIFICATION ---
from mmengine.registry import TRANSFORMS
from mmdet.datasets.transforms import PackDetInputs
from mmpose.datasets.transforms import PackPoseInputs

# Register mmdet transforms into mmpose registry so it can find them
TRANSFORMS.register_module(module=PackDetInputs)
TRANSFORMS.register_module(module=PackPoseInputs)

from mmdet.apis import inference_detector, init_detector
from mmpose.apis import inference_topdown, init_model
from mmpose.visualization import PoseLocalVisualizer

# ... rest of your imports (PyQt, etc)

# ... rest of the imports ...
from PyQt6.QtWidgets import (QApplication, QLabel, QVBoxLayout, QWidget, QComboBox)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap


# --- CONFIG ---
DET_CONFIG = 'D:/uni/term 8/kar amoozi/task2\detection/rtmdet_nano.py'
DET_CHECKPOINT = 'detection/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth'
VIDEO_PATH = '60min_mb720p.mp4'

# Model Mapping
MODELS = {
    "RTMPose-T": {
        "config": "models/rtmpose-t_config.py",
        "checkpoint": "models/rtmpose-t_simcc-ucoco_dw-ucoco_270e-256x192-dcf277bf_20230728.pth"
    },
    "RTMPose-S": {
        "config": "models/rtmpose-s_config.py",
        "checkpoint": "models/rtmpose-s_simcc-ucoco_dw-ucoco_270e-256x192-3fd922c8_20230728.pth"
    },
    "RTMPose-M": {
        "config": "models/rtmpose-m_config.py",
        "checkpoint": "models/rtmpose-m_simcc-ucoco_dw-ucoco_270e-256x192-c8b76419_20230728.pth"
    }
}

class PoseApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RTMPose WholeBody Detector")
        
        # Initialize Detector
        self.detector = init_detector(DET_CONFIG, DET_CHECKPOINT, device='cpu')
        
        # UI Setup
        layout = QVBoxLayout()
        self.combo = QComboBox()
        self.combo.addItems(MODELS.keys())
        self.combo.currentTextChanged.connect(self.load_model)
        layout.addWidget(QLabel("Select Model:"))
        layout.addWidget(self.combo)
        
        self.label = QLabel("Initializing...")
        layout.addWidget(self.label)
        self.setLayout(layout)
        
        # Initialize first model
        self.load_model(self.combo.currentText())
        
        self.cap = cv2.VideoCapture(VIDEO_PATH)
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)
        self.timer.start(1)

    def load_model(self, name):
        print(f"Loading {name}...")
        self.pose_model = init_model(
            MODELS[name]['config'], 
            MODELS[name]['checkpoint'], 
            device='cpu'
        )
        self.visualizer = PoseLocalVisualizer(radius=2, line_width=1)
        self.visualizer.set_dataset_meta(self.pose_model.dataset_meta)

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret: 
            self.timer.stop()
            return
        
        start = time.time()

        # 1. Detection 
        # We manually bypass the complex pipeline build if it keeps failing
        from mmengine.registry import init_default_scope
        init_default_scope('mmdet') 
        
        det_result = inference_detector(self.detector, frame)
        
        # Reset back to mmpose for the pose model
        init_default_scope('mmpose')
        
        pred_instances = det_result.pred_instances
        bboxes = pred_instances.bboxes[pred_instances.scores > 0.5].cpu().numpy()
        
        # 2. Pose Estimation
        pose_results = inference_topdown(self.pose_model, frame, bboxes)
        
        # 3. Visualization
        vis_frame = self.visualizer.add_datasample(
            'result', frame, pose_results[0], 
            draw_gt=False, draw_bbox=True, show=False
        )
        
        # 4. Display
        vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = vis_frame.shape
        qt_img = QImage(vis_frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qt_img).scaled(800, 600, Qt.AspectRatioMode.KeepAspectRatio))
        
        elapsed = (time.time() - start) * 1000
        self.setWindowTitle(f"Model: {self.combo.currentText()} | Processing: {elapsed:.1f}ms")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PoseApp()
    window.show()
    sys.exit(app.exec())