import sys
import cv2
import time
import threading
from queue import Queue, Empty
from mmengine.registry import TRANSFORMS, init_default_scope
from mmdet.datasets.transforms import PackDetInputs
from mmpose.datasets.transforms import PackPoseInputs

TRANSFORMS.register_module(module=PackDetInputs)
TRANSFORMS.register_module(module=PackPoseInputs)

from mmdet.apis import inference_detector, init_detector
from mmpose.apis import inference_topdown, init_model
from mmpose.visualization import PoseLocalVisualizer
from mmpose.structures import merge_data_samples

from PyQt6.QtWidgets import (QApplication, QLabel, QVBoxLayout, QWidget, QRadioButton, 
                             QPushButton, QSlider, QHBoxLayout, QCheckBox, QButtonGroup)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap

DET_CONFIG = 'detection/rtmdet_nano.py'
DET_CHECKPOINT = 'detection/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth'
VIDEO_PATH = '60min_mb720p.mp4'

MODELS = {
    "RTMPose-T": {"config": "models/rtmpose-t_config.py", "checkpoint": "models/rtmpose-t_simcc-ucoco_dw-ucoco_270e-256x192-dcf277bf_20230728.pth"},
    "RTMPose-S": {"config": "models/rtmpose-s_config.py", "checkpoint": "models/rtmpose-s_simcc-ucoco_dw-ucoco_270e-256x192-3fd922c8_20230728.pth"},
    "RTMPose-M": {"config": "models/rtmpose-m_config.py", "checkpoint": "models/rtmpose-m_simcc-ucoco_dw-ucoco_270e-256x192-c8b76419_20230728.pth"}
}

class InferenceWorker(threading.Thread):
    def __init__(self, task_queue, result_queue, model_name, detector):
        super().__init__()
        self.task_queue, self.result_queue = task_queue, result_queue
        self.detector = detector # Reuse existing detector
        self.daemon = True
        init_default_scope('mmpose')
        self.pose_model = init_model(MODELS[model_name]['config'], MODELS[model_name]['checkpoint'], device='cpu')

    def run(self):
        while True:
            frame = self.task_queue.get()
            if frame is None: break
            start = time.time()
            init_default_scope('mmdet')
            det_result = inference_detector(self.detector, frame)
            init_default_scope('mmpose')
            bboxes = det_result.pred_instances.bboxes[det_result.pred_instances.scores > 0.5].cpu().numpy()
            pose_results = inference_topdown(self.pose_model, frame, bboxes)
            self.result_queue.put((pose_results, bboxes, (time.time() - start) * 1000))
            self.task_queue.task_done()

class PoseApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RTMPose Real-Time System")
        # Global detector
        self.detector = init_detector(DET_CONFIG, DET_CHECKPOINT, device='cpu')
        self.task_queue, self.result_queue = Queue(maxsize=1), Queue(maxsize=1)
        self.cap = cv2.VideoCapture(VIDEO_PATH)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_sec = self.total_frames / self.fps
        self.playback_speed, self.start_time, self.is_playing = 1.0, time.time(), True
        self.last_frame = None

        layout = QVBoxLayout()
        title = QLabel("RTMPose Real-Time Analysis System"); title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;"); layout.addWidget(title)
        
        model_layout = QHBoxLayout(); model_layout.addWidget(QLabel("Select Model:")); self.button_group = QButtonGroup()
        for name in MODELS.keys():
            rb = QRadioButton(name)
            if name == "RTMPose-T": rb.setChecked(True)
            rb.toggled.connect(lambda checked, n=name: self.load_model(n) if checked else None)
            model_layout.addWidget(rb); self.button_group.addButton(rb)
        layout.addLayout(model_layout)

        toggle_layout = QHBoxLayout()
        self.chk_bbox = QCheckBox("Show Detection Box"); self.chk_bbox.setChecked(True)
        self.chk_pose = QCheckBox("Show Pose"); self.chk_pose.setChecked(True)
        toggle_layout.addWidget(self.chk_bbox); toggle_layout.addWidget(self.chk_pose); layout.addLayout(toggle_layout)
        
        self.video_label = QLabel("Initializing..."); layout.addWidget(self.video_label)
        
        ctrl_layout = QHBoxLayout()
        self.btn_play = QPushButton("Play/Pause"); self.btn_play.clicked.connect(lambda: setattr(self, 'is_playing', not self.is_playing))
        self.btn_reset = QPushButton("Start Over"); self.btn_reset.clicked.connect(lambda: [self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0), setattr(self, 'start_time', time.time())])
        ctrl_layout.addWidget(self.btn_play); ctrl_layout.addWidget(self.btn_reset); layout.addLayout(ctrl_layout)
        
        self.speed_slider = QSlider(Qt.Orientation.Horizontal); self.speed_slider.setRange(25, 300); self.speed_slider.setValue(100)
        self.speed_label = QLabel("Speed: 1.0x")
        self.speed_slider.valueChanged.connect(lambda v: [setattr(self, 'playback_speed', v/100), self.speed_label.setText(f"Speed: {v/100:.2f}x")])
        layout.addWidget(self.speed_label); layout.addWidget(self.speed_slider)
        
        self.progress_bar = QSlider(Qt.Orientation.Horizontal); self.progress_bar.setRange(0, self.total_frames)
        layout.addWidget(self.progress_bar); self.time_label = QLabel("00:00 / 00:00"); layout.addWidget(self.time_label)
        self.stats_label = QLabel("Stats: ..."); layout.addWidget(self.stats_label)
        
        self.setLayout(layout)
        self.load_model("RTMPose-T")
        self.timer = QTimer(); self.timer.timeout.connect(self.process_frame); self.timer.start(15)

    def load_model(self, name):
        self.worker = InferenceWorker(self.task_queue, self.result_queue, name, self.detector); self.worker.start()
        self.visualizer = PoseLocalVisualizer(radius=2, line_width=1)
        temp_model = init_model(MODELS[name]['config'], MODELS[name]['checkpoint'], device='cpu')
        self.visualizer.set_dataset_meta(temp_model.dataset_meta)

    def format_time(self, s): return f"{int(s//60):02d}:{int(s%60):02d}"

    def process_frame(self):
        if not self.is_playing: return
        elapsed_video_time = (time.time() - self.start_time) * self.playback_speed
        target_frame = int(elapsed_video_time * self.fps)
        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

        if target_frame > current_frame:
            ret, frame = self.cap.read()
            if not ret: return
            self.last_frame = frame.copy()
            if self.task_queue.empty(): self.task_queue.put(frame)
        
        try:
            pose_results, bboxes, latency = self.result_queue.get_nowait()
            data_to_draw = merge_data_samples(pose_results) if (self.chk_pose.isChecked() and len(pose_results) > 0) else None
            self.visualizer.set_image(self.last_frame)
            vis = self.visualizer.add_datasample('res', self.last_frame, data_sample=data_to_draw, draw_gt=False,
                                                 draw_bbox=self.chk_bbox.isChecked(), draw_pred=self.chk_pose.isChecked(), show=False)
            vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            qt_img = QImage(vis.data, vis.shape[1], vis.shape[0], vis.shape[1]*3, QImage.Format.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_img).scaled(800, 600, Qt.AspectRatioMode.KeepAspectRatio))
            self.stats_label.setText(f"People: {len(pose_results)} | Latency: {latency:.1f}ms | FPS: {self.fps * self.playback_speed:.1f}")
            self.progress_bar.setValue(current_frame)
            self.time_label.setText(f"{self.format_time(elapsed_video_time)} / {self.format_time(self.duration_sec)}")
        except Empty: pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PoseApp()
    window.show()
    sys.exit(app.exec())