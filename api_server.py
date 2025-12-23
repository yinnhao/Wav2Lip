import os
import sys
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchaudio
import onnxruntime as ort
from transformers import HubertModel, Wav2Vec2FeatureExtractor
from tqdm import tqdm
import subprocess
import platform
import uuid
import shutil
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
import uvicorn
from contextlib import asynccontextmanager

# -----------------------------------------------------------------------------
# 环境配置与依赖
# -----------------------------------------------------------------------------
# 确保 Wav2Lip 在路径中 (假设当前目录下有 Wav2Lip 文件夹)
sys.path.append('Wav2Lip')
try:
    import face_detection
except ImportError:
    print("Error: Could not import 'face_detection'. Make sure 'Wav2Lip' folder is present.")
    sys.exit(1)

# -----------------------------------------------------------------------------
# 全局配置与状态
# -----------------------------------------------------------------------------
class ServiceConfig:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.img_size = 512
        self.checkpoint_path = None
        self.hubert_path = None
        
        # 默认推理参数
        self.pads = [0, 10, 0, 0]
        self.face_det_batch_size = 16
        self.wav2lip_batch_size = 8
        self.resize_factor = 1
        self.crop = [0, -1, 0, -1]
        self.box = [-1, -1, -1, -1]
        self.rotate = False
        self.nosmooth = False
        self.static = False
        self.fps = 25.0

service_config = ServiceConfig()
models = {}

# -----------------------------------------------------------------------------
# 核心逻辑函数 (复用自 infer_v3.py)
# -----------------------------------------------------------------------------

def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes

def face_detect(images, batch_size=16, pads=[0, 10, 0, 0], nosmooth=False):
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
                                            flip_input=False, device=service_config.device)
    
    while 1:
        predictions = []
        try:
            for i in tqdm(range(0, len(images), batch_size), desc="Face Detection"):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1: 
                raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    last_valid_rect = None
    pady1, pady2, padx1, padx2 = pads
    for rect, image in zip(predictions, images):
        if rect is None:
            if last_valid_rect is not None:
                rect = last_valid_rect
            else:
                h, w = image.shape[:2]
                rect = [w//4, h//4, w*3//4, h*3//4] # Fallback to center

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
        
        results.append([x1, y1, x2, y2])
        last_valid_rect = rect

    boxes = np.array(results)
    if not nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    del detector
    return results 

def load_hubert_and_processor(model_path):
    model = HubertModel.from_pretrained(model_path).to(service_config.device).eval()
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    return model, processor

def extract_hubert_features(audio_path):
    # 使用全局加载的模型
    model = models.get('hubert_model')
    processor = models.get('hubert_processor')
    
    wav, sr = torchaudio.load(audio_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        wav = resampler(wav)
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)

    wav_np = wav.squeeze(0).cpu().numpy()
    processed = processor(wav_np, sampling_rate=16000, return_tensors="pt").input_values.to(service_config.device)

    kernel, stride = 400, 320
    clip_length = stride * 1000 
    expected_T = (processed.shape[1] - (kernel - stride)) // stride

    with torch.no_grad():
        if processed.shape[1] <= clip_length:
            hidden_states = model(processed).last_hidden_state
        else:
            res_lst = []
            num_iter = processed.shape[1] // clip_length
            for i in range(num_iter):
                start_idx = i * clip_length
                end_idx = start_idx + (clip_length - stride + kernel)
                chunk = processed[:, start_idx:end_idx]
                if chunk.shape[1] >= kernel:
                    res_lst.append(model(chunk).last_hidden_state[0])
            remaining_start = num_iter * clip_length
            remaining = processed[:, remaining_start:]
            if remaining.shape[1] >= kernel:
                res_lst.append(model(remaining).last_hidden_state[0])
            if len(res_lst) > 0:
                ret = torch.cat(res_lst, dim=0)
                if ret.shape[0] >= expected_T:
                    ret = ret[:expected_T]
                else:
                    ret = F.pad(ret, (0, 0, 0, expected_T - ret.shape[0]))
                hidden_states = ret.unsqueeze(0)
            else:
                hidden_states = model(wav).last_hidden_state

    hidden_states = hidden_states.permute(0, 2, 1)  # (B, 1024, T)
    return hidden_states.cpu().numpy()

def create_rep_chunks(audio_tensor, num_frames, fps, static=False):
    rep_step_size = 10
    rep_idx_multiplier = 2.0 # Fixed for 25fps online service

    chunks, frame_indices = [], []
    total = audio_tensor.shape[-1]
    i = 0
    while True:
        start_idx = int(i * rep_idx_multiplier)
        if start_idx + rep_step_size > total:
            if start_idx > total - 1:
                break
            chunk = audio_tensor[0, :, total - rep_step_size:]
        else:
            chunk = audio_tensor[0, :, start_idx: start_idx + rep_step_size]

        if chunk.shape[1] < rep_step_size:
            pad = rep_step_size - chunk.shape[1]
            chunk = np.pad(chunk, ((0, 0), (0, pad)), mode='edge')

        chunks.append(chunk)
        
        if static:
            idx = 0
        else:
            idx = i % num_frames if i // num_frames % 2 == 0 else num_frames - 1 - i % num_frames
        frame_indices.append(idx)
        i += 1

    return chunks, frame_indices

def datagen(frames, face_det_results, rep_chunks, frame_indices, img_size, batch_size):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    for chunk, idx in zip(rep_chunks, frame_indices):
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (img_size, img_size))

        img_batch.append(face)
        mel_batch.append(chunk)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= batch_size:
            img_batch_np = np.asarray(img_batch)
            mel_batch_np = np.asarray(mel_batch)

            img_concat = np.concatenate((img_batch_np, img_batch_np), axis=3)
            img_input = np.transpose(img_concat, (0, 3, 1, 2)) / 255.
            mel_input = np.transpose(mel_batch_np, (0, 2, 1))

            yield img_input, mel_input, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch_np = np.asarray(img_batch)
        mel_batch_np = np.asarray(mel_batch)
        img_concat = np.concatenate((img_batch_np, img_batch_np), axis=3)
        img_input = np.transpose(img_concat, (0, 3, 1, 2)) / 255.
        mel_input = np.transpose(mel_batch_np, (0, 2, 1))
        yield img_input, mel_input, frame_batch, coords_batch

def load_onnx_model(path):
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if service_config.device == 'cuda' else ['CPUExecutionProvider']
    return ort.InferenceSession(path, providers=providers)

# -----------------------------------------------------------------------------
# FastAPI 服务
# -----------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load Models
    print(f"Loading HuBERT from {service_config.hubert_path}...")
    models['hubert_model'], models['hubert_processor'] = load_hubert_and_processor(service_config.hubert_path)
    
    print(f"Loading ONNX model from {service_config.checkpoint_path}...")
    models['onnx_session'] = load_onnx_model(service_config.checkpoint_path)
    print("Models loaded successfully.")
    
    yield
    
    # Shutdown
    models.clear()

app = FastAPI(lifespan=lifespan)

@app.post("/wav2lip")
async def inference(
    face_file: UploadFile = File(...),
    audio_file: UploadFile = File(...),
    static: bool = Form(False),
    bbox: str = Form("-1,-1,-1,-1", description="Bounding box x1,y1,x2,y2"),
):
    """
    服务端推理接口
    - face_file: 视频或图片文件
    - audio_file: 音频文件 (.wav, .mp3 等)
    - static: 是否只使用第一帧 (图片模式)
    - bbox: 手动指定人脸框 (逗号分隔整数)
    """
    
    # 创建临时目录
    req_id = str(uuid.uuid4())
    temp_dir = os.path.join("temp", req_id)
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # 保存上传文件
        face_path = os.path.join(temp_dir, face_file.filename)
        audio_path = os.path.join(temp_dir, audio_file.filename)
        out_path = os.path.join(temp_dir, "result_voice.mp4")
        temp_avi = os.path.join(temp_dir, "result.avi")
        temp_wav = os.path.join(temp_dir, "temp.wav")
        
        with open(face_path, "wb") as f:
            shutil.copyfileobj(face_file.file, f)
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(audio_file.file, f)
            
        # 解析参数
        try:
            box = [int(x) for x in bbox.split(',')]
            if len(box) != 4: box = [-1, -1, -1, -1]
        except:
            box = [-1, -1, -1, -1]
            
        # 1. 预处理输入 (图片/视频)
        if face_path.lower().endswith(('.jpg', '.png', '.jpeg')):
            static = True # 图片强制 static
            full_frames = [cv2.imread(face_path)]
            fps = service_config.fps
        else:
            video_stream = cv2.VideoCapture(face_path)
            fps = video_stream.get(cv2.CAP_PROP_FPS)
            full_frames = []
            while 1:
                ret, frame = video_stream.read()
                if not ret: break
                
                if service_config.resize_factor > 1:
                    frame = cv2.resize(frame, (frame.shape[1]//service_config.resize_factor, frame.shape[0]//service_config.resize_factor))

                if service_config.rotate:
                    try:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    except AttributeError:
                        # Fallback for some older opencv bindings
                        frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

                y1, y2, x1, x2 = service_config.crop
                if x2 == -1: x2 = frame.shape[1]
                if y2 == -1: y2 = frame.shape[0]

                frame = frame[y1:y2, x1:x2]
                full_frames.append(frame)
            video_stream.release()

        if len(full_frames) == 0:
            raise HTTPException(status_code=400, detail="Could not read frames from face file.")

        # 2. 音频预处理
        # 转换音频为 wav 并提取特征
        command = f'ffmpeg -y -i "{audio_path}" -strict -2 "{temp_wav}" -loglevel error'
        subprocess.call(command, shell=True)
        
        audio_tensor = extract_hubert_features(temp_wav)
        
        # 3. 人脸检测
        if box[0] == -1:
            if not static:
                face_det_results = face_detect(full_frames, service_config.face_det_batch_size, service_config.pads, service_config.nosmooth)
            else:
                face_det_results = face_detect([full_frames[0]], service_config.face_det_batch_size, service_config.pads, service_config.nosmooth)
        else:
            y1, y2, x1, x2 = box
            # 简单的边界检查
            face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in full_frames]

        # 4. 准备数据生成器
        rep_chunks, frame_indices = create_rep_chunks(audio_tensor, len(full_frames), fps, static=static)
        
        gen = datagen(full_frames.copy(), face_det_results, rep_chunks, frame_indices, service_config.img_size, service_config.wav2lip_batch_size)

        # 5. 推理循环
        frame_h, frame_w = full_frames[0].shape[:-1]
        out_writer = cv2.VideoWriter(temp_avi, cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))
        
        onnx_sess = models['onnx_session']
        onnx_inputs = onnx_sess.get_inputs()
        input_names = [inp.name for inp in onnx_inputs]
        dtype = np.float16 if any('float16' in inp.type for inp in onnx_inputs) else np.float32

        for img_batch, mel_batch, frames, coords in tqdm(gen, desc="Inference"):
            img_batch = img_batch.astype(dtype)
            mel_batch = mel_batch.astype(dtype)

            bsz = img_batch.shape[0]
            hn = np.zeros((2, bsz, 512), dtype=dtype)
            cn = np.zeros((2, bsz, 512), dtype=dtype)

            inputs = {}
            for name in input_names:
                if 'audio' in name or 'mel' in name:
                    inputs[name] = mel_batch
                elif 'face' in name or 'img' in name:
                    inputs[name] = img_batch
                elif 'hn' in name:
                    inputs[name] = hn
                elif 'cn' in name:
                    inputs[name] = cn

            try:
                pred = onnx_sess.run(['pred_face'], inputs)[0]
            except Exception:
                pred = onnx_sess.run(None, inputs)[0]

            pred = pred.astype(np.float32).transpose(0, 2, 3, 1) * 255.

            for p, f, c in zip(pred, frames, coords):
                y1, y2, x1, x2 = c
                # 安全 Resize
                if x2-x1 <= 0 or y2-y1 <= 0: continue
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                f[y1:y2, x1:x2] = p
                out_writer.write(f)

        out_writer.release()

        # 6. 合成最终视频
        command = f'ffmpeg -y -i "{temp_wav}" -i "{temp_avi}" -strict -2 -q:v 1 "{out_path}" -loglevel error'
        subprocess.call(command, shell=platform.system() != 'Windows')
        
        if not os.path.exists(out_path):
             raise HTTPException(status_code=500, detail="FFmpeg failed to merge video.")

        # 返回视频文件
        return FileResponse(out_path, media_type="video/mp4", filename="result.mp4")

    except Exception as e:
        print(f"Error: {e}")
        # Clean up in case of error (optional, or rely on periodic cleanup)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000, help='Service port')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to Wav2Lip ONNX checkpoint')
    parser.add_argument('--hubert_path', type=str, required=True, help='Path to HuBERT model')
    args = parser.parse_args()
    
    service_config.checkpoint_path = args.checkpoint_path
    service_config.hubert_path = args.hubert_path
    
    print(f"Starting server on port {args.port}...")
    print(f"Device: {service_config.device}")
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)

