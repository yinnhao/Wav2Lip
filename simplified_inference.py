#!/usr/bin/env python3
# coding=utf-8
import os
import sys
import argparse
import json
import subprocess
import traceback
import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image
import io
import shutil
import torch
import torchaudio
try:
    from transformers import HubertModel
except ImportError:
    HubertModel = None

# Constants from the analysis
IMG_SIZE = 512 # Default, but model input shape might differ
WAV2LIP_BATCH_SIZE = 8
MEL_STEP_SIZE = 10 # Referenced from get_lips_result.py (rep_step_size)
REP_IDX_MULTIPLIER = 2 # Referenced from get_lips_result.py

class HubertFeatureExtractor:
    def __init__(self, model_path, device='cpu'):
        if HubertModel is None:
            raise ImportError("transformers is required for HuBERT. Please pip install transformers.")
        self.device = device
        self.hubert_model = HubertModel.from_pretrained(model_path).to(self.device).eval()

    def infer(self, input_values):
        """
        Infer features from audio input (Batch, T)
        Logic ported from hubert_audio/models/hubert.py
        """
        input_values = torch.tensor(input_values).to(self.device)
        if input_values.ndim == 1:
            input_values = input_values.unsqueeze(0)

        with torch.no_grad():
            kernel, stride = 400, 320
            clip_length = stride * 1000
            
            # Simple case
            if input_values.shape[1] <= clip_length:
                hidden_states = self.hubert_model(input_values).last_hidden_state
            else:
                # Sliding window logic
                num_iter = input_values.shape[1] // clip_length
                expected_T = (input_values.shape[1] - (kernel - stride)) // stride
                
                slices = []
                for i in range(num_iter):
                    start_idx = i * clip_length
                    end_idx = start_idx + (clip_length - stride + kernel)
                    slices.append((start_idx, end_idx))

                remaining_start = num_iter * clip_length
                # Logic from original code: check if remaining is long enough
                remaining = input_values[:, remaining_start:]
                if remaining.shape[1] >= kernel:
                    slices.append((remaining_start, input_values.shape[1]))

                res_lst = []
                for start, end in slices:
                    chunk = input_values[:, start:end]
                    # Ensure chunk is valid
                    if chunk.shape[1] < kernel:
                         continue
                    clip_hidden_states = self.hubert_model(chunk).last_hidden_state
                    res_lst.append(clip_hidden_states[0])
                
                if not res_lst:
                     # Fallback if no slices? Should not happen if logic is correct
                     hidden_states = self.hubert_model(input_values).last_hidden_state
                else:
                    ret = torch.cat(res_lst, dim=0) # (Total_T, 1024)
                    
                    # Pad or trim to expected_T
                    if ret.shape[0] >= expected_T:
                        ret = ret[:expected_T]
                    else:
                        ret = torch.nn.functional.pad(ret, (0, 0, 0, expected_T - ret.shape[0]))
                    
                    hidden_states = ret.unsqueeze(0) # (1, T, 1024)
        
        return hidden_states.cpu().numpy()

class Wav2LipInference:
    def __init__(self, model_path, hubert_path=None, device='cpu'):
        self.model_path = model_path
        self.hubert_path = hubert_path
        self.device = device
        self.img_size = IMG_SIZE # Initialize with default
        
        # Load Wav2Lip Model
        self.session = self.load_onnx_model(model_path)
        
        # Load HuBERT Model if provided
        self.hubert_extractor = None
        if hubert_path and os.path.exists(hubert_path):
            try:
                print(f"Loading HuBERT model from {hubert_path}...")
                self.hubert_extractor = HubertFeatureExtractor(hubert_path, device)
            except Exception as e:
                print(f"Failed to load HuBERT model: {e}")
        
        # Check model input shapes to adjust IMG_SIZE
        self.check_model_input()

    def load_onnx_model(self, checkpoint_path):
        """Load ONNX model"""
        if self.device == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        try:
            print(f"Loading model from {checkpoint_path}...")
            session = ort.InferenceSession(checkpoint_path, providers=providers)
            return session
        except Exception as e:
            print(f"Error loading ONNX model: {e}")
            sys.exit(1)

    def check_model_input(self):
        """Check ONNX model input shapes"""
        inputs = self.session.get_inputs()
        for inp in inputs:
            # Expected inputs: audio, face, in_hn, in_cn
            print(f"Model Input: {inp.name}, Shape: {inp.shape}, Type: {inp.type}")
            if 'face' in inp.name and len(inp.shape) == 4:
                # Shape: [Batch, Channels, H, W]
                h, w = inp.shape[2], inp.shape[3]
                if isinstance(h, int) and isinstance(w, int):
                    self.img_size = h
                    print(f"Set IMG_SIZE to {self.img_size} based on model input.")

    def get_audio_features(self, audio_path):
        """
        Extract audio features.
        If audio_path is .npy: load directly.
        If audio_path is .wav: extract using HuBERT if available.
        """
        if audio_path.endswith('.npy'):
            print(f"Loading pre-computed features from {audio_path}")
            feat = np.load(audio_path)
            # Ensure shape is (1, 1024, T)
            if feat.ndim == 2:
                # Assume (1024, T) -> (1, 1024, T)
                feat = feat[np.newaxis, :, :]
            return feat
        else:
            if self.hubert_extractor:
                print(f"Extracting features from {audio_path} using HuBERT...")
                try:
                    # Load audio using torchaudio
                    wav, sr = torchaudio.load(audio_path)
                    if sr != 16000:
                        resampler = torchaudio.transforms.Resample(sr, 16000)
                        wav = resampler(wav)
                    
                    # Mix to mono if needed
                    if wav.shape[0] > 1:
                        wav = torch.mean(wav, dim=0, keepdim=True)
                        
                    input_values = wav.numpy()
                    
                    # Extract features: (1, T, 1024)
                    hidden_states = self.hubert_extractor.infer(input_values)
                    
                    # Transpose to (1, 1024, T) as required by Wav2Lip
                    feat = np.transpose(hidden_states, (0, 2, 1))
                    
                    print(f"Extracted features shape: {feat.shape}")
                    return feat
                    
                except Exception as e:
                    print(f"Error extracting features: {e}")
                    traceback.print_exc()
                    print("Falling back to placeholder.")

            print(f"WARNING: Input is {audio_path}. The model expects 1024-dim features.")
            if not self.hubert_extractor:
                print("HuBERT model not loaded. Please provide --hubert_path to extract features from raw audio.")
            
            print("Returning a placeholder (random noise) for demonstration purposes.")
            
            # Placeholder: Estimate length based on duration
            try:
                # Use torchaudio to get duration
                info = torchaudio.info(audio_path)
                duration = info.num_frames / info.sample_rate
                # Estimate T (approx 50Hz)
                T = int(duration * 50) + 20 
            except Exception:
                print("Could not determine audio duration, defaulting to 10s.")
                T = 500

            return np.random.randn(1, 1024, T).astype(np.float32)

    def preprocess_batch(self, img_batch, mel_batch):
        """
        Preprocess batch for inference, logic from process_wav2lip.py
        img_batch: List of numpy images (H, W, 3) BGR (cv2 default)
        mel_batch: List of audio chunks (10, 1024)
        """
        # Process Images
        # process_wav2lip.py: 
        # 1. transpose to (2, 0, 1) -> (C, H, W)
        # 2. concatenate(image, image) axis=0 -> (2*C, H, W)
        # 3. float16 -> float32 for ONNX
        
        imgs_np = []
        for img in img_batch:
            # Resize if needed
            if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
                img = cv2.resize(img, (self.img_size, self.img_size))
            
            # CV2 is BGR, convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            img = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0
            img_concat = np.concatenate((img, img), axis=0)
            imgs_np.append(img_concat)
            
        imgs_np = np.array(imgs_np) # (Batch, 6, H, W)
        
        # Process Audio
        # process_wav2lip.py: reshapes to (10, 1024)
        # Here mel_batch is already (Batch, 10, 1024) based on our datagen logic
        audio_np = np.array(mel_batch, dtype=np.float32)
        
        # Initialize hidden states
        # hn, cn shape: (Batch, 2, 512)
        hn = np.zeros((imgs_np.shape[0], 2, 512), dtype=np.float32)
        cn = np.zeros((imgs_np.shape[0], 2, 512), dtype=np.float32)
        
        # Transpose hn, cn as in wav2lip.py infer()
        # hn = hn.transpose(1, 0, 2) -> (2, Batch, 512)
        hn = hn.transpose(1, 0, 2)
        cn = cn.transpose(1, 0, 2)
        
        return imgs_np, audio_np, hn, cn

    def run(self, frames, audio_feat, batch_size=WAV2LIP_BATCH_SIZE):
        """
        Main inference loop
        frames: List of video frames
        audio_feat: Audio features (1, 1024, T)
        """
        # Logic from get_lips_result.py: create_rep_chunks_faceresults
        # But simplified since we assume frames are already face crops or full frames?
        # The complex chain separates face detection (get_avatar_info) from inference.
        # Here we assume frames are full frames, we might need face detection.
        # But process_wav2lip.py seems to take *images* (bytes).
        # Let's assume input frames are already the target face area or we resize the whole frame.
        # For simplicity, we resize the whole frame to IMG_SIZE.
        
        results = []
        
        # Data generation
        # Logic from get_lips_result.py
        rep_chunks = []
        # rep_idx_multiplier = 2
        i = 0
        while True:
            start_idx = int(i * REP_IDX_MULTIPLIER)
            if start_idx + MEL_STEP_SIZE > audio_feat.shape[-1]:
                if start_idx > audio_feat.shape[-1] - 1:
                    break
                # Padding or partial chunk logic
                chunk = audio_feat[0, :, audio_feat.shape[-1] - MEL_STEP_SIZE:]
                rep_chunks.append(chunk)
            else:
                chunk = audio_feat[0, :, start_idx : start_idx + MEL_STEP_SIZE]
                rep_chunks.append(chunk)
            i += 1
            
        # Truncate to min length
        min_len = min(len(frames), len(rep_chunks))
        frames = frames[:min_len]
        rep_chunks = rep_chunks[:min_len]
        
        print(f"Processing {min_len} frames in batches of {batch_size}...")
        
        for i in tqdm(range(0, min_len, batch_size)):
            batch_frames = frames[i : i + batch_size]
            batch_audio = rep_chunks[i : i + batch_size]
            
            # Prepare batch - Transpose audio chunk from (1024, 10) to (10, 1024)
            # get_lips_result.py: mel_batch = np.transpose(mel_batch, (0, 2, 1))
            batch_audio_transposed = [np.transpose(a, (1, 0)) for a in batch_audio]
            
            img_in, audio_in, hn_in, cn_in = self.preprocess_batch(batch_frames, batch_audio_transposed)
            
            inputs = {
                'audio': audio_in,
                'face': img_in,
                'in_hn': hn_in,
                'in_cn': cn_in
            }
            
            # Run inference
            # Output: pred_face, out_hn, out_cn
            pred = self.session.run(['pred_face'], inputs)[0]
            
            # Postprocess
            # wav2lip.py: g = (g.transpose(0, 2, 3, 1) * 255.).astype(np.uint8)
            # pred shape: (Batch, C, H, W)
            pred = pred.transpose(0, 2, 3, 1) * 255.0
            pred = pred.clip(0, 255).astype(np.uint8)
            
            results.extend(pred)
            
        return results

def main():
    parser = argparse.ArgumentParser(description='Simplified Wav2Lip Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to model.onnx')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--audio', type=str, required=True, help='Path to input audio (.wav) or features (.npy)')
    parser.add_argument('--output', type=str, default='result.mp4', help='Path to output video')
    parser.add_argument('--hubert_path', type=str, default=None, help='Path to chinese-hubert-large model')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    args = parser.parse_args()

    # 1. Load Audio Features
    inference = Wav2LipInference(args.model, args.hubert_path, args.device)
    audio_feat = inference.get_audio_features(args.audio)
    
    # 2. Load Video
    if not os.path.exists(args.video):
        print(f"Video file not found: {args.video}")
        sys.exit(1)
        
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    if len(frames) == 0:
        print("No frames read from video.")
        sys.exit(1)
        
    # 3. Run Inference
    result_frames = inference.run(frames, audio_feat)
    
    # 4. Save Video
    if len(result_frames) > 0:
        h, w = result_frames[0].shape[:2]
        out = cv2.VideoWriter('temp_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (w, h))
        for frame in result_frames:
            # Determine if we need to resize back or paste back?
            # For simplicity, we just save the face result (512x512 typically)
            # If original video was different, this will result in a cropped/resized video.
            # To do full face swap, we'd need the face detection/affine transform logic.
            out.write(frame)
        out.release()
        
        # Merge audio
        # If audio input is .wav, use it. If .npy, we can't merge audio easily unless original wav provided.
        audio_cmd = []
        if args.audio.endswith('.wav') or args.audio.endswith('.mp3'):
            cmd = f"ffmpeg -y -i temp_video.avi -i {args.audio} -c:v libx264 -c:a aac -strict experimental -shortest {args.output}"
            subprocess.call(cmd, shell=True)
            print(f"Saved result to {args.output}")
        else:
            print(f"Audio is .npy, saving video without audio merging to {args.output}")
            # Rename temp to output
            shutil.move('temp_video.avi', args.output)

if __name__ == "__main__":
    import shutil
    main()

