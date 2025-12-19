import os
import sys
import argparse
import numpy as np
import cv2
import onnxruntime as ort
from tqdm import tqdm
import subprocess
import platform

# Try to import librosa for audio duration check
try:
    import librosa
except ImportError:
    librosa = None

class Wav2LipONNXRunner:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.session = self.load_model(model_path)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        print(f"Model loaded from {model_path}")
        print(f"Inputs: {self.input_names}")

    def load_model(self, path):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
        try:
            return ort.InferenceSession(path, providers=providers)
        except Exception as e:
            print(f"Failed to load model with {providers}, falling back to CPU.")
            return ort.InferenceSession(path, providers=['CPUExecutionProvider'])

    def run(self, inputs):
        output_names = [out.name for out in self.session.get_outputs()]
        return self.session.run(output_names, inputs)

def get_audio_features(audio_path, expected_shape=(10, 1024)):
    """
    Load audio features. 
    Since the feature extraction model is remote/proprietary, this function expects 
    a pre-computed .npy file with shape (1, 1024, T) or similar.
    
    If a .wav is provided, we currently cannot extract the exact 1024-dim features 
    required by this specific ONNX model without the feature extractor.
    """
    if audio_path.endswith('.npy'):
        # Expecting shape [1, 1024, Time] or [1024, Time]
        feat = np.load(audio_path)
        if feat.ndim == 2:
            feat = feat[np.newaxis, :, :] # (1024, T) -> (1, 1024, T)
        return feat
    else:
        print(f"WARNING: Input is {audio_path}. This ONNX model requires pre-computed features (1024 dim).")
        print("Please provide the path to the .npy feature file.")
        print("Returning random features for demonstration purposes.")
        
        # Mock features: Assume 25 FPS, so T = duration * 25
        # If librosa is available, get duration
        T = 100 # default
        if librosa and os.path.exists(audio_path):
            y, sr = librosa.load(audio_path, sr=16000)
            duration = librosa.get_duration(y=y, sr=sr)
            T = int(duration * 25) # 25 FPS
            
        return np.random.randn(1, 1024, T).astype(np.float32)

def datagen(frames, audio_feat, batch_size=8, img_size=512):
    """
    Generator that yields batches of (img, audio_chunk, hn, cn)
    Based on get_lips_result.py logic
    """
    # audio_feat shape: (1, 1024, T)
    # We need to slice it into chunks of 10 steps
    # Stride appears to be 2 in get_lips_result.py? 
    # rep_idx_multiplier = 2 # per frame audio feature index multiplier?
    # Actually, let's look at get_lips_result.py logic again carefully:
    # rep_step_size=10, rep_idx_multiplier = 2
    # start_idx = int(i * rep_idx_multiplier)
    # chunk = audio_tensor[0, :, start_idx : start_idx + rep_step_size]
    
    rep_step_size = 10
    rep_idx_multiplier = 2 
    
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
    
    # Hidden states
    hn_batch, cn_batch = [], []

    n_frames = len(frames)
    
    # We iterate based on frames
    for i in range(len(frames)):
        # Audio chunk selection
        start_idx = int(i * rep_idx_multiplier)
        if start_idx + rep_step_size > audio_feat.shape[-1]:
             # Padding or last chunk reuse
             chunk = audio_feat[0, :, -rep_step_size:]
        else:
             chunk = audio_feat[0, :, start_idx : start_idx + rep_step_size]
        
        # Check chunk shape (1024, 10) -> Need (10, 1024) for model?
        # get_lips_result.py:
        # mel_batch.append(chunk) -> chunk is (1024, 10)
        # later: np.transpose(mel_batch, (0, 2, 1)) -> (B, 10, 1024)
        
        # Face processing
        frame = frames[i]
        # Resize to model input size (e.g. 512x512)
        resized_frame = cv2.resize(frame, (img_size, img_size))
        
        # Preprocess image: (H, W, C) -> (C, H, W) -> float32 0-1
        processed_img = resized_frame.astype(np.float32) / 255.0
        processed_img = np.transpose(processed_img, (2, 0, 1))
        
        # Concatenate with itself (reference code does this)
        # image_concat = np.concatenate((image, image), axis=0) -> (6, H, W)
        processed_img = np.concatenate((processed_img, processed_img), axis=0)
        
        img_batch.append(processed_img)
        mel_batch.append(chunk)
        frame_batch.append(frame) # Original frame for saving
        
        if len(img_batch) >= batch_size or i == len(frames) - 1:
            # Convert to numpy
            img_batch_np = np.array(img_batch, dtype=np.float32) # (B, 6, H, W)
            
            # Process audio batch
            # chunk list: [ (1024, 10), (1024, 10) ... ]
            mel_batch_np = np.array(mel_batch, dtype=np.float32) # (B, 1024, 10)
            mel_batch_np = np.transpose(mel_batch_np, (0, 2, 1)) # (B, 10, 1024)
            
            # Prepare Hidden States (B, 2, 512) -> Transposed to (2, B, 512)
            # Reference: hn = np.zeros((img_numpy.shape[0], 2, 512))
            curr_bs = img_batch_np.shape[0]
            hn = np.zeros((curr_bs, 2, 512), dtype=np.float32)
            cn = np.zeros((curr_bs, 2, 512), dtype=np.float32)
            
            hn = np.transpose(hn, (1, 0, 2))
            cn = np.transpose(cn, (1, 0, 2))
            
            yield img_batch_np, mel_batch_np, hn, cn, frame_batch
            
            img_batch, mel_batch, frame_batch = [], [], []

def main():
    parser = argparse.ArgumentParser(description="Wav2Lip ONNX Inference Script")
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to .onnx model')
    parser.add_argument('--face', type=str, required=True, help='Path to input video')
    parser.add_argument('--audio', type=str, required=True, help='Path to input audio (.wav) or feature (.npy)')
    parser.add_argument('--outfile', type=str, default='result.mp4', help='Output video path')
    parser.add_argument('--img_size', type=int, default=512, help='Model input image size')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device: cuda or cpu')
    
    args = parser.parse_args()
    
    # 1. Load Model
    runner = Wav2LipONNXRunner(args.checkpoint_path, args.device)
    
    # 2. Load Audio Features
    print("Loading audio...")
    audio_feat = get_audio_features(args.audio) # (1, 1024, T)
    print(f"Audio features shape: {audio_feat.shape}")
    
    # 3. Load Video
    print("Loading video...")
    cap = cv2.VideoCapture(args.face)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"Loaded {len(frames)} frames.")
    
    # 4. Prepare Output Video
    # Use original size for output or model size? Usually we resize back to original or save as model size.
    # Reference code saves the result.
    out_w, out_h = args.img_size, args.img_size
    # If you want to paste back to original, logic is more complex (affine transform). 
    # Here we output the generated face directly (aligned/cropped assumed or full frame).
    # Since we are just resizing the whole frame to img_size, we output img_size.
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('temp_result.mp4', fourcc, fps, (out_w, out_h))
    
    # 5. Inference Loop
    gen = datagen(frames, audio_feat, args.batch_size, args.img_size)
    total_batches = int(np.ceil(len(frames) / args.batch_size))
    
    print("Starting inference...")
    for img_batch, mel_batch, hn, cn, original_frames in tqdm(gen, total=total_batches):
        
        inputs = {
            runner.input_names[0]: mel_batch, # 'audio'
            runner.input_names[1]: img_batch, # 'face'
            runner.input_names[2]: hn,        # 'in_hn'
            runner.input_names[3]: cn         # 'in_cn'
        }
        
        # Name mapping fallback
        if 'audio' not in runner.input_names[0]:
             # Try mapping by name
             inputs = {}
             for name in runner.input_names:
                 if 'audio' in name or 'mel' in name: inputs[name] = mel_batch
                 elif 'face' in name or 'img' in name: inputs[name] = img_batch
                 elif 'hn' in name: inputs[name] = hn
                 elif 'cn' in name: inputs[name] = cn
        
        try:
            res = runner.run(inputs)
            # Output order: pred_face, out_hn, out_cn
            pred = res[0] # (B, 3, H, W)
            
            # Post-process
            # (B, 3, H, W) -> (B, H, W, 3) -> 0-255 -> uint8
            pred = np.transpose(pred, (0, 2, 3, 1))
            pred = (pred * 255.0).astype(np.uint8)
            
            for p in pred:
                out.write(p)
                
        except Exception as e:
            print(f"Inference error: {e}")
            break
            
    out.release()
    
    # 6. Merge Audio
    if args.audio.endswith('.wav'):
        print("Merging audio...")
        cmd = f"ffmpeg -y -i temp_result.mp4 -i {args.audio} -c:v copy -c:a aac {args.outfile} -loglevel quiet"
        subprocess.call(cmd, shell=True)
        print(f"Saved to {args.outfile}")
    else:
        print(f"Video saved to temp_result.mp4 (No audio merged as input was .npy)")
        os.rename('temp_result.mp4', args.outfile)

if __name__ == "__main__":
    main()

