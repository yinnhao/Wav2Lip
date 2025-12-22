import os
import sys
import numpy as np
import cv2
import onnxruntime as ort
import argparse
import subprocess
from tqdm import tqdm
import torch
import face_detection
import platform

# 尝试导入 torchaudio 和 transformers 用于 HuBERT
try:
    import torchaudio
    from transformers import HubertModel
except ImportError:
    torchaudio = None
    HubertModel = None
    print("Warning: torchaudio or transformers not found. HuBERT feature extraction will fail if needed.")

def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes

def face_detect(images, args):
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
                                            flip_input=False, device=args.device)

    batch_size = args.face_det_batch_size
    
    while 1:
        predictions = []
        try:
            for i in tqdm(range(0, len(images), batch_size)):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1: 
                raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = args.pads
    
    # Handle cases where face is not detected in some frames
    last_valid_rect = None

    for rect, image in zip(predictions, images):
        if rect is None:
            if last_valid_rect is not None:
                rect = last_valid_rect # Temporal consistency fallback
            else:
                # If no face found yet, we might need to skip or error out
                # For now, let's try to look ahead or just raise error if it's the first frame
                # Or just use the whole image (bad idea)
                # Let's raise error for now as in original script, but maybe log it
                # cv2.imwrite('temp/faulty_frame.jpg', image)
                # raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')
                # Try to use center crop?
                h, w = image.shape[:2]
                rect = [w//4, h//4, w*3//4, h*3//4] # Fallback dummy box
        
        last_valid_rect = rect

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
        
        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    del detector
    return results

class HubertFeatureExtractor:
    def __init__(self, model_path, device='cpu'):
        if HubertModel is None:
            raise ImportError("transformers is required for HuBERT.")
        self.device = device
        
        print(f"Loading HuBERT model from {model_path}...")
        try:
            self.hubert_model = HubertModel.from_pretrained(model_path).to(self.device).eval()
        except Exception as e:
            print(f"Standard loading failed: {e}. Trying manual loading (config + bin)...")
            try:
                config_path = os.path.join(model_path, "config.json")
                bin_path = os.path.join(model_path, "pytorch_model.bin")
                from transformers import HubertConfig
                config = HubertConfig.from_pretrained(config_path)
                self.hubert_model = HubertModel(config)
                if os.path.exists(bin_path):
                    state_dict = torch.load(bin_path, map_location=device)
                    self.hubert_model.load_state_dict(state_dict, strict=False)
                self.hubert_model = self.hubert_model.to(self.device).eval()
            except Exception as e2:
                print(f"Manual loading failed: {e2}")
                raise e

    def infer(self, input_values):
        # Input: numpy array (1, T)
        input_values = torch.tensor(input_values).to(self.device)
        if input_values.ndim == 1:
            input_values = input_values.unsqueeze(0)

        with torch.no_grad():
            # Based on common HuBERT usage in Wav2Lip variants (e.g. SadTalker, Linly)
            # The model outputs hidden states.
            outputs = self.hubert_model(input_values)
            hidden_states = outputs.last_hidden_state # (B, T, 1024)
            
        return hidden_states.cpu().numpy()

def datagen(frames, audio_tensor, face_det_results, args):
    """
    Generator that yields batches of (img_batch, mel_batch, frame_batch, coords_batch)
    audio_tensor: (1, 1024, T) - Transposed HuBERT features
    """
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    # Logic from get_lips_result.py: create_rep_chunks_faceresults
    # But adapted for generator to save memory
    
    rep_step_size = 10
    rep_idx_multiplier = 2  # Per video frame audio index multiplier (25fps video, 50Hz audio)
    
    # audio_tensor is expected to be (1, 1024, T)
    # We iterate through frames
    
    n_frames = len(frames)
    
    for i in range(len(frames)):
        # Calculate audio window
        start_idx = int(i * rep_idx_multiplier)
        
        # Check boundary
        if start_idx + rep_step_size > audio_tensor.shape[-1]:
             # Padding or repeating last part? 
             # get_lips_result.py logic:
             # if start_idx > ... break
             # else append last part
             if start_idx > audio_tensor.shape[-1] - 1:
                 break
             m = audio_tensor[0, :, audio_tensor.shape[-1] - rep_step_size:]
        else:
            m = audio_tensor[0, :, start_idx : start_idx + rep_step_size]
            
        # m shape is (1024, 10) -> We might need to transpose it back for ONNX model?
        # get_lips_result.py: rep_chunks.append(audio_tensor[...]) -> (1024, 10)
        # Then in preprocess: mel_batch.append(m)
        # Then: mel_batch = np.transpose(mel_batch, (0, 2, 1)) -> (B, 10, 1024)
        # So here m is (1024, 10).
        
        idx = i % len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx] # face is already cropped
        
        face = cv2.resize(face, (args.img_size, args.img_size))
        
        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)
        
        if len(img_batch) >= args.wav2lip_batch_size:
            # Batch preparation
            img_batch_np = np.asarray(img_batch) # (B, H, W, 3)
            mel_batch_np = np.asarray(mel_batch) # (B, 1024, 10)
            
            # Transpose mel to (B, 10, 1024) as expected by typical ONNX models
            mel_batch_np = np.transpose(mel_batch_np, (0, 2, 1))
            
            # Prepare Image Batch for Wav2Lip (Masking + Concatenation)
            # Standard Wav2Lip: (masked_face, reference_face)
            img_masked = img_batch_np.copy()
            img_masked[:, args.img_size//2:] = 0 # Mask lower half
            
            # Concatenate: (B, H, W, 6)
            # Reference: typically Wav2Lip takes (reference_face, masked_face) or vice versa
            # The original code: np.concatenate((img_masked, img_batch), axis=3)
            # Let's verify standard Wav2Lip input order: (window, 6, H, W) where 6 is (face_with_masked_mouth, reference_face)
            
            # Let's try to match original inference.py:
            # img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            
            # NOTE: In Wav2Lip, the input channel order is [masked_img, reference_img]
            # masked_img: image with lower half masked
            # reference_img: original image (or reference)
            # The original code logic was:
            # img_masked[:, args.img_size//2:] = 0
            # img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            # This matches our current logic: [masked, reference]
            
            img_input = np.concatenate((img_masked, img_batch_np), axis=3)
            
            # Normalize and Transpose to (B, 6, H, W)
            img_input = img_input / 255.
            img_input = np.transpose(img_input, (0, 3, 1, 2))
            
            yield img_input, mel_batch_np, frame_batch, coords_batch
            
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    # Yield remaining
    if len(img_batch) > 0:
        img_batch_np = np.asarray(img_batch)
        mel_batch_np = np.asarray(mel_batch)
        mel_batch_np = np.transpose(mel_batch_np, (0, 2, 1))
        
        img_masked = img_batch_np.copy()
        img_masked[:, args.img_size//2:] = 0
        img_input = np.concatenate((img_masked, img_batch_np), axis=3)
        img_input = img_input / 255.
        img_input = np.transpose(img_input, (0, 3, 1, 2))
        
        yield img_input, mel_batch_np, frame_batch, coords_batch

def load_onnx_model(path, device):
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
    try:
        sess = ort.InferenceSession(path, providers=providers)
        return sess
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

    parser.add_argument('--checkpoint_path', type=str, required=True, help='Name of saved checkpoint to load weights from')
    parser.add_argument('--face', type=str, required=True, help='Filepath of video/image that contains faces to use')
    parser.add_argument('--audio', type=str, required=True, help='Filepath of video/audio file to use as raw audio source')
    parser.add_argument('--outfile', type=str, default='results/result_voice.mp4', help='Video path to save result')
    parser.add_argument('--static', type=bool, default=False, help='If True, then use only first video frame for inference')
    parser.add_argument('--fps', type=float, default=25., help='Can be specified only if input is a static image')
    parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], help='Padding (top, bottom, left, right)')
    parser.add_argument('--face_det_batch_size', type=int, default=16, help='Batch size for face detection')
    parser.add_argument('--wav2lip_batch_size', type=int, default=128, help='Batch size for Wav2Lip model(s)')
    parser.add_argument('--resize_factor', default=1, type=int, help='Reduce the resolution by this factor')
    parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1], help='Crop video to a smaller region')
    parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], help='Specify a constant bounding box')
    parser.add_argument('--rotate', default=False, action='store_true', help='Rotate video 90deg')
    parser.add_argument('--nosmooth', default=False, action='store_true', help='Prevent smoothing face detections')
    
    # New args
    parser.add_argument('--hubert_path', type=str, default=None, help='Path to HuBERT model for feature extraction')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cpu or cuda)')
    
    args = parser.parse_args()
    args.img_size = 96 # Will be overwritten by ONNX input check

    if not os.path.isfile(args.face):
        raise ValueError('--face argument must be a valid path to video/image file')

    # 1. Load Face
    if args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        full_frames = [cv2.imread(args.face)]
        fps = args.fps
        args.static = True
    else:
        video_stream = cv2.VideoCapture(args.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        print('Reading video frames...')
        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            if args.resize_factor > 1:
                frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))
            if args.rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            y1, y2, x1, x2 = args.crop
            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]
            frame = frame[y1:y2, x1:x2]
            full_frames.append(frame)
    print ("Number of frames available for inference: "+str(len(full_frames)))

    # 2. Process Audio (HuBERT Feature Extraction)
    if not args.audio.endswith('.wav'):
        print('Extracting raw audio...')
        command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, 'temp/temp.wav')
        subprocess.call(command, shell=True)
        args.audio = 'temp/temp.wav'

    print("Extracting audio features...")
    # Load audio
    wav, sr = torchaudio.load(args.audio)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        wav = resampler(wav)
    
    # Mix down to mono if needed
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
        
    # Extract HuBERT features
    if args.hubert_path:
        feature_extractor = HubertFeatureExtractor(args.hubert_path, args.device)
        audio_feat = feature_extractor.infer(wav.numpy()[0]) # (1, T, 1024)
        # Transpose to (1, 1024, T) to match get_lips_result.py logic
        audio_tensor = audio_feat.transpose(0, 2, 1) 
    else:
        # Fallback or error?
        print("WARNING: No --hubert_path provided. Using random noise for testing.")
        # Create dummy features matching expected duration
        # HuBERT is 50Hz. Duration = wav length / 16000
        duration = wav.shape[1] / 16000
        num_frames = int(duration * 50)
        audio_tensor = np.random.randn(1, 1024, num_frames).astype(np.float32)

    print(f"Audio tensor shape: {audio_tensor.shape}")

    # 3. Face Detection
    if args.box[0] == -1:
        if not args.static:
            face_det_results = face_detect(full_frames, args)
        else:
            face_det_results = face_detect([full_frames[0]], args)
    else:
        print('Using the specified bounding box instead of face detection...')
        y1, y2, x1, x2 = args.box
        face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in full_frames]

    # 4. Load Model
    model = load_onnx_model(args.checkpoint_path, args.device)
    print("Model loaded")
    
    # Check model inputs
    onnx_inputs = model.get_inputs()
    input_names = [inp.name for inp in onnx_inputs]
    print(f"Model inputs: {input_names}")
    
    # Check for img_size in ONNX model
    for inp in onnx_inputs:
        if 'face' in inp.name or 'img' in inp.name:
            if len(inp.shape) == 4:
                # Assuming (Batch, C, H, W)
                if isinstance(inp.shape[2], int):
                    args.img_size = inp.shape[2]
                    print(f"Set img_size to {args.img_size} based on model.")
    
    # 5. Inference Loop
    gen = datagen(full_frames.copy(), audio_tensor, face_det_results, args)
    
    # Prepare video writer
    frame_h, frame_w = full_frames[0].shape[:-1]
    out = cv2.VideoWriter('temp/result.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))
    
    batch_size = args.wav2lip_batch_size
    
    # Check dtype
    dtype = np.float32
    if 'float16' in onnx_inputs[0].type:
        dtype = np.float16
        
    for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen)):
        
        img_batch = img_batch.astype(dtype)
        mel_batch = mel_batch.astype(dtype)
        
        # Prepare hidden states (hn, cn)
        # Assuming (Batch, 2, 512) transposed to (2, Batch, 512)
        batch_size_curr = img_batch.shape[0]
        hn = np.zeros((batch_size_curr, 2, 512), dtype=dtype)
        cn = np.zeros((batch_size_curr, 2, 512), dtype=dtype)
        
        # Transpose to (2, B, 512)
        hn = np.transpose(hn, (1, 0, 2))
        cn = np.transpose(cn, (1, 0, 2))
        
        inputs = {}
        # Map inputs based on names
        for name in input_names:
            if 'audio' in name:
                inputs[name] = mel_batch
            elif 'face' in name:
                inputs[name] = img_batch
            elif 'in_hn' in name:
                inputs[name] = hn
            elif 'in_cn' in name:
                inputs[name] = cn
                
        # Run inference
        try:
            # We assume the first output is the predicted face
            # Or use explicit names if known: ['pred_face', 'out_hn', 'out_cn']
            pred = model.run(['pred_face'], inputs)[0]
        except Exception as e:
            print(f"Inference failed: {e}")
            sys.exit(1)
            
        pred = pred.astype(np.float32)
        # Output shape is typically (Batch, C, H, W). Transpose to (Batch, H, W, C)
        pred = pred.transpose(0, 2, 3, 1) * 255.
        
        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            
            # Resize prediction to original face box size
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            
            # Simple Poisson Blending or Soft Blending could help reducing the "box" effect
            # But first let's ensure the color space is correct.
            # cv2.imread reads as BGR.
            # Wav2Lip training usually assumes RGB input if using skvideo/PIL, or BGR if using cv2.
            # Standard Wav2Lip inference.py uses cv2.imread (BGR) and doesn't convert to RGB for model input explicitly in datagen.
            # But let's check if the model output needs BGR/RGB conversion.
            # If the model output is RGB (common in PyTorch models trained with PIL), we might need to convert to BGR for cv2.imwrite.
            
            # If the result has color shift (blue tint?), it means model output is RGB but we save as BGR (or vice versa).
            # Let's assume model output is BGR (since input was BGR).
            
            f[y1:y2, x1:x2] = p
            out.write(f)
            
    out.release()
    
    command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio, 'temp/result.avi', args.outfile)
    subprocess.call(command, shell=platform.system() != 'Windows')

if __name__ == '__main__':
    main()
