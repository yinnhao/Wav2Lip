import sys
import os
import argparse
import numpy as np
import cv2
import torch
import torchaudio
import onnxruntime as ort
from transformers import HubertModel
from tqdm import tqdm
import subprocess
import platform

# Add Wav2Lip to sys.path to import face_detection
sys.path.append('Wav2Lip')
import face_detection

parser = argparse.ArgumentParser(description='Inference code to lip-sync videos using Wav2Lip models with Hubert features')

parser.add_argument('--checkpoint_path', type=str, 
					help='Path to the ONNX model', required=True)
parser.add_argument('--hubert_path', type=str, 
					help='Path to the Hubert model directory', required=True)
parser.add_argument('--face', type=str, 
					help='Filepath of video/image that contains faces to use', required=True)
parser.add_argument('--audio', type=str, 
					help='Filepath of video/audio file to use as raw audio source', required=True)
parser.add_argument('--outfile', type=str, help='Video path to save result', 
					default='results/result_voice.mp4')

parser.add_argument('--static', type=bool, 
					help='If True, then use only first video frame for inference', default=False)
parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)', 
					default=25., required=False)
parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], 
					help='Padding (top, bottom, left, right)')
parser.add_argument('--face_det_batch_size', type=int, 
					help='Batch size for face detection', default=16)
parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=128)
parser.add_argument('--resize_factor', default=1, type=int, 
			help='Reduce the resolution by this factor.')
parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1], 
					help='Crop video to a smaller region (top, bottom, left, right).')
parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], 
					help='Specify a constant bounding box for the face.')
parser.add_argument('--rotate', default=False, action='store_true',
					help='Sometimes videos taken from a phone can be flipped 90deg')
parser.add_argument('--nosmooth', default=False, action='store_true',
					help='Prevent smoothing face detections over a short temporal window')

args = parser.parse_args()
args.img_size = 96

# Device handling
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

# -----------------------------------------------------------------------------
# Face Detection Logic (Adapted from Wav2Lip/inference.py)
# -----------------------------------------------------------------------------

def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes

def face_detect(images):
	detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
											flip_input=False, device=device)

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
	for rect, image in zip(predictions, images):
		if rect is None:
			cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
			raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

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

# -----------------------------------------------------------------------------
# Hubert Feature Extraction (Adapted from hubert_audio)
# -----------------------------------------------------------------------------

class HubertFeatureExtractor:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hubert_model = HubertModel.from_pretrained(model_path).to(self.device).eval()

    def infer(self, audio_path):
        # Load audio using torchaudio
        wav, sr = torchaudio.load(audio_path)
        
        # Resample to 16000 if needed (Hubert expects 16k)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            wav = resampler(wav)
        
        # Mono channel
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
            
        input_values = wav.to(self.device)

        res_lst = []
        with torch.no_grad():
            kernel, stride = 400, 320
            clip_length = stride * 1000 # Process 1000 frames at a time to avoid OOM
            
            if input_values.shape[1] <= clip_length:
                hidden_states = self.hubert_model(input_values).last_hidden_state
            else:
                num_iter = input_values.shape[1] // clip_length
                expected_T = (input_values.shape[1] - (kernel - stride)) // stride
                slices = []
                for i in range(num_iter):
                    start_idx = i * clip_length
                    end_idx = start_idx + (clip_length - stride + kernel)
                    slices.append((start_idx, end_idx))

                remaining_start = num_iter * clip_length
                remaining = input_values[:, remaining_start:]
                if remaining.shape[1] >= kernel:
                    slices.append((remaining_start, input_values.shape[1]))

                res_lst = []
                for start, end in slices:
                    chunk = input_values[:, start:end]
                    # Note: Handle edge case where chunk might be too small for the model kernel
                    if chunk.shape[1] >= kernel:
                        clip_hidden_states = self.hubert_model(chunk).last_hidden_state
                        res_lst.append(clip_hidden_states[0])
                
                if len(res_lst) > 0:
                    ret = torch.cat(res_lst, dim=0)
                    # Adjust length
                    if ret.shape[0] >= expected_T:
                         ret = ret[:expected_T]
                    else:
                         # Padding if slightly short
                         ret = torch.nn.functional.pad(ret, (0, 0, 0, expected_T - ret.shape[0]))
                    hidden_states = ret.unsqueeze(0)
                else:
                    # Fallback for short remaining
                    hidden_states = self.hubert_model(input_values).last_hidden_state

        # Post-processing: Permute to (Batch, Hidden, Time) as per hubert_audio/operators/hubert.py logic
        # logic: repst = result.permute(0, 2, 1)
        hidden_states = hidden_states.permute(0, 2, 1)
        return hidden_states.cpu().numpy()

# -----------------------------------------------------------------------------
# Wav2Lip Model (ONNX)
# -----------------------------------------------------------------------------

class Wav2LipModel:
    def __init__(self, checkpoint_path):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(checkpoint_path, providers=providers)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        
    def infer(self, img_batch, audio_batch):
        # inputs: 
        # img_batch: (B, 6, 96, 96) - float32 or float16? 
        # audio_batch: (B, 10, 1024) - float16 (based on process_wav2lip.py)
        
        # NOTE: process_wav2lip.py uses float16 for inputs. 
        # We need to check what the model actually expects or just try float32 if on CPU/standard.
        # But process_wav2lip.py explicitly casts to float16.
        # Let's try to infer from session inputs types if possible, otherwise follow process_wav2lip.
        
        # Prepare hidden states (hn, cn)
        # Shape: (2, B, 512)
        batch_size = img_batch.shape[0]
        hn = np.zeros((2, batch_size, 512), dtype=np.float16)
        cn = np.zeros((2, batch_size, 512), dtype=np.float16)
        
        # Prepare inputs dict
        # Ensure types match. process_wav2lip uses float16 for images and audio.
        # However, ONNX Runtime in Python often works with float32 unless strict.
        # Let's check input types from session if possible or default to float32 if float16 fails.
        # But for now, let's stick to what process_wav2lip does: float16.
        
        img_batch = img_batch.astype(np.float16)
        audio_batch = audio_batch.astype(np.float16)
        
        inputs = {
            'audio': audio_batch,
            'face': img_batch,
            'in_hn': hn,
            'in_cn': cn
        }
        
        # Run
        # Output: pred_face, out_hn, out_cn
        res = self.session.run(['pred_face'], inputs)[0]
        
        # Post process
        # Output shape is typically (B, 3, 96, 96) or (B, C, H, W)
        # Convert back to uint8 image
        res = res.astype(np.float32)
        res = res.transpose(0, 2, 3, 1) * 255.
        return res

# -----------------------------------------------------------------------------
# Main Pipeline
# -----------------------------------------------------------------------------

def main():
    if not os.path.isfile(args.face):
        raise ValueError('--face argument must be a valid path to video/image file')

    # 1. Load Video
    print('Reading video frames...')
    if args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        full_frames = [cv2.imread(args.face)]
        fps = args.fps
    else:
        video_stream = cv2.VideoCapture(args.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        
        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            if args.resize_factor > 1:
                frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))
            if args.rotate:
                frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
            
            y1, y2, x1, x2 = args.crop
            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]
            frame = frame[y1:y2, x1:x2]
            full_frames.append(frame)

    print ("Number of frames available for inference: "+str(len(full_frames)))

    # 2. Extract Audio Features (Hubert)
    print('Extracting audio features...')
    # If video file is provided as audio source, extract wav first
    if not args.audio.endswith('.wav'):
        print('Extracting raw audio...')
        if not os.path.exists('temp'): os.makedirs('temp')
        command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, 'temp/temp.wav')
        subprocess.call(command, shell=True)
        args.audio = 'temp/temp.wav'

    hubert_extractor = HubertFeatureExtractor(args.hubert_path)
    # hubert_features shape: (1, 1024, T)
    hubert_features = hubert_extractor.infer(args.audio) 
    # Transpose to (T, 1024) for easier slicing
    hubert_features = hubert_features.squeeze(0).transpose(1, 0) 
    print("Hubert features shape:", hubert_features.shape)

    # 3. Detect Faces
    if args.box[0] == -1:
        if not args.static:
            face_det_results = face_detect(full_frames)
        else:
            face_det_results = face_detect([full_frames[0]])
    else:
        print('Using the specified bounding box instead of face detection...')
        y1, y2, x1, x2 = args.box
        face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in full_frames]

    # 4. Prepare Data for Inference
    # Logic for alignment:
    # Hubert features are 50Hz (20ms stride).
    # Video is `fps`.
    # We need to map each video frame to a window of Hubert features.
    # Window size = 10 (from process_wav2lip.py).
    
    # Calculate how many hubert frames correspond to one video frame
    # Hubert rate = 16000 / 320 = 50 Hz
    hubert_fps = 50.0
    
    inputs = []
    
    # We can cycle video frames if audio is longer
    # Or cut audio if video is shorter (Wav2Lip behavior usually cuts to shortest)
    # But usually we want to cover the whole audio.
    
    # Calculate number of video frames needed for the audio
    num_audio_frames = hubert_features.shape[0]
    duration_sec = num_audio_frames / hubert_fps
    num_video_frames_needed = int(np.ceil(duration_sec * fps))
    
    print(f"Audio duration: {duration_sec:.2f}s, Required video frames: {num_video_frames_needed}")
    
    # Prepare batch data
    img_batch = []
    audio_batch = []
    frame_batch = []
    coords_batch = []
    
    wav2lip_model = Wav2LipModel(args.checkpoint_path)
    print("Model loaded")
    
    if not os.path.exists('temp'): os.makedirs('temp')
    
    # Setup Video Writer
    frame_h, frame_w = full_frames[0].shape[:-1]
    out = cv2.VideoWriter('temp/result.avi', 
                            cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

    for i in tqdm(range(num_video_frames_needed)):
        # Select video frame (loop if needed)
        idx = 0 if args.static else i % len(full_frames)
        frame_to_save = full_frames[idx].copy()
        face, coords = face_det_results[idx].copy() # face is BGR
        
        # Preprocess Face
        # 1. Resize to 96x96
        face = cv2.resize(face, (args.img_size, args.img_size))
        
        # 2. Masking (Lower half = 0)
        # inference.py logic: img_masked[:, args.img_size//2:] = 0
        face_masked = face.copy()
        face_masked[args.img_size//2:, :] = 0
        
        # 3. Concatenate (Masked, Original) - Channels: (B, G, R, B, G, R)
        # inference.py: np.concatenate((img_masked, img_batch), axis=3)
        # Here we do it per image first
        # Result shape: (96, 96, 6)
        face_concat = np.concatenate((face_masked, face), axis=2)
        
        # 4. Transpose to (6, 96, 96) and Normalize
        face_input = np.transpose(face_concat, (2, 0, 1)) / 255.
        
        # Select Audio Window
        # Center of the window should correspond to the current video frame time
        # Current time = i / fps
        # Hubert index = current_time * hubert_fps
        # Window size 10 means we need [center - 5, center + 5]?
        # Or [start, start + 10]?
        # Usually centered.
        
        center_idx = int((i / fps) * hubert_fps)
        # Assuming window of 10. Let's try centered: [center-5, center+5]
        # But we must ensure indices are valid.
        start_idx = center_idx - 5
        end_idx = center_idx + 5
        
        # Handle boundary conditions with padding
        if start_idx < 0:
            # Pad beginning
            feat_window = np.pad(hubert_features, ((abs(start_idx), 0), (0, 0)), mode='edge')[:10]
            # Wait, easier logic:
            # Extract valid part and pad
            # Actually, let's just clamp indices or pad the whole feature array once.
            pass
        
        # Easier: Pad hubert_features globally before loop
        pass 
        
        # Let's adjust loop logic slightly.
        # We need to buffer inputs and run batch inference
        
        img_batch.append(face_input)
        # We need to extract the audio window here.
        # Let's handle audio extraction properly below.
        
        audio_batch.append((start_idx, end_idx)) # Store indices for now
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)
        
        if len(img_batch) >= args.wav2lip_batch_size or i == num_video_frames_needed - 1:
            # Prepare batch tensors
            img_batch_np = np.array(img_batch) # (B, 6, 96, 96)
            
            # Prepare audio batch
            # We need to pad hubert_features to handle out of bounds
            # Max index needed is max(end_idx)
            # Min index needed is min(start_idx)
            
            audio_batch_np = []
            for s, e in audio_batch:
                if s < 0:
                    # Pad left
                    pad_left = -s
                    window = hubert_features[0:e]
                    window = np.pad(window, ((pad_left, 0), (0, 0)), mode='edge')
                elif e > hubert_features.shape[0]:
                    # Pad right
                    pad_right = e - hubert_features.shape[0]
                    window = hubert_features[s:]
                    window = np.pad(window, ((0, pad_right), (0, 0)), mode='edge')
                else:
                    window = hubert_features[s:e]
                
                # Ensure shape is (10, 1024)
                if window.shape[0] != 10:
                     # Fallback pad
                     curr_len = window.shape[0]
                     if curr_len < 10:
                        window = np.pad(window, ((0, 10 - curr_len), (0, 0)), mode='edge')
                     else:
                        window = window[:10]
                
                audio_batch_np.append(window)
            
            audio_batch_np = np.array(audio_batch_np) # (B, 10, 1024)
            
            # Inference
            pred = wav2lip_model.infer(img_batch_np, audio_batch_np)
            
            # Write Output
            for p, f, c in zip(pred, frame_batch, coords_batch):
                y1, y2, x1, x2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                f[y1:y2, x1:x2] = p
                out.write(f)
                
            img_batch, audio_batch, frame_batch, coords_batch = [], [], [], []

    out.release()
    
    # Merge Audio
    print("Merging audio...")
    command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio, 'temp/result.avi', args.outfile)
    subprocess.call(command, shell=platform.system() != 'Windows')

if __name__ == '__main__':
    main()

