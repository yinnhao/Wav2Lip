import os
import sys
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchaudio
import onnxruntime as ort
from transformers import HubertModel
from tqdm import tqdm
import subprocess
import platform

# 复用 Wav2Lip 的人脸检测
sys.path.append('Wav2Lip')
import face_detection  # noqa: E402

parser = argparse.ArgumentParser(description='Lip-sync inference with HuBERT features and ONNX Wav2Lip')

parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to Wav2Lip ONNX checkpoint')
parser.add_argument('--hubert_path', type=str, required=True, help='Path or name of HuBERT model (transformers format)')
parser.add_argument('--face', type=str, required=True, help='Video/image containing the talking face')
parser.add_argument('--audio', type=str, required=True, help='Audio file or video with audio track')
parser.add_argument('--outfile', type=str, default='results/result_voice.mp4', help='Output video path')

parser.add_argument('--static', type=bool, default=False, help='Use only first frame if input is image')
parser.add_argument('--fps', type=float, default=25., help='FPS when input is a static image')
parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], help='Padding (top, bottom, left, right)')
parser.add_argument('--face_det_batch_size', type=int, default=16, help='Batch size for face detection')
parser.add_argument('--wav2lip_batch_size', type=int, default=8, help='Batch size for Wav2Lip inference')
parser.add_argument('--resize_factor', default=1, type=int, help='Resize factor for input video')
parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1], help='Crop region (top, bottom, left, right)')
parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], help='Optional fixed face bounding box')
parser.add_argument('--rotate', default=False, action='store_true', help='Rotate video 90deg clockwise')
parser.add_argument('--nosmooth', default=False, action='store_true', help='Disable smoothing for detected boxes')

args = parser.parse_args()
args.img_size = 96

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} for inference.')


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
	last_valid_rect = None  # 回退使用上一帧的检测框
	pady1, pady2, padx1, padx2 = args.pads
	for rect, image in zip(predictions, images):
		if rect is None:
			# 若检测失败，尝试使用上一帧结果，否则使用中心框兜底，避免整个流程中断
			if last_valid_rect is not None:
				rect = last_valid_rect
			else:
				h, w = image.shape[:2]
				rect = [w//4, h//4, w*3//4, h*3//4]

		y1 = max(0, rect[1] - pady1)
		y2 = min(image.shape[0], rect[3] + pady2)
		x1 = max(0, rect[0] - padx1)
		x2 = min(image.shape[1], rect[2] + padx2)
		
		results.append([x1, y1, x2, y2])
		last_valid_rect = rect

	boxes = np.array(results)
	if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
	results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

	del detector
	return results 


def load_hubert_model(model_path):
	model = HubertModel.from_pretrained(model_path)
	return model.to(device).eval()


def extract_hubert_features(audio_path, model_path):
	wav, sr = torchaudio.load(audio_path)
	if sr != 16000:
		resampler = torchaudio.transforms.Resample(sr, 16000)
		wav = resampler(wav)
	if wav.shape[0] > 1:
		wav = torch.mean(wav, dim=0, keepdim=True)

	wav = wav.to(device)
	model = load_hubert_model(model_path)

	kernel, stride = 400, 320
	clip_length = stride * 1000  # ~20s per chunk to avoid OOM
	expected_T = (wav.shape[1] - (kernel - stride)) // stride

	with torch.no_grad():
		if wav.shape[1] <= clip_length:
			hidden_states = model(wav).last_hidden_state
		else:
			res_lst = []
			num_iter = wav.shape[1] // clip_length
			for i in range(num_iter):
				start_idx = i * clip_length
				end_idx = start_idx + (clip_length - stride + kernel)
				chunk = wav[:, start_idx:end_idx]
				if chunk.shape[1] >= kernel:
					res_lst.append(model(chunk).last_hidden_state[0])
			remaining_start = num_iter * clip_length
			remaining = wav[:, remaining_start:]
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


def create_rep_chunks(audio_tensor, num_frames):
	rep_step_size = 10
	rep_idx_multiplier = 2
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
		idx = i % num_frames if (i // num_frames) % 2 == 0 else num_frames - 1 - i % num_frames
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

			# 对齐服务链：图像双份拼接；音频特征转为 (B, 10, 1024)
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
	providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
	return ort.InferenceSession(path, providers=providers)


def adjust_img_size_from_onnx(sess):
	for inp in sess.get_inputs():
		if len(inp.shape) == 4 and isinstance(inp.shape[2], int) and isinstance(inp.shape[3], int):
			return inp.shape[2]
	return args.img_size


def main():
	if not os.path.isfile(args.face):
		raise ValueError('--face argument must be a valid path to video/image file')

	if args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
		args.static = True
		full_frames = [cv2.imread(args.face)]
		fps = args.fps
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
				frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

			y1, y2, x1, x2 = args.crop
			if x2 == -1: x2 = frame.shape[1]
			if y2 == -1: y2 = frame.shape[0]

			frame = frame[y1:y2, x1:x2]
			full_frames.append(frame)

	print ("Number of frames available for inference: "+str(len(full_frames)))

	os.makedirs('temp', exist_ok=True)

	if not args.audio.endswith('.wav'):
		print('Extracting raw audio...')
		command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, 'temp/temp.wav')
		subprocess.call(command, shell=True)
		args.audio = 'temp/temp.wav'

	print('Extracting HuBERT features...')
	audio_tensor = extract_hubert_features(args.audio, args.hubert_path)
	print(f'Audio feature shape (B, 1024, T): {audio_tensor.shape}')

	if args.box[0] == -1:
		if not args.static:
			face_det_results = face_detect(full_frames)
		else:
			face_det_results = face_detect([full_frames[0]])
	else:
		print('Using the specified bounding box instead of face detection...')
		y1, y2, x1, x2 = args.box
		face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in full_frames]

	rep_chunks, frame_indices = create_rep_chunks(audio_tensor, len(full_frames))
	if len(rep_chunks) == 0:
		raise ValueError('Audio features too short to create chunks.')
	print(f'Number of audio chunks: {len(rep_chunks)}')

	model = load_onnx_model(args.checkpoint_path)
	print('Model loaded')

	args.img_size = adjust_img_size_from_onnx(model)
	print(f'Using img_size={args.img_size} based on model input.')

	onnx_inputs = model.get_inputs()
	input_names = [inp.name for inp in onnx_inputs]
	dtype = np.float16 if any('float16' in inp.type for inp in onnx_inputs) else np.float32

	gen = datagen(full_frames.copy(), face_det_results, rep_chunks, frame_indices, args.img_size, args.wav2lip_batch_size)

	frame_h, frame_w = full_frames[0].shape[:-1]
	out = cv2.VideoWriter('temp/result.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

	for img_batch, mel_batch, frames, coords in tqdm(gen, total=int(np.ceil(len(rep_chunks)/args.wav2lip_batch_size))):
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
			pred = model.run(['pred_face'], inputs)[0]
		except Exception:
			pred = model.run(None, inputs)[0]

		pred = pred.astype(np.float32).transpose(0, 2, 3, 1) * 255.

		for p, f, c in zip(pred, frames, coords):
			y1, y2, x1, x2 = c
			p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
			f[y1:y2, x1:x2] = p
			out.write(f)

	out.release()

	command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio, 'temp/result.avi', args.outfile)
	subprocess.call(command, shell=platform.system() != 'Windows')


if __name__ == '__main__':
	main()

