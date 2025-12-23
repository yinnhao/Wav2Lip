# python3 simplified_inference.py \
#   --checkpoint_path checkpoints/model.onnx \
#   --face example/raw_video_shot_011.mp4 \
#   --audio example/audio_shot_011.wav \
#   --hubert_path /root/paddlejob/workspace/env_run/zhuyinghao/checkpoints \
#   --outfile resultv2.mp4 \
#   --device cuda

  # python3 infer_v2.py \
  # --checkpoint_path checkpoints/model.onnx \
  # --face example/raw_video_shot_011.mp4 \
  # --audio example/audio_shot_011.wav \
  # --hubert_path /root/paddlejob/workspace/env_run/zhuyinghao/checkpoints \
  # --outfile result2.mp4 \
  # # --device cuda

  # python infer_v3.py \
  # --checkpoint_path checkpoints/model.onnx \
  # --hubert_path /root/paddlejob/workspace/env_run/zhuyinghao/checkpoints \
  # --face example/raw_video_shot_011.mp4 \
  # --audio example/audio_shot_011.wav \
  # --outfile result3.mp4

# python infer_v3.py \
# --checkpoint_path checkpoints/model.onnx \
# --hubert_path /root/paddlejob/workspace/env_run/zhuyinghao/checkpoints \
# --face example/raw_video_shot_003_25fps.mp4 \
# --audio example/audio_shot_003.wav \
# --outfile result_04.mp4 \

python infer_v3.py \
--checkpoint_path checkpoints/model.onnx \
--hubert_path /root/paddlejob/workspace/env_run/zhuyinghao/hubert-large-ls960-ft \
--face example/raw_video_shot_003_25fps.mp4 \
--audio example/audio_shot_003.wav \
--outfile result_05.mp4 \