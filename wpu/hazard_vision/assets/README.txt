Add your test video as test.mp4 in this folder.

To create a minimal placeholder (requires OpenCV):
  python -c "import cv2; c=cv2.VideoWriter('test.mp4',cv2.VideoWriter_fourcc(*'mp4v'),10,(640,480)); c.write(__import__('numpy').zeros((480,640,3),dtype='uint8')); c.release()"

Or use any .mp4 file with --video_path.
