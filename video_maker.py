import cv2
import os

FRAME_RATE = 10

image_folder = 'C:\\Users\\navea\\Desktop\\video_test'
video_name = image_folder+'\\video.avi'

images = []
for i in range(len(os.listdir(image_folder))+1):
    images.append('{}.png'.format(i))

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, FRAME_RATE, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()