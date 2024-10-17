import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

images_dir = Path(__file__).resolve().parent / "images"
image_paths_list = list(images_dir.glob("*ghost.png"))
ghost_images = [cv2.imread(img) for img in image_paths_list]
search_image = cv2.imread(images_dir / "lab7.png")

sift = cv2.SIFT_create()

def detect_ghosts(ghost_image, filled_image, found_image):
    found = True
    while(found):
        keypoints_1, descriptors_1 = sift.detectAndCompute(ghost_image, None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(filled_image, None)

        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
        matches = bf.knnMatch(descriptors_1,descriptors_2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) > 10: 
            src_pts = np.float32([keypoints_1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints_2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            h, w, _  = ghost_image.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)

            dst = cv2.perspectiveTransform(pts, H)
            found_image = cv2.drawContours(found_image, [np.int32(dst)], -1, (0, 255, 0), 3, cv2.LINE_AA)
            filled_image = cv2.drawContours(filled_image, [np.int32(dst)], -1, (0, 255, 0), cv2.FILLED)

            # pair = np.concatenate((found_image, filled_image), axis=0)
            # pair = cv2.resize(pair , (960, 540))
            # cv2.imshow('Pair', pair)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        else: break
    return found_image, filled_image

found_ghosts = search_image.copy()
filled_ghosts = search_image.copy()
for ghost_image in ghost_images:
    print("ghost")
    found_ghosts, filled_ghosts = detect_ghosts(ghost_image, filled_ghosts, found_ghosts)
    found_ghosts, filled_ghosts = detect_ghosts(cv2.flip(ghost_image, 1), filled_ghosts, found_ghosts)

plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(found_ghosts, cv2.COLOR_BGR2RGB))
plt.show()
    