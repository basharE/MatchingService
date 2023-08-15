import cv2


def calculate_similarity(img1_, img2_, algo):
    # Read images
    img1 = cv2.imread(img1_, 0)
    img2 = cv2.imread(img2_, 0)

    # Initialize feature detector
    if algo == 0:
        algorithm = cv2.xfeatures2d.SIFT_create()
    elif algo == 1:
        algorithm = cv2.xfeatures2d.SURF_create()
    elif algo == 2:
        algorithm = cv2.ORB_create(nfeatures=100)

    # Detect key points and compute descriptors
    _, des1 = algorithm.detectAndCompute(img1, None)
    _, des2 = algorithm.detectAndCompute(img2, None)

    # Match key points using BFMatcher
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.95 * n.distance:
            good.append(m)

    score = 1 - (len(good) / len(matches))

    return score
