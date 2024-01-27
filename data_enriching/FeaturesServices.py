import cv2


def calculate_similarity(img1_, img2_):
    # Initialize feature detector
    algorithm = cv2.ORB_create(nfeatures=500)

    # Detect key points and compute descriptors
    _, des1 = extract_descriptors(img1_, algorithm)
    _, des2 = extract_descriptors(img2_, algorithm)

    # Match key points using BFMatcher
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    return 1 - get_similarity(des1, des2, matches)


def extract_descriptors(img1_, algorithm):
    img1 = cv2.imread(img1_, 0)
    return algorithm.detectAndCompute(img1, None)


def get_similarity(des1, des2, matches):
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.99 * n.distance:
            good.append(m)

    return len(good) / min(len(des1), len(des2))
