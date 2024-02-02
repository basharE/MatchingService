import cv2


def calculate_similarity(img1_, img2_):
    # Initialize feature detector
    algorithm = cv2.ORB_create(nfeatures=2500)

    # Detect key points and compute descriptors
    _, des1 = extract_descriptors(img1_, algorithm)
    _, des2 = extract_descriptors(img2_, algorithm)

    # Match key points using BFMatcher
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    return get_similarity(des1, des2, matches)


def initialize_orb_detector():
    # Initialize feature detector
    algorithm = cv2.ORB_create(nfeatures=2500)
    return algorithm


def extract_descriptors(img1_, algorithm):
    img1 = cv2.imread(img1_, 0)
    return algorithm.detectAndCompute(img1, None)


def extract_descriptors_(img1_, algorithm):
    _, des = extract_descriptors(img1_, algorithm)
    return des


def get_similarity(des1, des2, matches):
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.99 * n.distance:
            good.append(m)

    return 1 - (len(good) / min(len(des1), len(des2)))


def get_similarity_(des1, des2):
    # Match key points using BFMatcher
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.99 * n.distance:
            good.append(m)

    return 1 - (len(good) / min(len(des1), len(des2)))
