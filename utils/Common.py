def remove_extension(filename):
    if filename.endswith(".mp4"):
        return filename[:-4]
    return filename
