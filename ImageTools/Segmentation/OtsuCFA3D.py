def segment_image(image):
    if len(image.shape) != 3:
        raise Exception("This segmentation method only accepts three-dimensional images (volumes). "
                        "The shape given is " + image.shape)


