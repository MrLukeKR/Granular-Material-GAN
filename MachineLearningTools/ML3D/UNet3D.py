import numpy as np

from keras import Input, Model
from keras.optimizers import Adam
from keras.layers import MaxPooling3D, Conv3D, BatchNormalization, UpSampling3D, Add as AddLayers


# 3D1 conversion of ML2D image segmentation:
# https://github.com/MrLukeKR/unsupervised-image-segmentation-by-WNet-with-NormalizedCut/blob/master/src/WNet_bright.py


def create_unet(vox_res, image_channels, segments, enable_normalisation):

    model_enc = list()

    # Width * Height * depth * channels
    input = Input(shape=(vox_res, vox_res, vox_res, image_channels))

    max_layers = 5

    print("DOWN-SAMPLING")
    model = input

    # Encoding ---------------------------------------------------------------------------------------------------------
    #   \ Down-sample 64, 128, 256, 512, 1024 ----------------------------------------------

    for i in range(0, max_layers):
        if i == max_layers - 1:
            print("BASE LAYER")
        else:
            print("\tLayer " + str(i + 1) + ":")

        factor = 2 ** i
        filters = vox_res * factor
        new_res = np.floor(np.uint(vox_res / factor))

        # 1 Convolutional unit: Conv, Conv, (Save Residual Copy), Max Pool
        for r in range(2):
            model = Conv3D(filters=filters, kernel_size=3, padding="same", activation='relu',
                           input_shape=(new_res, new_res, new_res))(model)
            if enable_normalisation:
                model = BatchNormalization()(model)

        # ---------------------------------------------------------------
        print("\t\tFactor: " + str(factor) + "\tFilters: " + str(filters) + "\tImage Resolution: [" + str(
            new_res) + "^3]\tModel shape: " + str(model.shape))

        if i != max_layers-1:
            model_enc.append(model)
            model = MaxPooling3D(pool_size=(2, 2, 2))(model)

    #   / Up-sample 512, 256, 128, 64 ------------------------------------------------------

    print("UP-SAMPLING")

    for i in range(max_layers - 1, 0, -1):
        print("\tLayer " + str(i) + ":")
        factor = 2 ** (i - 1)
        filters = vox_res * factor
        new_res = np.floor(np.uint(vox_res / factor))

        # Residual/skip layer
        up_sample = UpSampling3D()(model)
        if enable_normalisation:
            up_sample = BatchNormalization()(up_sample)
        up_convolve = Conv3D(filters, kernel_size=(2, 2, 2), padding="same")(up_sample)
        if enable_normalisation:
            up_convolve = BatchNormalization()(up_convolve)

        model = AddLayers()([model_enc.pop(), up_convolve])
        if enable_normalisation:
            model = BatchNormalization()(model)

        for r in range(2):
            model = Conv3D(filters, kernel_size=(3, 3, 3), padding="same", activation='relu',
                           input_shape=(new_res, new_res, new_res))(model)
            if enable_normalisation:
                model = BatchNormalization()(model)

        print("\t\tFactor: " + str(factor) + "\tFilters: " + str(filters) + "\tImage Resolution: [" + str(new_res)
          + "^3]\tModel shape: " + str(model.shape))
#        model = Dropout(rate=0.5)(model)

    print("OUTPUT LAYER")

    # model = Conv3D(image_channels, kernel_size=(1, 1, 1), padding="same", activation='relu',
    #              input_shape=(vox_res, vox_res, vox_res))(model)

    model = Conv3D(segments, kernel_size=(1, 1, 1), padding="same", activation='relu', input_shape=(vox_res, vox_res, vox_res))(model)
    # model = Softmax()(model)

    model = Conv3D(segments, kernel_size=(1, 1, 1), padding="same", activation="softmax", input_shape=(vox_res, vox_res, vox_res))(model)

    print("\t\tFactor: 0\tFilters: " + str(image_channels) + "\tImage Resolution: [" + str(new_res) +
          "^3]\tModel shape: " + str(model.shape))

    # ------------------------------------------------------------------------------------------------------------------

    predictions = model

    unet = Model(inputs=input, outputs=predictions)
    unet.compile(optimizer=Adam(0.1), loss='binary_crossentropy', metrics=['accuracy'])

    unet.summary()

    return unet

