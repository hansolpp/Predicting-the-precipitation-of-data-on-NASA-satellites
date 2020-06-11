#from keras import Model
from keras.activations import relu
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Conv2DTranspose, MaxPooling2D, BatchNormalization, \
    Activation, concatenate, GlobalAveragePooling2D
#from tensorflow_core.python.keras import Input


def resnet_model(input_layer):

    inputs = input_layer

    bn = BatchNormalization()(inputs)
    conv0 = Conv2D(256, kernel_size=1, strides=1, padding='same', activation='relu')(bn)

    bn = BatchNormalization()(conv0)
    conv = Conv2D(128, kernel_size=2, strides=1, padding='same', activation='relu')(bn)
    concat = concatenate([conv0, conv], axis=3)

    bn = BatchNormalization()(concat)
    conv = Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu')(bn)
    concat = concatenate([concat, conv], axis=3)

    for i in range(5):
        bn = BatchNormalization()(concat)
        conv = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(bn)
        concat = concatenate([concat, conv], axis=3)

    bn = BatchNormalization()(concat)
    outputs = Conv2D(1, kernel_size=1, strides=1, padding='same', activation='relu')(bn)

    return outputs


def unet_model(input_layer, start_neurons):

    input_layer = BatchNormalization()(input_layer)
    # 40 x 40 -> 20 x 20
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(input_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(conv1)
    pool1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D((2, 2))(pool1)
    pool1 = Dropout(0.2)(pool1)

    # 20 x 20 -> 10 x 10
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(conv2)
    pool2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D((2, 2))(pool2)
    pool2 = Dropout(0.2)(pool2)

    # 10 x 10
    convm = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(pool2)
    convm = BatchNormalization()(convm)

    # 10 x 10 -> 20 x 20
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same", kernel_initializer='he_normal')(convm)
    deconv2 = BatchNormalization()(deconv2)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = BatchNormalization()(uconv2) #
    uconv2 = Dropout(0.2)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(uconv2)
    uconv2 = BatchNormalization()(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(uconv2)
    uconv2 = BatchNormalization()(uconv2)

    # 20 x 20 -> 40 x 40
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same", kernel_initializer='he_normal')(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = BatchNormalization()(uconv1) #
    uconv1 = Dropout(0.2)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(uconv1)
    uconv1 = BatchNormalization()(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(uconv1)
    uconv1 = BatchNormalization()(uconv1)
    #uconv1 = Dropout(0.25)(uconv1)
    output_layer = Conv2D(1, (1, 1), padding="same", activation='relu', kernel_initializer='he_normal')(uconv1)

    return output_layer


def train2_unet_model(input_layer, start_neurons):

    inputs = BatchNormalization()(input_layer)

    # 40 x 40 -> 20 x 20
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(inputs)
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D((2, 2))(pool1)
    pool1 = Dropout(0.25)(pool1)

    # 20 x 20 -> 10 x 10
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D((2, 2))(pool2)
    pool2 = Dropout(0.25)(pool2)

    # 10 x 10
    convm = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)

    # 10 x 10 -> 20 x 20
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(0.25)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
    uconv2 = BatchNormalization()(uconv2)

    # 20 x 20 -> 40 x 40
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(0.25)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    uconv1 = BatchNormalization()(uconv1)
    uconv1 = Dropout(0.25)(uconv1)
    output_layer = Conv2D(1, (1, 1), padding="same", activation='relu')(uconv1)

    return output_layer

def train2_unet2_model(input_layer, start_neurons):

    input_layer = BatchNormalization()(input_layer)
    # 40 x 40 -> 20 x 20
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(input_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(conv1)
    pool1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D((2, 2))(pool1)
    pool1 = Dropout(0.2)(pool1)

    # 20 x 20 -> 10 x 10
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(conv2)
    pool2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D((2, 2))(pool2)
    pool2 = Dropout(0.2)(pool2)

    # 10 x 10
    convm = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(pool2)
    convm = BatchNormalization()(convm)

    # 10 x 10 -> 20 x 20
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same", kernel_initializer='he_normal')(convm)
    deconv2 = BatchNormalization()(deconv2)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = BatchNormalization()(uconv2) #
    uconv2 = Dropout(0.2)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(uconv2)
    uconv2 = BatchNormalization()(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(uconv2)
    uconv2 = BatchNormalization()(uconv2)

    # 20 x 20 -> 40 x 40
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same", kernel_initializer='he_normal')(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = BatchNormalization()(uconv1) #
    uconv1 = Dropout(0.2)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(uconv1)
    uconv1 = BatchNormalization()(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(uconv1)
    uconv1 = BatchNormalization()(uconv1)
    #uconv1 = Dropout(0.25)(uconv1)
    output_layer = Conv2D(1, (1, 1), padding="same", activation='relu', kernel_initializer='he_normal')(uconv1)

    return output_layer

