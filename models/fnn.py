import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers

def make_fnn_model(cols, val_nums):
    cat_num = len(cols)
    cat_field_input = []
    field_embedding = []
    
    for cat, val_n in zip(cols, val_nums):
        input_ = layers.Input(shape=(1,))
        cat_field_input.append(input_)
        x_embed = layers.Embedding(val_n, 10, input_length=1, trainable=True)(input_)
        x_reshape = layers.Reshape((10,))(x_embed)
        field_embedding.append(x_reshape)

    #######ffm layer##########
    embed_layer = layers.concatenate(field_embedding, axis=-1)

    #######dnn layer##########
    embed_layer = layers.Dense(64)(embed_layer)
    embed_layer = layers.BatchNormalization()(embed_layer)
    embed_layer = layers.Activation('relu')(embed_layer)
    embed_layer = layers.Dense(64)(embed_layer)
    embed_layer = layers.BatchNormalization()(embed_layer)
    embed_layer = layers.Activation('relu')(embed_layer)
    embed_layer = layers.Dense(1)(embed_layer)
    
    ########linear layer##########
    lr_layer = embed_layer
    preds = layers.Activation('sigmoid')(lr_layer)
    
    opt = optimizers.Adam(0.001)
    model = models.Model(inputs=cat_field_input, outputs=preds)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
    return model
