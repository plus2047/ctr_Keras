import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers

def make_lr_model(cat_columns, val_nums):
    cat_num = len(cat_columns)
    cat_field_input = []
    lr_embedding = []
    
    for cat, val_n in zip(cat_columns, val_nums):
        input_ = layers.Input(shape=(1,), name=cat)
        cat_field_input.append(input_)
        embed = layers.Embedding(val_n, 1, input_length=1, trainable=True)(input_)
        reshape = layers.Reshape((1,))(embed)
        lr_embedding.append(reshape)
    
    lr_layer = layers.add(lr_embedding)
    preds = layers.Activation('sigmoid')(lr_layer)
    opt = optimizers.Adam(0.001)
    model = models.Model(inputs=cat_field_input, outputs=preds)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
    
    return model