import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers


def make_nfm_model(cat_columns, val_nums, interact="multiply", merge="add"):
    assert interact in ("dot", "multiply")
    assert merge in ("add", "concat")

    cat_num = len(cat_columns)
    cat_field_input = []
    field_embedding = []

    # FM embedding
    for cat, val_n in zip(cat_columns, val_nums):
        input_ = layers.Input(shape=(1,))
        cat_field_input.append(input_)
        embed = layers.Embedding(val_n, 10, input_length=1, trainable=True)(input_)
        reshape = layers.Reshape((10,))(embed)
        field_embedding.append(reshape)
    
    # Bi-Interaction Pooling
    interaction_layers = []
    for i in range(cat_num):
        for j in range(i + 1, cat_num):
            embeds = [field_embedding[i], field_embedding[j]]
            prod = layers.dot(embeds, axes=-1) if interact == "dot" else layers.multiply(embeds)
            interaction_layers.append(prod)
            
    # merge layers
    merge_fn = layers.add if interact == 'add' else layers.concatenate
    embed_layer = merge_fn(interaction_layers)

    # dnn layers
    embed_layer = layers.Dense(64)(embed_layer)
    embed_layer = layers.BatchNormalization()(embed_layer)
    embed_layer = layers.Activation('relu')(embed_layer)
    embed_layer = layers.Dense(64)(embed_layer)
    embed_layer = layers.BatchNormalization()(embed_layer)
    embed_layer = layers.Activation('relu')(embed_layer)
    embed_layer = layers.Dense(1)(embed_layer)
    preds = layers.Activation('sigmoid')(embed_layer)
    
    opt = optimizers.Adam(0.001)
    model = models.Model(inputs=cat_field_input, outputs=preds)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
    return model
