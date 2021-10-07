import tensorflow as tf
from transformers import  TFBertModel, WarmUp
from tensorflow.keras.layers import  Dense, Dropout, Lambda
from tensorflow.keras.metrics import BinaryAccuracy, AUC
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from time import time
def define_tensor_callbacks(results_dir='',metric='val_loss',patience=3,mcp_folder='',mode="auto"):

    """CALLBACKS"""
    earlyStopping = EarlyStopping(monitor=metric,mode=mode,baseline=None,restore_best_weights=True,patience=patience)
    mcp_save = ModelCheckpoint(filepath=mcp_folder, save_best_only=True, monitor=metric, mode='min',save_weights_only=True)

    TensorBoard_log_dir = results_dir + "./logs_"
    log_dir = TensorBoard_log_dir + str(time())
    print('TENSORBOARD log_dir:', log_dir)
    tensorboard_callback = TensorBoard(log_dir=log_dir)
    return tensorboard_callback, mcp_save, earlyStopping

def create_encoder(trainable = True):

    model = TFBertModel.from_pretrained(
        'bert-base-uncased', trainable=trainable)  # TFBertModel.from_pretrained('bert-base-uncased') #
    max_seq_length = 512  # 156# 512
    input_ids = layers.Input(shape=(max_seq_length), dtype=tf.int32)
    attention_mask = layers.Input(shape=(max_seq_length), dtype=tf.int32)
    encoder_input = [input_ids, attention_mask]
    encoder = model(encoder_input)

    return encoder, encoder_input, model



def create_cross_encoder(learning_rate,train_data_size , batch,epochs, warmup_epochs=1, rateDacedy = 0.0001,dropout_rate=0.1,num_of_classes = 2,retrun_all=False, encoder_objs=[]):

    if len(encoder_objs) <2:
        encoder, encoder_input, model = create_encoder()
    else:
        print('use previous embeder objs')
        encoder, encoder_input, model = encoder_objs

    x = Lambda((lambda x: x[0][:, 0, :]))(encoder)
    droped_x = Dropout(dropout_rate)(x)
    if num_of_classes ==2:
        score = Dense(1, use_bias=False, activation='sigmoid')(droped_x)
        loss = 'binary_crossentropy'
    else:
        score = Dense(num_of_classes, use_bias=False, activation='sigmoid')(droped_x)
        loss = 'CategoricalCrossentropy'

    cross_encoder = Model(inputs=encoder_input, outputs=score, name='model1')

    """WARM UP AND LEARNING RATE"""
    print('use warmup')
    initial_learning_rate = learning_rate

    def scheduler(step):
        LearningRate = initial_learning_rate * 1 / (1 + rateDacedy * step)
        return LearningRate

    steps_per_epoch = int(train_data_size / batch)

    warm_up_Schedule = WarmUp(decay_schedule_fn=scheduler, initial_learning_rate=initial_learning_rate,
                              warmup_steps= steps_per_epoch*warmup_epochs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=warm_up_Schedule)

    cross_encoder.compile(optimizer=optimizer, loss=loss,
                          metrics=[BinaryAccuracy(), AUC(curve='PR',multi_label=True)])  # ,metrics=[CategoricalAccuracy()]
    cross_encoder.summary()
    if retrun_all:
        return cross_encoder, encoder, encoder_input, model
    else:

        return cross_encoder, model


