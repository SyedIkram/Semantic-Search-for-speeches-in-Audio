import keras
from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, TimeDistributed, Activation, 
                          Bidirectional, SimpleRNN, GRU, LSTM)
from keras.layers import Lambda,Flatten,Reshape
from keras.models import load_model
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras import optimizers
import numpy as np

# ASR Model - Credits to: https://github.com/soheillll/Automatic-Speech-Recognition
def asr_model(input_dim, units, output_dim=29):

    input_data = Input(name='the_input', shape=(None, input_dim))
    
    bidirectional_rnn = Bidirectional(GRU(units, activation=None,return_sequences=True, implementation=2, name='bidir_rnn'))(input_data)
    batch_normalization = BatchNormalization(name = "batch_normalization_bidirectional_rnn")(bidirectional_rnn)
    activation = Activation('relu',name='activation1')(batch_normalization)

    bidirectional_rnn = Bidirectional(GRU(units, activation=None,return_sequences=True, implementation=2, name='bidir_rnn2'))(activation)
    batch_normalization = BatchNormalization(name = "bn_bidir_rnn_2")(bidirectional_rnn)
    activation = Activation('relu',name='activation2')(batch_normalization)

    time_dense = TimeDistributed(Dense(output_dim))(activation)
    y_pred = Activation('softmax', name='softmax')(time_dense)

    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


# Define our own model with the ASR model
class Model_1:

	def __init__(self,path=None,wt_paths = None,asr_path=None,mtype = 'long'):

		if path:
			if not wt_paths:
				raise Exception('Need weight parameters along with the model architecture to load the model')
			self.model = load_model(path)
			self.model.load_weights(wt_paths)
			self.model.layers[1].trainable = False # Set ASR model's layers as not trainable

		else:

			if not asr_path:
				raise Exception('Need pre-trained ASR model path')
			asr_model = asr_model(input_dim=161, units=200)
			asr_model.load_weights(asr_path_path)
			enc_model = Model(asr_model.input,asr_model.layers[-5].output,name='ASR-model')
			
			encoder_inputs = Input(shape=(None,161), name='Encoder-Input')
			asr_out = enc_model(encoder_inputs)

			states = Lambda(lambda x: x[:,-1,:])(asr_out) # get final state

			if mtype == 'short':
				x = Dense(300, activation='relu')(states)
				x = BatchNormalization(name='bn-1')(x)
				out = Dense(300)(x)

			else:
				x = Dense(500, activation='relu')(states)
				x = BatchNormalization(name='bn-1')(x)
				x = Dense(500)(x)
				x = BatchNormalization(name='bn-2')(x)
				out = Dense(300)(x)

			self.model = Model(encoder_inputs, out) # NEW MODEL

			# Freeze ASR model's layers
			self.model.layers[0].trainable = False
			self.model.layers[1].trainable = False
			self.model.layers[2].trainable = False
			for l in self.model.layers[1].layers:
				l.trainable = False

	def copmile(self,optimizer=optimizers.Nadam(lr=0.002),loss='cosine_proximity'):
		self.model.compile(optimizer=optimizers, loss=loss)

	def train(self,data,epochs,save_final=True,save_path = ''):

		script_name_base = 'model_'
		csv_logger = CSVLogger('{:}.log'.format(script_name_base))
		model_checkpoint = ModelCheckpoint('{:}.epoch{{epoch:02d}}-val{{val_loss:.5f}}.hdf5'.format(script_name_base),
                                   save_best_only=True)

		history = best_model.fit_generator(generator=data.gen_train(), steps_per_epoch=data.train_steps, \
			validation_data=data.gen_dev(), validation_steps=data.dev_steps, epochs=epochs, \
			callbacks=[csv_logger, model_checkpoint],verbose = True)

		if save_final:
			self.model.save(path+'/final_model.h5')
			self.model.save_weights(path+'/final_model_weights.h5')

		return history

	def predict(self,spectrogram):

		return self.model.predict(np.expand_dims(spectrogram, axis=0))


