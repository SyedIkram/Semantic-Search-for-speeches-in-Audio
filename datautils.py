import os
import numpy as np
import torchaudio
import pandas as pd
from tqdm import tqdm
from feature_extractor import FeatureExtractor

def spectrogrm_preprocessor():
    fft_length = int(0.001 * 20 * 16000)
    spectrogram_preprocess = torchaudio.transforms.Spectrogram(n_fft = fft_length, normalized=True)
    return spectrogram_preprocess

def get_data(download=True):

	if download:
		os.system('mkdir data')

	train_data = torchaudio.datasets.LIBRISPEECH('data',url="train-clean-100", \
                                                folder_in_archive='LibriSpeech', download=download)
	dev_data = torchaudio.datasets.LIBRISPEECH('data',url="dev-clean", \
                                                folder_in_archive='LibriSpeech', download=download)
	test_data = torchaudio.datasets.LIBRISPEECH('data',url="test-clean", \
                                                folder_in_archive='LibriSpeech', download=download)

	return train_data, dev_data, test_data

def get_transcripts(data):
	pass

def create_map(inp,out):

    df = pd.DataFrame(columns=['location', 'utterance'])

    for group in tqdm(os.listdir(inp)):
        if group.startswith('.'):
            continue
        speaker_path = os.path.join(inp, group)
        for speaker in os.listdir(speaker_path):
            if speaker.startswith('.'):
                continue
            labels_file = os.path.join(speaker_path, speaker,'{}-{}.trans.txt'.format(group, speaker))
            for line in open(labels_file):
                split = line.strip().split()
                file_id = split[0]
                label = ' '.join(split[1:]).lower()
                audio_file = os.path.join(speaker_path, speaker,file_id) + '.flac'

                df = df.append({'location': audio_file, 'utterance': label}, ignore_index=True)

    df.to_csv(out)
    print('CSV file saved at: ', out)

class LibriSpeech():

    def __init__(self,train=None,dev=None,test=None,se=None, batch_size=3,max_train_len = 5000):

          self.train = pd.read_csv(train)[:max_train_len]
          self.dev = pd.read_csv(dev)[:int(max_train_len*0.1)]
          self.test = pd.read_csv(test)[:int(max_train_len*0.1)]
          self.featurize = FeatureExtractor()
          self.sent_emb = se
          self.batch_size = batch_size

          self.train_steps = len(self.train)//self.batch_size
          self.dev_steps = len(self.dev)//self.batch_size

          self.train_idx = 0
          self.dev_idx = 0

    # Function to generate features and labels batch wise to feed the model
    def gen_train(self):

      while True:

        items = self.train['location'].tolist()[self.train_idx : self.train_idx+self.batch_size]
        utterances = self.train['utterance'].tolist()[self.train_idx : self.train_idx+self.batch_size]

        self.train_idx += self.batch_size
        if self.train_idx>=len(self.train):
          self.train_idx = 0

        feats = [self.featurize(i) for i in items]
        labels_embs = [self.sent_emb(l) for l in utterances]

        outs = np.zeros([len(labels_embs),300])

        max_length = max([feats[i].shape[0] for i in range(0, len(feats))])
        x_data = np.zeros([len(feats), max_length, 161])

        for i in range(0, len(labels_embs)):
              feat = feats[i]
              x_data[i, :feat.shape[0], :] = feat
              outs[i,:] = labels_embs[i][0][:]

        yield x_data,outs

    # Similar to gen_train except this is for dev set which is used as validation set
    def gen_dev(self):

      while True:

        items = self.dev['location'].tolist()[self.dev_idx : self.dev_idx+self.batch_size]
        utterances = self.dev['utterance'].tolist()[self.dev_idx : self.dev_idx+self.batch_size]

        self.dev_idx += self.batch_size
        if self.dev_idx>=len(self.dev):
          self.dev_idx = 0

        feats = [self.featurize(i) for i in items]
        labels_embs = [self.sent_emb(l) for l in utterances]

        outs = np.zeros([len(labels_embs),300])

        max_length = max([feats[i].shape[0] for i in range(0, len(feats))])
        x_data = np.zeros([len(feats), max_length, 161])

        for i in range(0, len(labels_embs)):
              feat = feats[i]
              x_data[i, :feat.shape[0], :] = feat
              outs[i,:] = labels_embs[i][0][:]

        yield x_data,outs

