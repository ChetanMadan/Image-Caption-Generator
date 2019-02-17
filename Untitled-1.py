import keras
from os import listdir
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import string
from numpy import array
from keras.applications import VGG16

from keras.models import Model
from keras.utils import to_categorical, plot_model

from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.sequence import pad_sequences

from keras.layers.merge import add

def extract_features(dir):
    model = VGG16()
    
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs = model.layers[-1].output)
    features = dict()
    
    for name in listdir(dir):
        filename = dir+'/'+name
        image = load_img(filename, target_size=(224,224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1],image.shape[2]))
        image= preprocess_input(image)
        feature = model.predict(image)
        
        image_id=name.split('.')[0]
        features[image_id]=feature
    return features
directory = 'Flickr_Data/Flickr Images/'
features = extract_features(directory)


dump(features, open('features.pkl','wb'))

filename = 'Flickr_Data/Flickr Text Data/Flickr8k.token.txt'
# load descriptions
doc = load_doc(filename)


#%%
def clean_text(desc):
    table = str.maketrans('','',string.punctuation)
    for key, desc_list in desc.items():
        for i in range(len(desc_list)):
            d = desc_list[i]
            d=d.split()
            d = [j.lower() for j in d]
            d = [j.translate(table) for j in d]
            d = [j for j in d if len(j)>1]
            d = [j for j in d if j.isalpha()]
            desc_list=''.join(d)


#%%
def to_vocab(desc):
    all_desc=set()
    for key in desc.keys():
        [all_desc.update(d.split()) for d in desc[key]]
    return all_desc



#%%
def save_desc(desc, file):
    lines = list()
    for key, desc_list in desc.items():
        for desc in desc_list:
            lines.append(key+' '+desc)
            
    data = '\n'.join(lines)
    
    with open(file,'w') as outfile:
        outfile.write(data)
        
        


#%%
# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text
 
# extract descriptions for images
def load_descriptions(doc):
	mapping = dict()
	# process lines
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		if len(line) < 2:
			continue
		# take the first token as the image id, the rest as the description
		image_id, image_desc = tokens[0], tokens[1:]
		# remove filename from image id
		image_id = image_id.split('.')[0]
		# convert description tokens back to string
		image_desc = ' '.join(image_desc)
		# create the list if needed
		if image_id not in mapping:
			mapping[image_id] = list()
		# store description
		mapping[image_id].append(image_desc)
	return mapping
 
def clean_descriptions(descriptions):
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for key, desc_list in descriptions.items():
		for i in range(len(desc_list)):
			desc = desc_list[i]
			# tokenize
			desc = desc.split()
			# convert to lower case
			desc = [word.lower() for word in desc]
			# remove punctuation from each token
			desc = [w.translate(table) for w in desc]
			# remove hanging 's' and 'a'
			desc = [word for word in desc if len(word)>1]
			# remove tokens with numbers in them
			desc = [word for word in desc if word.isalpha()]
			# store as string
			desc_list[i] =  ' '.join(desc)
            

def to_vocabulary(descriptions):
	# build a list of all description strings
	all_desc = set()
	for key in descriptions.keys():
		[all_desc.update(d.split()) for d in descriptions[key]]
	return all_desc
 
# save descriptions to file, one per line
def save_descriptions(descriptions, filename):
	lines = list()
	for key, desc_list in descriptions.items():
		for desc in desc_list:
			lines.append(key + ' ' + desc)
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()
 
filename = 'Flickr_Data/Flickr Text Data/Flickr8k.token.txt'
# load descriptions
doc = load_doc(filename)
# parse descriptions
descriptions = load_descriptions(doc)
print('Loaded: %d ' % len(descriptions))
# clean descriptions
clean_descriptions(descriptions)
# summarize vocabulary
vocabulary = to_vocabulary(descriptions)
print('Vocabulary Size: %d' % len(vocabulary))
# save to file
save_descriptions(descriptions, 'descriptions.txt')


#%%
def load_data(file):
    
    with open(file,'r') as infile:
        txt = infile.read()
    return txt


def load_set(file):
    doc=load_data(file)
    dataset=list()
    
    for line in doc.split('\n'):
        if len(line)<1:
            continue
        identifier = line.split('.')[0]
        dataset.append(identifier)
        
        
    return set(dataset)


#%%
def load_clean(file, dataset):
    # load document
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        # split line by white space
        tokens = line.split()
        # split id from description
        image_id, image_desc = tokens[0], tokens[1:]
        # skip images not in the set
        if image_id in dataset:
        # create list
            if image_id not in descriptions:
                descriptions[image_id] = list()
            # wrap description in tokens
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            # store
            descriptions[image_id].append(desc)
    return descriptions


#%%
def load_image_features(file, dataset):
    feat = load(open(file, 'b'))
    features = {k:feat[i] for i in dataset}
    return features


#%%
from pickle import load


#%%
filename = 'Flickr_Data/Flickr Text Data/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# photo features
train_features = load_photo_features('features.pkl', train)
print('Photos: train=%d' % len(train_features))


#%%
# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text
 
# load a pre-defined list of photo identifiers
def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
	# process line by line
	for line in doc.split('\n'):
		# skip empty lines
		if len(line) < 1:
			continue
		# get the image identifier
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return set(dataset)
 
# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
	# load document
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# skip images not in the set
		if image_id in dataset:
			# create list
			if image_id not in descriptions:
				descriptions[image_id] = list()
			# wrap description in tokens
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			# store
			descriptions[image_id].append(desc)
	return descriptions
 
# load photo features
def load_photo_features(filename, dataset):
	# load all features
	all_features = load(open(filename, 'rb'))
	# filter features
	features = {k: all_features[k] for k in dataset}
	return features
 
# load training dataset (6K)
filename = 'Flickr_Data/Flickr Text Data//Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# photo features
train_features = load_photo_features('features.pkl', train)


#%%
def to_line(descriptions):
    all_desc=list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
        
    return all_desc


#%%
def create_token(descriptions):
    lines = to_line(descriptions)
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


#%%
tokenizer = create_token(train_descriptions)


#%%
vocab_size = len(tokenizer.word_index)+1


#%%
def create_sequences(tokenizer, max_len, descriptions, photos):
    X1, X2, y= list(),list(),list()
    
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            seq = tokenizer.texts_to_sequences([desc])[0]
            for i in range(1, len(seq)):
                
                in_seq, out_seq=seq[:i], seq[i]
                
                in_seq=pad_sequences([in_seq], max_len)[0]
                out_seq=to_categorical([out_seq], num_classes=vocab_size)[0]
                X1.append(photos[key][0])
                X2.append(in_seq)
                y.append(out_seq)
        return array(X1), array(X2), array(y)
    


#%%
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc


#%%
def max_len(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)


#%%
filename = 'Flickr_Data/Flickr Text Data/Flickr_8k.devImages.txt'
test=load_set(filename) 
test_descriptions = load_clean_descriptions('descriptions.txt', test)
max_length = max_len(train_descriptions)
test_features = load_photo_features('features.pkl', test)
X1train, X2train, ytrain = create_sequences(tokenizer, max_length, train_descriptions, train_features)
X1test, X2test, ytest = create_sequences(tokenizer, max_length, test_descriptions, test_features)


#%%
from keras.layers import Input, Dense, Dropout, LSTM, Embedding


#%%
def define_model(vocab_size, max_len):
    inp1 = Input(shape = (4096,))
    f1=Dropout(0.5)(inp1)
    f2=Dense(256, activation='relu')(f1)
    
    inp2=Input(shape=(max_len,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inp2)
    se2=Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    
    decoder = add([f2,se3])
    d2 = Dense(256, activation='relu')(decoder)
    
    out = Dense(vocab_size, activation='softmax')(d2)
    
    model = Model(inputs = [inp1,inp2], outputs = out)
    model.compile(loss= 'categorical_crossentropy', optimizer='adam')
    
    print(model.summary())
    
    #plot_model(model, to_file='model.png', show_shapes=True)
    return model


#%%
from keras.callbacks import ModelCheckpoint


#%%
filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss={val_loss:.3f}.h5'

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only = True, mode='min')


#%%
model = define_model(vocab_size, max_length)


#%%
model.fit([X1train, X2train], ytrain, epochs=100, verbose=2, callbacks=[checkpoint], validation_data=([X1test, X2test], ytest))


#%%
epochs = 20
steps = len(train_descriptions)
for i in range(epochs):
	# create the data generator
	generator = data_generator(train_descriptions, train_features, tokenizer, max_length)
	# fit for one epoch
	model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
	# save model
	model.save('model_' + str(i) + '.h5')


#%%
tokenizer = load(open('tokenizer.pkl', 'rb'))
# pre-define the max sequence length (from training)
max_length = 34


#%%
'model-ep020-loss3.010-val_loss=8.057.h5'


