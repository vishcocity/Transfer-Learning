#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import, division, print_function, unicode_literals


# In[2]:


try:
  
  get_ipython().run_line_magic('tensorflow_version', '2.x')
except Exception:
  pass
import tensorflow as tf

import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle


# In[3]:


# Download caption annotation files
annotation_folder = '/annotations/'
#if os.path.exists(os.path.abspath('.') + annotation_folder):
annotation_zip = tf.keras.utils.get_file('captions.zip',
                                          cache_subdir=os.path.abspath('.'),
                                          origin = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                          extract = True)
annotation_file = os.path.dirname(annotation_zip)+'/annotations/captions_train2014.json'
os.remove(annotation_zip)

# Download image files
image_folder = '/train2014/'
if not os.path.exists(os.path.abspath('.') + image_folder):
  image_zip = tf.keras.utils.get_file('train2014.zip',
                                      cache_subdir=os.path.abspath('.'),
                                      origin = 'http://images.cocodataset.org/zips/train2014.zip',
                                      extract = True)
  PATH = os.path.dirname(image_zip) + image_folder
  os.remove(image_zip)
else:
  PATH = os.path.abspath('.') + image_folder


# In[4]:


print(PATH)


# In[5]:


files_downloaded = set(os.listdir("/home/jupyter/train2014/"))


# In[6]:


len(files_downloaded)


# In[7]:


# Read the json file
with open(annotation_file, 'r') as f:
    annotations = json.load(f)

# Store captions and image names in vectors
all_captions = []
all_img_name_vector = []

for annot in annotations['annotations']:
  caption = '<start> ' + annot['caption'] + ' <end>'
  image_id = annot['image_id']  
  img_filename = 'COCO_train2014_' + '%012d.jpg' % (image_id)
  full_coco_image_path = PATH + img_filename
  if img_filename in files_downloaded:
    all_img_name_vector.append(full_coco_image_path)
    all_captions.append(caption)


# Shuffle captions and image_names together
# Set a random state
train_captions, img_name_vector = shuffle(all_captions,
                                          all_img_name_vector,
                                          random_state=1)

# Select the first 30000 captions from the shuffled set
num_examples = 30000
train_captions = train_captions[:num_examples]
img_name_vector = img_name_vector[:num_examples]


# In[8]:


len(train_captions), len(all_captions)


# In[9]:


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img, image_path


# In[10]:


image_model = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False,
                                                weights='imagenet')


new_input = image_model.input
hidden_layer = image_model.layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)


# In[11]:


hidden_layer.shape 


# In[12]:


# Get unique images
encode_train = sorted(set(img_name_vector))

# Feel free to change batch_size according to your system configuration
image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
image_dataset = image_dataset.map(
  load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)


# In[13]:


for img, path in image_dataset:
  batch_features = image_features_extract_model(img)
  batch_features = tf.reshape(batch_features,
                              (batch_features.shape[0], -1, batch_features.shape[3]))
  
  for bf, p in zip(batch_features, path):
    path_of_feature = p.numpy().decode("utf-8")
    np.save(path_of_feature, bf.numpy())


# In[14]:


# Find the maximum length of any caption in our dataset
def calc_max_length(tensor):
    return max(len(t) for t in tensor)


# In[15]:


# Choose the top 5000 words from the vocabulary
top_k = 5000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                  oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(train_captions)
train_seqs = tokenizer.texts_to_sequences(train_captions)


# In[16]:


tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'


# In[17]:


# Create the tokenized vectors
train_seqs = tokenizer.texts_to_sequences(train_captions)


# In[18]:


# Pad each vector to the max_length of the captions
# If you do not provide a max_length value, pad_sequences calculates it automatically
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')


# In[19]:


# Calculates the max_length, which is used to store the attention weights
max_length = calc_max_length(train_seqs)


# In[20]:


# Create training and validation sets using an 80-20 split
img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector,
                                                                    cap_vector,
                                                                    test_size=0.2,
                                                                    random_state=0)


# In[21]:


len(img_name_train), len(cap_train), len(img_name_val), len(cap_val)


# In[22]:


# Feel free to change these parameters according to your system's configuration

BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512
vocab_size = top_k + 1
num_steps = len(img_name_train) // BATCH_SIZE
# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
features_shape = 2048
attention_features_shape = 64


# In[23]:


# Load the numpy files
def map_func(img_name, cap):
  img_tensor = np.load(img_name.decode('utf-8')+'.npy')
  return img_tensor, cap


# In[24]:


dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

# Use map to load the numpy files in parallel
dataset = dataset.map(lambda item1, item2: tf.numpy_function(
          map_func, [item1, item2], [tf.float32, tf.int32]),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Shuffle and batch
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


# In[25]:


class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, features, hidden):
    # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

    # hidden shape == (batch_size, hidden_size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
    hidden_with_time_axis = tf.expand_dims(hidden, 1)

    # score shape == (batch_size, 64, hidden_size)
    score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

    # attention_weights shape == (batch_size, 64, 1)
    # you get 1 at the last axis because you are applying score to self.V
    attention_weights = tf.nn.softmax(self.V(score), axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights


# In[26]:


class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


# In[27]:


class RNN_Decoder(tf.keras.Model):
  def __init__(self, embedding_dim, units, vocab_size):
    super(RNN_Decoder, self).__init__()
    self.units = units

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc1 = tf.keras.layers.Dense(self.units)
    self.fc2 = tf.keras.layers.Dense(vocab_size)

    self.attention = BahdanauAttention(self.units)

  def call(self, x, features, hidden):
    # defining attention as a separate model
    context_vector, attention_weights = self.attention(features, hidden)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # shape == (batch_size, max_length, hidden_size)
    x = self.fc1(output)

    # x shape == (batch_size * max_length, hidden_size)
    x = tf.reshape(x, (-1, x.shape[2]))

    # output shape == (batch_size * max_length, vocab)
    x = self.fc2(x)

    return x, state, attention_weights

  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units))


# In[28]:


encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)


# In[29]:


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)


# In[30]:


checkpoint_path = "./checkpoints_mobilenetv2/train"
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer = optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)


# In[31]:


start_epoch = 0
if ckpt_manager.latest_checkpoint:
  start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
  # restoring the latest checkpoint in checkpoint_path
  ckpt.restore(ckpt_manager.latest_checkpoint)


# In[32]:


# adding this in a separate cell because if you run the training cell
# many times, the loss_plot array will be reset
loss_plot = []


# In[33]:


@tf.function
def train_step(img_tensor, target):
  loss = 0

  # initializing the hidden state for each batch
  # because the captions are not related from image to image
  hidden = decoder.reset_state(batch_size=target.shape[0])

  dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

  with tf.GradientTape() as tape:
      features = encoder(img_tensor)

      for i in range(1, target.shape[1]):
          # passing the features through the decoder
          predictions, hidden, _ = decoder(dec_input, features, hidden)

          loss += loss_function(target[:, i], predictions)

          # using teacher forcing
          dec_input = tf.expand_dims(target[:, i], 1)

  total_loss = (loss / int(target.shape[1]))

  trainable_variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, trainable_variables)

  optimizer.apply_gradients(zip(gradients, trainable_variables))

  return loss, total_loss


# In[34]:


EPOCHS = 20

for epoch in range(start_epoch, EPOCHS):
    start = time.time()
    total_loss = 0

    for (batch, (img_tensor, target)) in enumerate(dataset):
        batch_loss, t_loss = train_step(img_tensor, target)
        total_loss += t_loss

        if batch % 100 == 0:
            print ('Epoch {} Batch {} Loss {:.4f}'.format(
              epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
    # storing the epoch end loss value to plot later
    loss_plot.append(total_loss / num_steps)

    if epoch % 5 == 0:
      ckpt_manager.save()

    print ('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                         total_loss/num_steps))
    print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


# In[35]:


plt.plot(loss_plot)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Plot')
plt.show()


# In[74]:


def evaluate(image):
    attention_plot = np.zeros((max_length, 49))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot


# In[75]:


def plot_attention(image, result, attention_plot):
    temp_image = np.array(Image.open(image))
    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(len_result//2, len_result//2, l+1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()


# In[76]:


# captions on the validation set
rid = np.random.randint(0, len(img_name_val))
image = img_name_val[rid]
real_caption = ' '.join([tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]])
result, attention_plot = evaluate(image)

print ('Real Caption:', real_caption)
print ('Prediction Caption:', ' '.join(result))
plot_attention(image, result, attention_plot)


# In[77]:


plt.imshow(np.array(Image.open(image)))


# In[78]:


# captions on the validation set
rid = np.random.randint(0, len(img_name_val))
image = img_name_val[rid]
real_caption = ' '.join([tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]])
result, attention_plot = evaluate(image)

print ('Real Caption:', real_caption)
print ('Prediction Caption:', ' '.join(result))
plot_attention(image, result, attention_plot)


# In[79]:


plt.imshow(np.array(Image.open(image)))


# In[80]:


# captions on the validation set
rid = np.random.randint(0, len(img_name_val))
image = img_name_val[rid]
real_caption = ' '.join([tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]])
result, attention_plot = evaluate(image)

print ('Real Caption:', real_caption)
print ('Prediction Caption:', ' '.join(result))
plot_attention(image, result, attention_plot)


# In[81]:


plt.imshow(np.array(Image.open(image)))


# In[82]:


#Predicting captions for all images
from tqdm.notebook import tqdm


# In[ ]:


# captions on the validation set
#rid = np.random.randint(0, len(img_name_val)) # -- Just picking 1 image
rid=None
predicted_captions = {}
real_caption = None
for rid,image in tqdm(enumerate(img_name_val)):
    real_caption = ' '.join([tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]])
    result, attention_plot = evaluate(image)
    predicted_captions[image.split("/")[-1]] = result

print ('Real Caption:', real_caption)
print ('Prediction Caption:', ' '.join(result))
#plot_attention(image, result, attention_plot)


# In[ ]:


with open('predicted_caption_MobileNet.json','w') as j:
    json.dump(predicted_captions,j)


# In[ ]:


#Loading Predicted captions and creating reference and candidate for BLEU Score
with open("predicted_caption_MobileNet.json", 'r') as f:
    predicted_captions = json.load(f)


# In[ ]:


def modify_name(x):
    return x.split("/")[-1]


# In[ ]:


bleu_score_list = {}
img_set = set([modify_name(img_name) for img_name in img_name_val])
for annot in annotations["annotations"]:
    img_name = 'COCO_train2014_' + '%012d.jpg' % (annot["image_id"])
    if img_name in img_set:
        if img_name in bleu_score_list:
            bleu_score_list[img_name]["real"].append(annot["caption"].split())
        else:
            bleu_score_list[img_name] = {"predicted":predicted_captions[img_name]
                                            ,"real":[annot["caption"].split()]}


# In[ ]:


with open('predicted_caption_MobileNet_with_real_captions.json','w') as j:
    json.dump(bleu_score_list,j)


# In[ ]:


#Calculating BLEU Scores
import json
from nltk.translate.bleu_score import sentence_bleu


# In[ ]:


bleu_scores_for_images = {}


# In[ ]:


with open("predicted_caption_MobileNet_with_real_captions.json", 'r') as f:
    bleu_scores_for_images = json.load(f)


# In[ ]:


## BLEU1
for img, captions in bleu_score_list.items():
    reference = captions["real"]
    candidate = [ x for x in captions["predicted"] if x not in ["<start>","<end>","<unk>"] ]
    
    bleu_scores_for_images[img] = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))


# In[ ]:


avg_bleu = 0
for k,v in bleu_scores_for_images.items():
    avg_bleu=avg_bleu+v
avg_bleu = avg_bleu/len(bleu_scores_for_images)


# In[ ]:


avg_bleu


# In[ ]:


## BLEU2


# In[ ]:


with open("predicted_caption_MobileNet_with_real_captions.json", 'r') as f:
    bleu_scores_for_images = json.load(f)


# In[ ]:


for img, captions in bleu_score_list.items():
    reference = captions["real"]
    candidate = [ x for x in captions["predicted"] if x not in ["<start>","<end>","<unk>"] ]
    
    bleu_scores_for_images[img] = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))


# In[ ]:


avg_bleu = 0
for k,v in bleu_scores_for_images.items():
    avg_bleu=avg_bleu+v
avg_bleu2 = avg_bleu/len(bleu_scores_for_images)


# In[ ]:


avg_bleu2


# In[ ]:


## BLEU3


# In[ ]:


for img, captions in bleu_score_list.items():
    reference = captions["real"]
    candidate = [ x for x in captions["predicted"] if x not in ["<start>","<end>","<unk>"] ]
    
    bleu_scores_for_images[img] = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))


# In[ ]:


avg_bleu = 0
for k,v in bleu_scores_for_images.items():
    avg_bleu=avg_bleu+v
avg_bleu3 = avg_bleu/len(bleu_scores_for_images)


# In[ ]:


avg_bleu3


# In[ ]:


## BLEU4


# In[ ]:


for img, captions in bleu_score_list.items():
    reference = captions["real"]
    candidate = [ x for x in captions["predicted"] if x not in ["<start>","<end>","<unk>"] ]
    
    bleu_scores_for_images[img] = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))


# In[ ]:


avg_bleu = 0
for k,v in bleu_scores_for_images.items():
    avg_bleu=avg_bleu+v
avg_bleu4 = avg_bleu/len(bleu_scores_for_images)


# In[ ]:


avg_bleu4


# In[ ]:





# In[ ]:





# In[ ]:




