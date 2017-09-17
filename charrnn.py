import tensorflow as tf
import numpy as np
import os
import os.path
from random import shuffle
import mido

#c-rnn-gan
#char-rnn 을 gan형식으로 제작할 것. dur fre vel

lstm_dim = 4
global songs
start_num = 0
batch_size = 50

#final multiply
w = tf.Variable([lstm_dim], tf.float32)
song_num = song_var = {'train':0, 'validation': 0, 'test': 0}


def init_batch():
    song_num['train'] = len(os.listdir('./train'))
    song_num['validation'] = len(os.listdir('./validation'))
    song_num['test'] = len(os.listdir('./test'))


def get_batch(what='train', batch_size = 50):
    out_batch = []
    start_ptn = song_var[what]
    end_ptn = start_ptn + batch_size
    if end_ptn>song_num[what]:
        for i in range(start_ptn, song_num[what]):
            tmp_in = np.load('./'+what+'/'+str(i)+'.npy')
            out_batch.append(tmp_in)
        for i in range(0, end_ptn%song_num[what]):
            tmp_in = np.load('./'+what+'/'+str(i)+'.npy')
            out_batch.append(tmp_in)
    else:
        for i in range(start_ptn, end_ptn):
            tmp_in = np.load('./'+what+'/'+str(i)+'.npy')
            out_batch.append(tmp_in)

    song_var[what] = end_ptn%song_num[what]
    return out_batch



def lstmcell(outdim_size):
    return tf.contrib.rnn.BasicLSTMCell(outdim_size)

def outcell(outdim_size):
    return tf.contrib.rnn.BasicLSTMCell(outdim_size)

def disciriminator(x_input, x_size):
    # Batch Normal
    x_input = tf.contrib.layers.batch_norm(x_input)
    lstm = tf.contrib.rnn.MultiRNNCell([lstmcell(lstm_dim) for _ in range(4)])
    output, state = tf.nn.dynamic_rnn(lstm, x_input, dtype = tf.float32, sequence_length = x_size)
    return output[-1]

def generator(z_prior, z_size):
    lstm = tf.contrib.rnn.MultiRNNCell([ outcell(lstm_dim) for _ in range(3)])
    output, state = tf.nn.dynamic_rnn(lstm, z_prior, dtype= tf.float32, sequence_length = z_size)
    return output


def main():
    init_batch()
    z_in = tf.placeholder(tf.float32, shape = [None,None,lstm_dim])
    z_size = tf.placeholder(tf.int32, shape= [None])
    gen = generator(z_in, z_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tmp_batch = get_batch()
        batch = np.asarray(tmp_batch, dtype=np.uint32)
        print(batch)
        myin = tmp_batch[0:2]
        generated = sess.run(gen, feed_dict = {z_in: np.array([myin]), z_size: np.array([len(myin)])})
        print(generated)


if __name__ == '__main__':
    main()