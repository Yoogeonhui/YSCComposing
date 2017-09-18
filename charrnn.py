import tensorflow as tf
import numpy as np
import os
import os.path

#c-rnn-gan
#char-rnn 을 gan형식으로 제작할 것. dur fre vel

lstm_dim = 4
start_num = 0
batch_size = 50
max_song_size = 400
EPOCH = 200
isAdam = False
AdamLearningrate = 0.001
GDLearningrate = 0.05
print_batch = 1


#final multiply in G
weight_g = tf.get_variable("weight_g", shape= [lstm_dim],dtype= tf.float32, initializer=tf.truncated_normal_initializer)
bias_g = tf.get_variable("bias_g", shape = [lstm_dim],dtype = tf.float32, initializer= tf.constant_initializer([0.1]*lstm_dim))
song_num = song_var = {'train':0, 'validation': 0, 'test': 0}



def init_batch():
    song_num['train'] = len(os.listdir('./train'))
    song_num['validation'] = len(os.listdir('./validation'))
    song_num['test'] = len(os.listdir('./test'))


def get_batch(what='train', batchsize = 50):
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


def out_sigmoid(outdim_size):
    return tf.contrib.rnn.BasicLSTMCell(outdim_size, activation = tf.sigmoid)


cell_list_fw = [lstmcell(lstm_dim) for _ in range(2)]
cell_list_fw.append(out_sigmoid(lstm_dim))
cell_list_bw = [lstmcell(lstm_dim) for _ in range(2)]
cell_list_bw.append(out_sigmoid(lstm_dim))


def discriminator(x_input, x_size):
    # Batch Normal
    x_input = tf.contrib.layers.batch_norm(x_input)
    lstm_fw = tf.contrib.rnn.MultiRNNCell(cell_list_fw)
    lstm_bw = tf.contrib.rnn.MultiRNNCell(cell_list_bw)
    output, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw,lstm_bw, x_input, x_size, dtype= tf.float32)
    funcout = (output[0]+output[1])/2
    return funcout


def generator(z_prior, z_size):
    lstm = tf.contrib.rnn.MultiRNNCell([lstmcell(lstm_dim) for _ in range(3)])
    output, _ = tf.nn.dynamic_rnn(lstm, z_prior, dtype= tf.float32, sequence_length = z_size)

    output = output*weight_g + bias_g
    return output


def main():
    init_batch()
    z_in = tf.placeholder(tf.float32, shape = [None,None,lstm_dim])
    z_size = tf.placeholder(tf.int32, shape= [None])

    x_in = tf.placeholder(tf.float32, shape = [None, None, lstm_dim])
    x_size = tf.placeholder(tf.int32, shape = [None])
    with tf.variable_scope("Generator") as scope:
        gen = generator(z_in, z_size)
        generator_variables = [v for v in tf.global_variables()
                          if v.name.startswith(scope.name)]

    with tf.variable_scope("discriminator") as scope2:
        dis_gen = discriminator(gen, z_size)
        dis_real = discriminator(x_in, x_size)
        dis_variables = [v for v in tf.global_variables()
                               if v.name.startswith(scope2.name)]

    print('genvars: ', generator_variables)
    print('disvars: ', dis_variables)
    # Generator Feature Matching Loss
    g_fm_loss = tf.reduce_sum(tf.squared_difference(dis_real, dis_gen))
    d_loss = tf.reduce_mean(-tf.log(dis_real)-tf.log(1.0-dis_gen))

    if isAdam:
        trainG = tf.train.AdamOptimizer(AdamLearningrate).minimize(g_fm_loss, var_list = generator_variables)
        trainD = tf.train.AdamOptimizer(AdamLearningrate).minimize(d_loss, var_list = dis_variables)
    else:
        trainG = tf.train.GradientDescentOptimizer(GDLearningrate).minimize(g_fm_loss, var_list = generator_variables)
        trainD = tf.train.GradientDescentOptimizer(GDLearningrate).minimize(d_loss, var_list = dis_variables)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(EPOCH):
            cnt_batch = 0
            while True:
                song_data = np.zeros([batch_size, max_song_size, lstm_dim], dtype=np.uint32)
                song_getsoo = []
                read_batch = get_batch()
                for i, newbat in enumerate(read_batch):
                    tmp_max = max_song_size
                    if max_song_size > newbat.shape[0]:
                        tmp_max = newbat.shape[0]
                    song_data[i, :tmp_max, :] = newbat[:tmp_max, :]
                    song_getsoo.append(tmp_max)
                '''z_input = tf.random_uniform(shape = [batch_size, max_song_size, lstm_dim], minval = 0.0, maxval = 1.0,
                                            dtype=tf.float32)'''
                z_input = np.random.normal(size=[batch_size,max_song_size,lstm_dim])
                z_input_size = [max_song_size] * batch_size
                train_dict = {x_in: song_data, x_size: song_getsoo, z_in: z_input, z_size: z_input_size}
                train_g_loss, train_d_loss, _, _ = sess.run([g_fm_loss, d_loss, trainG, trainD], feed_dict=train_dict)
                if cnt_batch % print_batch == 0:
                    print('epoch: ', epoch, ' batch_num: ', cnt_batch, ' G loss: ', train_g_loss,' D loss: ', train_d_loss)
                cnt_batch+=1
                if song_var['train'] + batch_size > song_num['train']:
                    break



if __name__ == '__main__':
    main()