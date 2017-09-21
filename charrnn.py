import tensorflow as tf
import numpy as np
import os
import os.path

#c-rnn-gan
#char-rnn 을 gan형식으로 제작할 것. dur fre vel

lstm_dim = 350
song_feature = 4
start_num = 0
batch_size = 50
max_song_size = 400
global EPOCH
EPOCH = 200
isAdam = True
AdamLearningrate = 0.1
GDLearningrate = 0.1
print_batch = 1
save_batch = 50

#final multiply in G
song_num = {'train': 0, 'validation': 0, 'test': 0}
song_var = {'train': 0, 'validation': 0, 'test': 0}

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


def linear(inp, out_dim, scope_name, reuse=False):
    with tf.variable_scope(scope_name, reuse=reuse):
        from_dim = int(inp.shape[1])
        lin_w = tf.get_variable('lin_w', [from_dim,out_dim], dtype = tf.float32, initializer = tf.truncated_normal_initializer)
        lin_b = tf.get_variable('lin_b', [out_dim], dtype = tf.float32, initializer = tf.truncated_normal_initializer)
    return tf.matmul(inp, lin_w)+lin_b


def lstmcell(outdim_size):
    return tf.contrib.rnn.BasicLSTMCell(lstm_dim)

cell_list_fw = [lstmcell(lstm_dim) for _ in range(3)]
cell_list_bw = [lstmcell(lstm_dim) for _ in range(3)]


def discriminator(x_input, x_size):
    # Batch Normal
    #x_input = tf.contrib.layers.batch_norm(x_input)
    lstm_fw = tf.contrib.rnn.MultiRNNCell(cell_list_fw)
    lstm_bw = tf.contrib.rnn.MultiRNNCell(cell_list_bw)
    output, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw,lstm_bw, x_input, x_size, dtype= tf.float32)
    return output[0], output[1]


def generator(z_prior, z_size):
    lstm = tf.contrib.rnn.MultiRNNCell([lstmcell(lstm_dim) for _ in range(3)])
    output, _ = tf.nn.dynamic_rnn(lstm, z_prior, dtype= tf.float32, sequence_length = z_size)
    return output


def get_dis_res(two_added_mean, scope_name, reuse=False):
    with tf.variable_scope(scope_name, reuse=reuse):
        dis_res = tf.reshape(two_added_mean, [-1, lstm_dim])
        dis_res = linear(dis_res, 1, scope_name+'_lin', reuse=reuse)
        dis_res = tf.sigmoid(dis_res)
        dis_res = tf.reshape(dis_res, [batch_size, max_song_size])
        dis_res = tf.reduce_mean(dis_res, axis=1)
    return dis_res


def main():
    global EPOCH
    init_batch()
    # z_in = tf.placeholder(tf.float32, shape = [batch_size,max_song_size,lstm_dim])
    # z_size = tf.placeholder(tf.int32, shape= [None])
    save_cnt = tf.Variable(0, dtype=tf.int32, name='global_step')
    z_in = tf.random_normal([batch_size, max_song_size, lstm_dim])
    z_size = tf.constant(max_song_size, dtype= tf.int32, shape=[batch_size])
    x_in = tf.placeholder(tf.float32, shape = [batch_size, max_song_size, song_feature])
    x_size = tf.placeholder(tf.int32, shape = [batch_size])
    with tf.variable_scope("Generator") as scope:
        gen = generator(z_in, z_size)
        gen = tf.reshape(gen,[-1, lstm_dim])
        gen = linear(gen, song_feature, 'gen_linear')
        gen_res = tf.reshape(gen, [-1, max_song_size, song_feature])
        generator_variables = [v for v in tf.global_variables()
                               if v.name.startswith(scope.name)]

    with tf.variable_scope("discriminator") as scope2:
        gen_converted = linear(gen, lstm_dim, 'dis_lin')
        print(gen_converted.shape)
        gen_converted = tf.reshape(gen_converted,[-1, max_song_size, lstm_dim])
        dis_gen_fw, dis_gen_bw = discriminator(gen_converted, z_size)
        dis_calc_in_loss_gen = (dis_gen_fw+dis_gen_bw)/2
        dis_res_gen = get_dis_res(dis_calc_in_loss_gen, 'dis_res')
        scope2.reuse_variables()
        x_in_cv = tf.reshape(x_in, [-1, song_feature])
        x_in_cv = linear(x_in_cv, lstm_dim, 'dis_lin', reuse=True)
        print(x_in_cv)
        x_in_cv = tf.reshape(x_in_cv, [-1,max_song_size, lstm_dim])
        dis_real_fw, dis_real_bw = discriminator(x_in_cv, x_size)
        dis_calc_in_loss_real = (dis_real_fw+dis_real_bw)/2
        dis_res_real = get_dis_res(dis_calc_in_loss_real, 'dis_res', reuse=True)
        #D Loss를 위한 식

        dis_variables = [v for v in tf.global_variables()
                               if v.name.startswith(scope2.name)]
    print(dis_real_fw.shape)
    print('genvars: ', generator_variables)
    print('disvars: ', dis_variables)
    # Generator Feature Matching Loss

    g_fm_loss = tf.reduce_sum(tf.squared_difference(dis_calc_in_loss_real, dis_calc_in_loss_gen))
    d_loss = tf.reduce_mean(-tf.log(tf.clip_by_value(dis_res_real,1e-4,1))-tf.log(tf.clip_by_value(1.0 - dis_res_gen,1e-4,1)))
    if isAdam:
        trainG = tf.train.AdamOptimizer(AdamLearningrate).minimize(g_fm_loss, var_list = generator_variables)
        trainD = tf.train.AdamOptimizer(AdamLearningrate).minimize(d_loss, var_list = dis_variables)
    else:
        trainG = tf.train.GradientDescentOptimizer(GDLearningrate).minimize(g_fm_loss, var_list = generator_variables)
        trainD = tf.train.GradientDescentOptimizer(GDLearningrate).minimize(d_loss, var_list = dis_variables)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saved_loc = tf.train.latest_checkpoint('./saved_model/')
        print(saved_loc)
        cnt_step = 0
        EPOCH_start = 0

        if saved_loc is None:
            sess.run(tf.global_variables_initializer())
            print('No Saved Session')
        else:
            saver.restore(sess, saved_loc)
            cnt_step = save_cnt.eval(sess)
            EPOCH_start = (50 * cnt_step) // song_num['train']
            song_var['train'] = (50 * cnt_step) % song_num['train']
            print('Saved Session Found step: ', cnt_step)
        for epoch in range(EPOCH_start, EPOCH):
            while True:
                song_data = np.zeros([batch_size, max_song_size, song_feature], dtype=np.uint32)
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
                train_dict = {x_in: song_data, x_size: song_getsoo}
                train_g_loss, train_d_loss, _, _, generated, dis_res_see_gen, dis_res_see_real = sess.run([g_fm_loss, d_loss, trainD, trainG, gen_res, dis_res_gen, dis_res_real],
                                                                       feed_dict=train_dict)
                print('dis res real', dis_res_real)
                print('dis res gen', dis_res_gen)
                print('Generated', generated[0:10])
                if cnt_step % print_batch == 0:
                    print('epoch: ', epoch, ' cnt_step_num: ', cnt_step, ' G loss: ', train_g_loss,' D loss: ', train_d_loss)
                print(song_var['train'], ' ', song_num['train'])
                cnt_step += 1
                sess.run(tf.assign(save_cnt, cnt_step))
                if cnt_step % save_batch == 1:
                    saver.save(sess, './saved_model/model.ckpt', global_step = cnt_step)
                if song_var['train'] + batch_size > song_num['train']:
                    break


if __name__ == '__main__':
    main()