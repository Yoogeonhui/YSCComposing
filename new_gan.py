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
max_song_size = 150
EPOCH = 200
isAdam = True
AdamLearningrate = {'G': 0.0005, 'D': 0.0005, 'P': 0.01}
GDLearningrate = {'G': 0.1, 'D': 0.1, 'P': 1e-7}
print_batch = 5
save_batch = 50
DEBUG = True
reg_scale = 0.5


#final multiply in G
song_num = {'train': 0, 'validation': 0, 'test': 0}
song_var = {'train': 0, 'validation': 0, 'test': 0}

def init_batch():
    song_num['train'] = len(os.listdir('./train'))
    song_num['validation'] = len(os.listdir('./validation'))
    song_num['test'] = len(os.listdir('./test'))


def get_batch(what='train'):
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

def get_batch_matrix(what = 'train'):
    song_data = np.zeros([batch_size, max_song_size, song_feature], dtype=np.float32)
    song_getsoo = []
    read_batch = get_batch(what)
    last_note = np.zeros([batch_size, 1, song_feature], dtype=np.uint32)
    for i, newbat in enumerate(read_batch):
        tmp_max = max_song_size
        if max_song_size + 1 > newbat.shape[0]:
            tmp_max = newbat.shape[0]
        else:
            last_note[i, 0, :] = newbat[tmp_max, :]
            last_note[i, 0, 0] -= newbat[tmp_max-1, 0]
        song_data[i, :tmp_max, :] = newbat[:tmp_max, :]
        song_getsoo.append(tmp_max)
        song_data[i,1:tmp_max,0] = song_data[i,1:tmp_max,0] - song_data[i,0:tmp_max-1,0]
    return song_data, song_getsoo, last_note

def linear(inp, out_dim, scope_name, reuse=False):
    with tf.variable_scope(scope_name, reuse=reuse):
        from_dim = int(inp.shape[1])
        lin_w = tf.get_variable('lin_w', [from_dim,out_dim], dtype = tf.float32, initializer = tf.truncated_normal_initializer)
        lin_b = tf.get_variable('lin_b', [out_dim], dtype = tf.float32, initializer = tf.truncated_normal_initializer)
    return tf.matmul(inp, lin_w)+lin_b


def lstmcell(outdim_size = lstm_dim):
    return tf.contrib.rnn.BasicLSTMCell(outdim_size)


def dropout_lstm_cell(outdim_size = lstm_dim, keep_prob = 0.6):
    cell = lstmcell(outdim_size)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = keep_prob)
    return cell


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
    lstm = tf.contrib.rnn.MultiRNNCell([lstmcell() for _ in range(3)])
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


init_batch()
save_cnt = tf.Variable(0, dtype=tf.int32, name='global_step')
z_in = tf.placeholder(tf.float32, shape = [batch_size, max_song_size, song_feature])
z_size = tf.placeholder(tf.int32, shape = [batch_size])
x_in = tf.placeholder(tf.float32, shape = [batch_size, max_song_size, song_feature])
x_size = tf.placeholder(tf.int32, shape = [batch_size])


with tf.variable_scope("Generator") as scope:
    scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=reg_scale))
    gen = tf.reshape(z_in,[-1, song_feature])
    gen = linear(gen, lstm_dim, 'gen_linear_1')
    gen = tf.contrib.layers.batch_norm(gen)
    gen = tf.reshape(gen, [-1, max_song_size, lstm_dim])
    gen = generator(gen, z_size)
    gen = tf.reshape(gen,[-1, lstm_dim])
    gen = tf.nn.relu(linear(gen, song_feature, 'gen_linear_2'))
    gen_res = tf.reshape(gen, [-1, max_song_size, song_feature])
    generator_variables = [v for v in tf.global_variables()
                           if v.name.startswith(scope.name)]


with tf.variable_scope("discriminator") as scope2:
    scope2.set_regularizer(tf.contrib.layers.l2_regularizer(scale=reg_scale))
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


predict_real = tf.round(dis_res_real)
predict_gen = tf.round(dis_res_gen)
accuracy_res = (tf.reduce_mean(predict_real) + tf.reduce_mean(1-predict_gen))/2
print('genvars: ', generator_variables)
print('disvars: ', dis_variables)
# Generator Feature Matching Loss
'''
front_sliced = tf.slice(gen_res,[0,0,0], [batch_size,max_song_size-1, song_feature])
back_sliced = tf.slice(gen_res,[0,1,0], [batch_size, max_song_size-1, song_feature])
back_diff= tf.squared_difference(front_sliced, back_sliced)
'''
g_fm_loss = 15 * tf.reduce_mean(tf.squared_difference(dis_calc_in_loss_real, dis_calc_in_loss_gen))

#+ (1e-5) * tf.reduce_mean(1/tf.clip_by_value(back_diff, 1e-5, 1e+5))
g_loss = tf.reduce_mean(tf.log(tf.clip_by_value(1.0 - dis_res_gen, 1e-20, 1)))
d_loss = tf.reduce_mean(-tf.log(tf.clip_by_value(dis_res_real, 1e-20, 1))-tf.log(tf.clip_by_value(1.0 - dis_res_gen, 1e-20, 1)))
train_AdamG = tf.train.AdamOptimizer(AdamLearningrate['G']).minimize(g_fm_loss, var_list = generator_variables)
train_AdamD = tf.train.AdamOptimizer(AdamLearningrate['D']).minimize(d_loss, var_list = dis_variables)

trainG = tf.train.GradientDescentOptimizer(GDLearningrate['G']).minimize(g_fm_loss, var_list = generator_variables)
trainD = tf.train.GradientDescentOptimizer(GDLearningrate['D']).minimize(d_loss, var_list = dis_variables)

pretrain_loss = tf.reduce_mean(tf.squared_difference(gen_res, x_in))

train_pre_G = tf.train.GradientDescentOptimizer(GDLearningrate['P']).minimize(pretrain_loss, var_list = generator_variables)
train_pre_A = tf.train.AdamOptimizer(AdamLearningrate['P']).minimize(pretrain_loss, var_list = generator_variables)

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
        EPOCH_start = (batch_size * cnt_step) // song_num['train']
        song_var['train'] = (batch_size * cnt_step) % song_num['train']
        print('Saved Session Found step: ', cnt_step)

    for epoch in range(EPOCH_start, 6):
        while True:
            song_data, song_getsoo, song_last_note = get_batch_matrix()
            song_data=np.concatenate((song_data,song_last_note), axis=1)


            #Pretraining
            pretrain_dict = {z_in: song_data[:,:-1,:], z_size: song_getsoo, x_in: song_data[:,1:,:], x_size: song_getsoo}
            if isAdam:
                _, p_loss, generated_song = sess.run([train_pre_A, pretrain_loss, gen_res], feed_dict = pretrain_dict)
                np.savetxt('generated_song.txt', generated_song[25], fmt="%d")
                np.savetxt('origin_song.txt', song_data[25], fmt="%d")
                if cnt_step % print_batch == 0:
                    print('P: ', p_loss)
            else:
                _, p_loss = sess.run([train_pre_G, pretrain_loss], feed_dict = pretrain_dict)


            if song_var['train'] + batch_size > song_num['train']:
                saver.save(sess, './saved_model/model.ckpt', global_step = cnt_step)
                break

    EPOCH_start = 6
    print('Pretrain G completed')
    if isAdam:
        train_tens_G = train_AdamG
        train_tens_D = train_AdamD
    else:
        train_tens_G = trainG
        train_tens_D = trainD

    for epoch in range(EPOCH_start, 7):
        while True:
            song_data, song_getsoo, _ = get_batch_matrix()
            train_dict = {z_in: np.random.normal(size=(batch_size, max_song_size, song_feature)), z_size: song_getsoo, x_in: song_data, x_size: song_getsoo}
            accuracy, train_d_loss, _ = sess.run([accuracy_res, d_loss, train_tens_D], feed_dict = train_dict)
            print('acc', accuracy, 'd_loss', train_d_loss)
            if song_var['train'] + batch_size > song_num['train']:
                saver.save(sess, './saved_model/model.ckpt', global_step = cnt_step)
                break

    print('Pretrain D Completed')
    for epoch in range(EPOCH_start, EPOCH):
        while True:
            song_data, song_getsoo, _ = get_batch_matrix()

            train_dict = {z_in: np.random.normal(size=(batch_size, max_song_size, song_feature)), z_size: song_getsoo, x_in: song_data, x_size: song_getsoo}
            # Freezing을 위한 Accuracy 구하기
            accuracy = sess.run(accuracy_res, feed_dict=train_dict)
            print('accuracy : ', accuracy)
            '''
            if accuracy>=0.8:
            else:
                train_g_loss, train_d_loss, _, _, generated, dis_res_see_gen, dis_res_see_real = sess.run([g_fm_loss, d_loss, train_tens_D, train_tens_G, gen_res, dis_res_gen, dis_res_real],
            '                                                                                        feed_dict=train_dict)
            '''
            train_g_loss, train_d_loss = sess.run(
                [g_fm_loss, d_loss],
                feed_dict=train_dict)
            if train_g_loss<=0.7*train_d_loss:
                _ = sess.run(train_tens_D, feed_dict = train_dict)
            elif train_d_loss<=0.7*train_g_loss:
                _ = sess.run(train_tens_G, feed_dict= train_dict)
            else:
                _, _ = sess.run([train_tens_D, train_tens_G], feed_dict= train_dict)
            generated = sess.run(gen_res, feed_dict = train_dict)
            generated = generated.astype(int)
            print(generated.shape)
            np.savetxt('generated_song.txt', generated[25], fmt="%d")
            print('Generated', generated[25][-1])
            print('epoch: ', epoch, ' cnt_step_num: ', cnt_step, ' G loss: ', train_g_loss,' D loss: ', train_d_loss)
            print(song_var['train'], ' ', song_num['train'])
            cnt_step += 1
            sess.run(tf.assign(save_cnt, cnt_step))
            if cnt_step % save_batch == 0:
                saver.save(sess, './saved_model/model.ckpt', global_step = cnt_step)
            if song_var['train'] + batch_size > song_num['train']:
                break