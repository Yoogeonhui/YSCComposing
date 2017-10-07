import midtotxt, npytomid

import numpy as np

file_name = './classical/chopin/chet2511.mid'

a = midtotxt.read_file(file_name)
save_array = np.asarray(a, dtype=np.uint32)
np.save('tmp.npy',save_array)

npytomid.save_data('tmp.npy', 'tmp.mid')