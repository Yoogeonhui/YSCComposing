import mido
import os
from random import shuffle
import math

import numpy as np

pace_events= False

validation_num=0
test_num=0
train_num=0
song_data={'training':[], 'validation':[], 'test':[]}


tempos=[]

NUM_FEATURES_PER_TONE = 3

def get_rel_time(time, tmp):
    return (time-tempos[tmp]['time'])/tempos[tmp]['new_tick'] + tempos[tmp]['kara']

def read_file(root):
    TICKS_FROM_PREV_START      = 0
    LENGTH     = 1
    FREQ       = 2
    VELOCITY   = 3
    BEGIN_TICK = 0
    song_data = []
    global tempos

    midi = mido.MidiFile(root)
    # Tempo:
    ticks_per_quarter_note = midi.ticks_per_beat
    #print('Resoluton: {}'.format(ticks_per_quarter_note))

    #if debug == 'overfit': input_ticks_per_output_tick = 1.0
    input_ticks_per_output_tick = (ticks_per_quarter_note*120)/(384.0 * 45)
    # Multiply with output_ticks_pr_input_tick for output ticks.
    tempos = [{'time':0, 'new_tick': input_ticks_per_output_tick, 'kara': 0.0}]
    for track in midi.tracks:

        last_event_input_tick=0
        not_closed_notes = []
        for msg in track:
            if(msg.type=='set_tempo'):
                now_new_tick = ticks_per_quarter_note*mido.tempo2bpm(msg.tempo) / (384.0*45)
                tempos.append({'time':last_event_input_tick+msg.time,
                               'new_tick':now_new_tick,
                               'kara': tempos[len(tempos)-1]['kara']+((last_event_input_tick+msg.time-tempos[len(tempos)-1]['time'])/tempos[len(tempos)-1]['new_tick'])})
            last_event_input_tick += msg.time

    #print(tempos)
    for track in midi.tracks:
        tempo_var = 0
        last_event_input_tick=0
        for msg in track:
            if (msg.type == 'note_off') or \
                    (msg.type == 'note_on' and \
                                 msg.velocity == 0):
                now_abs = msg.time+last_event_input_tick
                while tempo_var+1 < len(tempos) and tempos[tempo_var+1]['time']<=now_abs:
                    tempo_var+=1
                    input_ticks_per_output_tick = tempos[tempo_var]['new_tick']
                retained_not_closed_notes = []

                for e in not_closed_notes:
                    if msg.note == e[FREQ]:
                        #start abs 절대값 빼고 rate처리 뒤 전 tempo 의 dueto 값 더하고, 마찬가지로
                        #END타이밍에도 이를 행함, 둘을 빼서 구할 것
                        finish = get_rel_time(now_abs, tempo_var)
                        e[LENGTH] = finish-e[BEGIN_TICK]
                        if(e[LENGTH]<0):
                            print('wtf', e[LENGTH], 'begin: ', e[BEGIN_TICK], ' finish: ', finish,' now abs: ', now_abs, 'tempo: ', tempos[tempo_var]['time'], 'rate', tempos[tempo_var]['new_tick'])
                            print(tempos)
                        song_data.append(e)
                    else:
                        retained_not_closed_notes.append(e)
                #if len(not_closed_notes) == len(retained_not_closed_notes):
                #  print('Warning. NoteOffEvent, but len(not_closed_notes)({}) == len(retained_not_closed_notes)({})'.format(len(not_closed_notes), len(retained_not_closed_notes)))
                #  print('NoteOff: {}'.format(tone_to_freq(event.data[0])))
                #  print('not closed: {}'.format(not_closed_notes))
                not_closed_notes = retained_not_closed_notes
            elif msg.type == 'note_on':
                now_abs = msg.time+last_event_input_tick
                while tempo_var+1 < len(tempos) and tempos[tempo_var+1]['time']<=now_abs:
                    tempo_var+=1
                    input_ticks_per_output_tick = tempos[tempo_var]['new_tick']
                #BEGIN TICK 또한 절대값 빼고 rate처리 후 더해서 저장
                begin_tick = get_rel_time(now_abs, tempo_var)
                note = [0.0]*(NUM_FEATURES_PER_TONE+1)
                note[FREQ]       = msg.note
                note[VELOCITY]   = msg.velocity
                note[BEGIN_TICK] = begin_tick
                not_closed_notes.append(note)
                #not_closed_notes.append([0.0, tone_to_freq(event.data[0]), velocity, begin_tick, event.channel])
            #print(msg.time)
            last_event_input_tick += msg.time
        for e in not_closed_notes:
            #print('Warning: found no NoteOffEvent for this note. Will close it. {}'.format(e))
            e[LENGTH] = ticks_per_quarter_note/input_ticks_per_output_tick
            song_data.append(e)
    song_data.sort(key=lambda e: e[BEGIN_TICK])

    return song_data

getsoo = 0

for dirpath, dirnames, filenames in os.walk("."):
    for i, filename in enumerate([f for f in filenames if f.endswith(".mid")],0):
        try:
            getsoo+=1
            if(getsoo%50==0):
                print(getsoo,"개")
            abs_path = os.path.abspath(dirpath+'\\'+filename)
            asdf = read_file(abs_path)
            save_root="./"
            if(i%10 == 0):
                save_root+="validation/"
                sfn =str(validation_num)
                validation_num+=1

            elif(i%30==1):
                save_root+="test/"
                sfn =str(test_num)
                test_num+=1
            else:
                save_root+="train/"
                sfn =str(train_num)
                train_num+=1

            save_array = np.asarray(asdf, dtype = np.uint32)
            np.save(save_root+sfn, save_array)
        except:
            print('Error')


'''
asdf = read_file('./classical/mozart/k452.mid')
save_array = np.asarray(asdf, dtype = np.uint32)
np.save('k452.npy', save_array)
'''