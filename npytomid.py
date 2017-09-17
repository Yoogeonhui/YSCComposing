import mido, math
import numpy as np

def tone_to_freq(tone):
    """
      returns the frequency of a tone.
      formulas from
        * https://en.wikipedia.org/wiki/MIDI_Tuning_Standard
        * https://en.wikipedia.org/wiki/Cent_(music)
    """
    return math.pow(2, ((float(tone)-69.0)/12.0)) * 440.0

def cents_to_pitchwheel_units(cents):
    return int(40.96*(float(cents)))

def freq_to_tone(freq):
    """
      returns a dict d where
      d['tone'] is the base tone in midi standard
      d['cents'] is the cents to make the tone into the exact-ish frequency provided.
                 multiply this with 8192 to get the midi pitch level.
      formulas from
        * https://en.wikipedia.org/wiki/MIDI_Tuning_Standard
        * https://en.wikipedia.org/wiki/Cent_(music)
    """
    if freq <= 0.0:
        return None
    float_tone = (69.0+12*math.log(float(freq)/440.0, 2))
    int_tone = int(float_tone)
    cents = int(1200*math.log(float(freq)/tone_to_freq(int_tone), 2))
    return {'tone': int_tone, 'cents': cents}

def get_midi_pattern( song_data):
    TICKS_FROM_PREV_START=0
    NUM_FEATURES_PER_TONE = 3
    LENGTH=1
    FREQ       = 2
    VELOCITY   = 3
    tones_per_cell=1
    output_ticks_per_quarter_note = 384.0

    """
    get_midi_pattern takes a song in internal representation
    (a tensor of dimensions [songlength, self.num_song_features]).
    the three values are length, frequency, velocity.
    if velocity of a frame is zero, no midi event will be
    triggered at that frame.
    returns the midi_pattern.
    Can be used with filename == None. Then nothing is saved, but only returned.
    """
    print('song_data[0:10]: {}'.format(song_data[0:40]))


    #
    # Interpreting the midi pattern.
    # A pattern has a list of tracks
    # (midi.Track()).
    # Each track is a list of events:
    #   * midi.events.SetTempoEvent: tick, data([int, int, int])
    #     (The three ints are really three bytes representing one integer.)
    #   * midi.events.TimeSignatureEvent: tick, data([int, int, int, int])
    #     (ignored)
    #   * midi.events.KeySignatureEvent: tick, data([int, int])
    #     (ignored)
    #   * midi.events.MarkerEvent: tick, text, data
    #   * midi.events.PortEvent: tick(int), data
    #   * midi.events.TrackNameEvent: tick(int), text(string), data([ints])
    #   * midi.events.ProgramChangeEvent: tick, channel, data
    #   * midi.events.ControlChangeEvent: tick, channel, data
    #   * midi.events.PitchWheelEvent: tick, data(two bytes, 14 bits)
    #
    #   * midi.events.NoteOnEvent:  tick(int), channel(int), data([int,int]))
    #     - data[0] is the note (0-127)
    #     - data[1] is the velocity.
    #     - if velocity is 0, this is equivalent of a midi.NoteOffEvent
    #   * midi.events.NoteOffEvent: tick(int), channel(int), data([int,int]))
    #
    #   * midi.events.EndOfTrackEvent: tick(int), data()
    #
    # Ticks are relative.
    #
    # Tempo are in microseconds/quarter note.
    #
    # This interpretation was done after reading
    # http://electronicmusic.wikia.com/wiki/Velocity
    # http://faydoc.tripod.com/formats/mid.htm
    # http://www.lastrayofhope.co.uk/2009/12/23/midi-delta-time-ticks-to-seconds/2/
    # and looking at some files. It will hopefully be enough
    # for the use in this project.
    #
    # This approach means that we do not currently support
    #   tempo change events.
    #

    # Tempo:
    # Multiply with output_ticks_pr_input_tick for output ticks.
    midi_pattern = mido.MidiFile(ticks_per_beat = 384)

    cur_track = mido.MidiTrack()
    midi_pattern.tracks.append(cur_track)
    cur_track.append(mido.MetaMessage('set_tempo', time = 0, tempo = mido.bpm2tempo(45)))
    future_events = {}
    last_event_tick = 0

    ticks_to_this_tone = 0.0
    song_events_absolute_ticks = []
    for frame in song_data:
        abs_tick_note_beginning = frame[TICKS_FROM_PREV_START]
        tick_len           = int(frame[LENGTH])
        freq               = int(frame[FREQ])
        velocity           = min(int(round(frame[VELOCITY])),127)
        print('abs',abs_tick_note_beginning,'ticklen',tick_len,'\n')
        #print('tick_len: {}, freq: {}, velocity: {}, ticks_from_prev_start: {}'.format(tick_len, freq, velocity, frame[TICKS_FROM_PREV_START]))
        #d = freq_to_tone(freq)
        #print('d: {}'.format(d))
        if velocity > 0 and tick_len > 0:
            # range-check with preserved tone, changed one octave:
            tone = freq
            while tone < 0:   tone += 12
            while tone > 127: tone -= 12
            #pitch_wheel = cents_to_pitchwheel_units(d['cents'])
            #print('tick_len: {}, freq: {}, tone: {}, pitch_wheel: {}, velocity: {}'.format(tick_len, freq, tone, pitch_wheel, velocity))
            #if pitch_wheel != 0:
            #midi.events.PitchWheelEvent(tick=int(ticks_to_this_tone),
            #                                            pitch=pitch_wheel)
            song_events_absolute_ticks.append((abs_tick_note_beginning,
                                               mido.Message(
                                                   'note_on',
                                                   time=tick_len,
                                                   velocity=velocity,
                                                   note=tone)))
            song_events_absolute_ticks.append((abs_tick_note_beginning+tick_len,
                                               mido.Message(
                                                   'note_off',
                                                   time=0,
                                                   velocity=0,
                                                   note=tone)))
    song_events_absolute_ticks.sort(key=lambda e: e[0])
    abs_tick_note_beginning = 0.0

    print(song_events_absolute_ticks)

    for abs_tick,event in song_events_absolute_ticks:
        rel_tick = abs_tick-abs_tick_note_beginning
        event.time = int(round(rel_tick))
        cur_track.append(event)
        abs_tick_note_beginning=abs_tick

    #cur_track.append(mido.MetaMessage('end_of_track', time=int(output_ticks_per_quarter_note)))
    #print 'Printing midi track.'
    #print midi_pattern
    return midi_pattern

def save_data(loc, save_loc):
    myin = np.load(loc)
    gm = myin.tolist()
    midi_pattern = get_midi_pattern(gm)
    midi_pattern.save(save_loc)


save_data('./train/199.npy', 'tmp.mid')