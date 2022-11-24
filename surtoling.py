import re
import whisper
from whisper.audio import SAMPLE_RATE
import stable_whisper
from nltk_contrib import textgrid
import sys

VOWELS = re.compile('[aeiouáàâéêíóôú]')
WORD = re.compile('\w[\w-]*\w|\w')

def readS1ts(textgridfile):
    grid = textgrid.TextGrid.load(textgridfile)
    for tier in grid.tiers:
        if tier.tier_name() == 'S1':
            S1 = tier.transcript.splitlines()
            break

    keyval_pat = re.compile('\s+(\w+)\s*=\s*(\S+)')
    keyval_dict = {'xmin': 0.0, 'xmax': 0.0, 'text': ''}
    S1ts = []
    for line in S1:
        m = keyval_pat.match(line)
        if m:
            if m.group(1) == 'text':
                keyval_dict['text'] = m.group(2)[1:-1]
            else:
                keyval_dict[m.group(1)] = float(m.group(2))
        elif keyval_dict['text'] != '':
            S1ts.append((keyval_dict['xmin'], keyval_dict['xmax']))
    if keyval_dict['text'] != '':
        S1ts.append((keyval_dict['xmin'], keyval_dict['xmax']))
    return S1ts

class Interval:
    def __init__(self, text, xmin, xmax):
        self.text = text
        self.xmin = xmin
        self.xmax = xmax

def loadS1audio(audio, S1_time_stamps):
    wavelet = whisper.load_audio(audio)
    i = 0
    for xmin, xmax in S1_time_stamps:
        wavelet[i:int(xmin*SAMPLE_RATE)+1] = 0
        i = int(xmax*SAMPLE_RATE) + 1
    wavelet[i:] = 0
    return wavelet

def word_strip(word):
    word_match = WORD.search(word.lower())
    if word_match:
        return word_match.group(0)
    return ''

def extract(audio, model_size='small'):
    def print_message(*message):
        print(*message, file=sys.stderr)

    if '.' not in audio:
        audio += '.wav'
    basename = audio[:audio.rfind('.')]
    S1ts = readS1ts(basename + '.TextGrid')
    print_message('Found', len(S1ts), 'S1 interval(s).')
    print_message('Loading audio...')
    wavelet = loadS1audio(audio, S1ts)
    print_message('Loading model...')
    model = whisper.load_model(model_size)
    stable_whisper.modify_model(model)
    print_message('Transcribing...')
    transcript = model.transcribe(wavelet, verbose=False, language='pt')
    word_transcript = stable_whisper.group_word_timestamps(transcript,
      combine_compound=True, min_dur=sys.float_info.epsilon)
    new_tier = []
    for word in word_transcript:
        w = word_strip(word['text'])
        if w != '' and w[-1] in 'aeo' and VOWELS.search(w[:-1]) and \
          w[-2] not in 'ãõ':
            new_tier.append(Interval(w, word['start'], word['end']))
