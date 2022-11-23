import re
import whisper
from whisper.audio import SAMPLE_RATE
import stable_whisper
from nltk_contrib import textgrid
from sys import float_info

VOWELS = re.compile('[aeiouáàâéêíóôú]')

def readS1ts(textgridfile):
    grid = textgrid.TextGrid.load(textgridfile)
    for tier in grid.tiers:
        if tier.tier_name() == 'S1':
            S1 = tier
            break
    #TODO: Read the xmin and xmax in each interval in S1.transcript
    return [(S1.xmin, S1.xmax)]

class Interval:
    def __init__(self, name, xmin, xmax):
        self.name = name
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

def extract(audio, model_size='small'):
    basename = audio[:audio.rfind('.')]
    S1ts = readS1ts(basename + '.TextGrid')
    wavelet = loadS1audio(audio, S1ts)
    model = whisper.load_model(model_size)
    stable_whisper.modify_model(model)
    transcript = model.transcribe(wavelet, verbose=False, language='pt')
    word_transcript = stable_whisper.group_word_timestamps(transcript,
      combine_compound=True, min_dur=float_info.epsilon)
    new_tier = []
    for word in word_transcript:
        w = word['text'].strip().lower() #TODO: strip non-alphabetic chars from start and end
        if w != '' and w[-1] in 'aeo' and VOWELS.search(w[:-1]) and \
          w[-2] not in 'ãõ':
            new_tier.append(Interval(w, word['start'], word['end']))
