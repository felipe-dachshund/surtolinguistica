import whisper
import stable_whisper
from nltk_contrib import textgrid

VOWELS = re.compile('[aeiou]')

def readS1ts(textgridfile):
    grid = textgrid.TextGrid.load(textgridfile)
    for tier in grid.tiers:
        if tier.name == 'S1':
            S1 = tier
            break
    #TODO: Read the xmin and xmax in each interval in S1.transcript
    return [(S1.xmin, S1.xmax)]

def is_last_nontonic(word):
    #TODO
    return True

class Interval:
    def __init__(self, name, xmin, xmax):
        self.name = name
        self.xmin = xmin
        self.xmax = xmax

def extract(audio, model_size='small'):
    basename = audio[:audio.rfind('.')]
    S1ts = readS1ts(basename + '.TextGrid')
    model = whisper.load_model(model_size)
    stable_whisper.modify_model(model)
    transcript = model.transcribe(audio, language='pt')
    new_tier = []
    w = ''
    ts = 0.0
    for segment in transcript['segments']:
        for word in segment['whole_word_timestamps']:
            if w != '' and VOWELS.match(w[-1].lower()) and is_last_nontonic(w):
                new_tier.append(Interval(w, ts, word['timestamp']))
            w = word['word'].strip() #TODO: strip non-alphabetic chars from start and end
            ts = word['timestamp']