import re
import whisper
from whisper.audio import SAMPLE_RATE
import stable_whisper as stable
from nltk_contrib.textgrid import TextGrid
import sys
import argparse

VOWELS = re.compile('[aeiouáàâéêíóôú]')
WORD = re.compile('\w[\w-]*\w|\w')

def readS1ts(grid):
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

def loadS1audio(audio, S1_time_stamps):
    wavelet = whisper.load_audio(audio)
    i = 0
    for xmin, xmax in S1_time_stamps:
        wavelet[i:int(xmin*SAMPLE_RATE)] = 0
        i = int(xmax*SAMPLE_RATE)
    wavelet[i:] = 0
    return wavelet

def word_strip(word):
    word_match = WORD.search(word.lower())
    if word_match:
        return word_match.group(0)
    return ''

def print_tier_head(idx, name, xmin, xmax, size):
    return '''    item[{}]:
        class = "IntervalTier"
        name = "{}"
        xmin = {:.3f}
        xmax = {:.3f}
        intervals: size = {}
'''.format(idx, name, xmin, xmax, size)

def print_new_textgrid(grid, new_tier):
    output = '''File type = "ooTextFile"
Object class = "TextGrid"

xmin = {:.3f}
xmax = {:.3f}
tiers? <exists>
size = {}
item []:
'''.format(grid.xmin, grid.xmax, grid.size + 1)
    i = 1
    for tier in grid:
        output += print_tier_head(i, tier.tier_name(), tier.xmin, tier.xmax,
                                  tier.size) + tier.transcript
        i += 1
        if tier.tier_name() == 'S1':
            output += print_tier_head(i, 'Palavras', grid.xmin, grid.xmax,
                                      len(new_tier))
            for j, word in enumerate(new_tier):
                output += '''        intervals [{}]
            xmin = {:.3f}
            xmax = {:.3f}
            text = "{}"
'''.format(j + 1, word[0], word[1], word[2])
            i += 1
    return output

def extract(audio, model_size='small'):
    def print_message(*message):
        print(*message, file=sys.stderr)

    if '.' not in audio:
        audio += '.wav'
    basename = audio[:audio.rfind('.')]

    grid = TextGrid.load(basename + '.TextGrid')
    S1ts = readS1ts(grid)
    print_message('Found', len(S1ts), 'S1 interval(s).')

    print_message('Loading audio...')
    wavelet = loadS1audio(audio, S1ts)
    print_message('Loading model...')
    model = whisper.load_model(model_size)
    stable.modify_model(model)

    print_message('Transcribing...')
    transcript = model.transcribe(wavelet, verbose=False, language='pt')
    word_transcript = stable.group_word_timestamps(transcript,
      combine_compound=True, min_dur=sys.float_info.epsilon)

    new_tier = []
    for word in word_transcript:
        w = word_strip(word['text'])
        if w != '' and w != 'que' and w[-1] in 'aeo' and \
          VOWELS.search(w[:-1]) and w[-2] not in 'ãõ':
            new_tier.append((word['start'], word['end'], w))

    output = print_new_textgrid(grid, new_tier)
    i = 2
    while True:
        try:
            new_textgrid_name = basename + '_v' + str(i) + '.TextGrid'
            new_textgrid = open(new_textgrid_name, mode='x')
            break
        except:
            i += 1
    new_textgrid.write(output)
    new_textgrid.close()
    print_message('New TextGrid saved to ' + new_textgrid_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='surtoling',
      description='Extrai palavras terminadas em vogal átona de áudio em ' + \
      'português, guiando-se por um TextGrid de mesmo nome. Gera um novo ' + \
      'TextGrid com as palavras extraídas.')
    parser.add_argument('audiofile', metavar='AUDIOFILE',
      help='audio file name/path. If extension is not given, .wav is assumed')
    parser.add_argument('--model-size', default='small', metavar='SIZE',
      choices=['tiny', 'base', 'small', 'medium', 'large'],
      help='size of the trained model. It can be one of tiny, base, small, ' + \
      'medium or large. More info at https://github.com/openai/whisper')
    args = parser.parse_args()
    extract(args.audiofile, model_size=args.model_size)
