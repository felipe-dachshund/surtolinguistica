import re, whisper
from whisper.audio import SAMPLE_RATE
import stable_whisper as stable
import tgt, sys, os, argparse

VOWELS = re.compile('[aeiouáàâéêíóôú]')
WORD = re.compile('\w[\w-]*\w|\w')

def loadS1audio(audio, S1):
    wavelet = whisper.load_audio(audio)
    i = 0
    for interval in S1:
        wavelet[i:int(interval.start_time*SAMPLE_RATE)] = 0
        i = int(interval.end_time*SAMPLE_RATE)
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

    grid = tgt.read_textgrid(basename + '.TextGrid')
    S1 = grid.get_tier_by_name('S1')
    print_message('Found', len(S1), 'S1 interval(s).')

    print_message('Loading audio...')
    wavelet = loadS1audio(audio, S1)
    print_message('Loading model...')
    model = whisper.load_model(model_size)
    stable.modify_model(model)

    print_message('Transcribing...')
    transcript = model.transcribe(wavelet, verbose=False, language='pt')
    word_transcript = stable.group_word_timestamps(transcript,
      combine_compound=True, min_dur=sys.float_info.epsilon)

    ocorrencia = tgt.IntervalTier(start_time=grid.start_time,
      end_time=grid.end_time, name='ocorrencia')
    for word in word_transcript:
        w = word_strip(word['text'])
        if w != '' and w != 'que' and w[-1] in 'aeo' and \
          VOWELS.search(w[:-1]) and w[-2] not in 'ãõ':
            ocorrencia.add_interval(tgt.Interval(word['start'], word['end'],
              text=w))

    grid.insert_tier(ocorrencia, grid.get_tier_names().index('S1') + 1)

    i = 2
    new_textgrid_name = basename + '_v2.TextGrid'
    while os.access(new_textgrid_name, os.F_OK):
        i += 1
        new_textgrid_name = basename + '_v' + str(i) + '.TextGrid'
    tgt.write_to_file(grid, new_textgrid_name, format='long')
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
