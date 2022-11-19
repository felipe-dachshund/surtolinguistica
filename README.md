# surtolinguistica
Extrai palavras terminadas em vogal átona de áudio em português

## Protótipo
A ideia é criar um script auxiliar para extrair palavras terminadas em vogal átona de um áudio em português para fins de pesquisa em sociolinguística. Inicialmente, pretende-se que o script execute as seguintes etapas:
1. Transcrever o áudio
2. Identificar as palavras terminadas em vogal átona
3. Localizar as palavras no áudio (i.e., os timestamps)
4. Criar um (novo) TextGrid para o áudio, com uma trilha contendo a seleção de palavras

Para a transcrição do áudio e localização dos timestamps das palavras, pode-se usar o pacote [whisper](https://github.com/openai/whisper) e o script [stable-ts](https://github.com/jianfch/stable-ts) que modifica o modelo de língua utilizado pelo whisper. Para trabalhar com o TextGrid, talvez o módulo [textgrid](https://github.com/nltk/nltk_contrib/blob/master/nltk_contrib/textgrid.py) do [NLTK](https://www.nltk.org/) seja útil.
