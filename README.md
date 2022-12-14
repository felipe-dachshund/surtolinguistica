# SurtoLinguística
Extrai palavras terminadas em vogal átona de áudio em português

## Protótipo
A ideia é criar um script auxiliar para extrair palavras terminadas em vogal átona de um áudio em português para fins de pesquisa em sociolinguística. Inicialmente, pretende-se que o script execute as seguintes etapas:
1. Transcrever o áudio
2. Identificar as palavras terminadas em vogal átona
3. Localizar as palavras no áudio (i.e., os timestamps)
4. Criar um (novo) TextGrid para o áudio, com uma trilha contendo a seleção de palavras

Para a transcrição do áudio e localização dos timestamps das palavras, pode-se usar os pacotes [whisper](https://github.com/openai/whisper) e [stable-ts](https://github.com/jianfch/stable-ts), que modifica o modelo de língua utilizado pelo whisper. Para trabalhar com o TextGrid, usamos o pacote [TextGridTools](https://github.com/hbuschme/TextGridTools).

- [x] Ler os limites de cada intervalo da trilha S1.
- [x] Fazer a transcrição de cada intervalo separadamente. Resolvido: não é preciso transcrever cada intervalo separadamente, basta mascarar (emudecer) os trechos do áudio que não pertencem a nenhum intervalo da trilha do informante S1.
- [x] O tokenizador usado pelo whisper e pelo stable-ts não tokeniza corretamente em palavras. Usar talvez o tokenizador do NLTK ou do SpaCy. Resolvido: o stable-ts possui uma solução satisfatória para sistemas de escrita que utilizam espaço para separar palavras.
- [x] Identificar se a palavra termina em vogal e se a última sílaba é átona. Resolvido com heurística: a palavra tem grandes chances de ter mais de uma sílaba e terminar com vogal átona se terminar em 'a', 'e' ou 'o' não acentuados graficamente, tiver alguma outra vogal e a penúltima letra não for uma vogal nasalizada.
- [x] Criar um novo TextGrid com uma trilha a mais, contendo as palavras terminadas em vogal átona.

## Setup para contribuidores
Caso você queira contribuir com o projeto, além de fazer fork, clonar, etc., é preciso instalar algumas outras coisas, como o [FFmpeg](https://ffmpeg.org/), o [ffmpeg-python](https://github.com/kkroening/ffmpeg-python), o [PyTorch](https://pytorch.org/), os [transformers do Hugging Face](https://huggingface.co/docs/transformers/index), o [whisper](https://github.com/openai/whisper), o [TextGridTools](https://github.com/hbuschme/TextGridTools) e o [stable-ts](https://github.com/jianfch/stable-ts). Você pode instalar estes pacotes e módulos pelo `pip` ou pelo [Anaconda](https://anaconda.org/). Como o projeto utiliza o PyTorch, o qual é instável e costuma apresentar conflito com outros pacotes e programas, recomenda-se criar um ambiente virtual para o projeto, seja por meio do Python, seja por meio do Anaconda. O Anaconda, além de criar ambientes virtuais, já vem com vários programas adicionais.

### Usando pip
Primeiro, instale o Python, caso ainda não o tenha instalado. Usuários Linux ou MacOS devem instalar o ffmpeg pelo repositório online, por exemplo, via (Ubuntu/Debian)
```
sudo apt install ffmpeg
```
Usuários Windows devem instalar da [página do FFmpeg](https://ffmpeg.org/).

Depois disso, é preciso criar um ambiente virtual no python para o projeto, evitando conflitos e quebras de pacotes devidos ao PyTorch e aos transformers. Em alguns sistemas, é necessário instalar um pacote para a criação de ambientes virtuais no python, por exemplo (Ubuntu/Debian),
```
sudo apt install python3-venv
```
Para criar um ambiente com o nome de, digamos, `.env`, dirija-se à pasta do projeto e use
```
python3 -m venv .env
```
Isto deve criar uma pasta `.env` com o ambiente virtual. Caso esteja contribuindo via git, opte por nomes constantes no arquivo `.gitignore`, como `env`, `venv`, `ENV`, `.env` ou `.venv`, para evitar que a pasta de seu ambiente virtual seja adicionada acidentalmente ao projeto.

Agora ative o ambiente usando um dos comandos a seguir, dependendo do sistema operacional:
```
source .env/bin/activate
.\.env\Scripts\activate
```
Agora os pacotes python podem ser instalados via pip:
```
pip install wheel
pip install tgt git+https://github.com/openai/whisper.git
pip install stable-ts
```

Para desativar o ambiente virtual, use
```
deactivate
```

### Usando Anaconda
Caso ainda não tenha o Anaconda, instale-o seguindo as instruções em da [página](https://www.anaconda.com/products/distribution).

Crie um ambiente de nome, digamos, `surtolinguistica` com alguns dos pacotes necessários e ative-o:
```
conda create -n surtolinguistica ffmpeg pytorch transformers more-itertools decorator=4.4.2 imageio
conda activate surtolinguistica
```
Instale os pacotes restantes usando o canal `conda-forge` e o pip:
```
conda install -c conda-forge ffmpeg-python moviepy
pip install tgt git+https://github.com/openai/whisper.git
pip install stable-ts
```
Agora você pode desativar o ambiente:
```
conda deactivate
```
