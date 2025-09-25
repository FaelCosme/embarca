import whisper # Biblioteca para transcrição de áudio
import torch # Biblioteca para computação em GPU
import os # Biblioteca para manipulação de arquivos e diretórios
import subprocess # Biblioteca para executar comandos do sistema
import requests # Biblioteca para fazer requisições HTTP
from datetime import datetime # Biblioteca para manipulação de datas

# URL do servidor no Raspberry Pi (ajuste o IP e porta conforme necessário)
URL_SERVIDOR = "http://192.168.1.239:8080"  # Exemplo: "http://<IP_DO_RASPBERRY>:8080"

# Função: baixa automaticamente o arquivo do servidor
def baixar_arquivo(url, destino="download_raspberry"):
    try:
        print(f"⬇️ Iniciando download do servidor: {url}")
        resposta = requests.get(url, stream=True)

        if resposta.status_code != 200: 
            print(f"❌ Erro ao conectar no servidor: {resposta.status_code}")
            return None

        # Criar pasta "downloadsRasp" se não existir
        pasta_destino = "downloadsRasp" # Pasta para salvar os downloads
        os.makedirs(pasta_destino, exist_ok=True) # Cria a pasta se não existir

        # Nome base do arquivo
        nome_arquivo = destino 
        if "content-disposition" in resposta.headers:
            # Se o servidor enviar cabeçalho com nome
            nome_arquivo = resposta.headers["content-disposition"].split("filename=")[-1].strip('"')

        # Adiciona data e hora ao nome do arquivo
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base, ext = os.path.splitext(nome_arquivo) # Separa nome e extensão
        nome_arquivo = f"{base}_{timestamp}{ext}" # Novo nome com timestamp

        # Caminho completo do arquivo dentro da pasta "downloadsRasp"
        caminho_arquivo = os.path.join(pasta_destino, nome_arquivo) # Caminho completo do arquivo

        with open(caminho_arquivo, "wb") as f: 
            for chunk in resposta.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"✅ Download concluído: {caminho_arquivo}")
        return caminho_arquivo
    except Exception as e:
        print(f"❌ Erro durante o download: {e}")
        return None

# Função: converte vídeo em áudio MP3 usando ffmpeg
def converter_para_mp3(caminho_arquivo): 
    nome_base, _ = os.path.splitext(caminho_arquivo)
    caminho_mp3 = nome_base + ".mp3"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", caminho_arquivo, "-vn", "-acodec", "mp3", caminho_mp3],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print(f"🎵 Arquivo convertido para áudio: {caminho_mp3}")
        return caminho_mp3
    except Exception as e:
        print(f"❌ Erro ao converter o arquivo: {e}")
        return None

# Função para transcrever áudio usando Whisper com GPU NVIDIA
def transcrever_com_nvidia(audio_path, modelo_tamanho="base"):
    print("Iniciando transcrição com GPU NVIDIA...")

    device = "cpu"
    if torch.cuda.is_available():
        print("GPU detectada. Usando CUDA.")
        device = "cuda"
    else:
        print("GPU NVIDIA não encontrada. Usando CPU.") 

    try:
        model = whisper.load_model(modelo_tamanho, device=device)

        resultado = model.transcribe(
            audio_path,
            fp16=torch.cuda.is_available(),
            language="pt",
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8),
            compression_ratio_threshold=2.4,
            logprob_threshold=-0.8,
            best_of=5,
            verbose=True,
            beam_size=5
        )
        return resultado

    except Exception as e:
        print(f"Ocorreu um erro durante a transcrição: {e}")
        return {"text": f"Erro na transcrição: {e}"}


if __name__ == "__main__":
    # Faz o download automático do servidor Raspberry Pi
    arquivo = baixar_arquivo(URL_SERVIDOR)

    if arquivo:
        # Extensões aceitas diretamente
        extensoes_audio = [".mp3", ".wav", ".m4a", ".ogg"]
        _, ext = os.path.splitext(arquivo)

        if ext.lower() not in extensoes_audio:
            print("🎬 Detectado vídeo. Convertendo para MP3...")
            arquivo = converter_para_mp3(arquivo)

        if arquivo:
            modelo = "base"
            resultado = transcrever_com_nvidia(arquivo, modelo)
            texto_transcrito = resultado.get("text", "").strip()

            print("\n" + "="*50)
            print("📄 TEXTO TRANSCRITO:")
            print("="*50)
            if texto_transcrito:
                print(texto_transcrito)
            else:
                print("Nenhum texto foi transcrito ou ocorreu um erro.")

            if texto_transcrito:
                try:
                    data_atual = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    nome_arquivo = f"transcricao_{data_atual}.txt"
                    with open(nome_arquivo, "w", encoding="utf-8") as f:
                        f.write(texto_transcrito)
                    print(f"\n✅ Transcrição salva com sucesso no arquivo: {nome_arquivo}")
                except Exception as e:
                    print(f"\n❌ Erro ao salvar o arquivo de transcrição: {e}")
        else:
            print("❌ Não foi possível processar o arquivo.")
    else:
        print("❌ Download não realizado.")
