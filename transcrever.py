# import whisper  # Biblioteca para transcrição de áudio
# import torch  # Biblioteca para computação em GPU
# import os  # Biblioteca para manipulação de arquivos e diretórios
# from datetime import datetime  # Biblioteca para manipulação de datas

# def transcrever_com_nvidia(audio_path, modelo_tamanho="base"):
#     # Função para transcrever áudio usando Whisper com suporte a GPU NVIDIA
#     print("Iniciando transcrição com GPU NVIDIA...")
    
#     device = "cpu"  # Inicia com fallback para CPU
#     # Tenta configurar o dispositivo para GPU NVIDIA
#     if torch.cuda.is_available():
#         print("GPU detectada. Usando CUDA.")
#         device = "cuda"
#     else:
#         print("GPU NVIDIA não encontrada. Usando CPU.") 

#     try:
#         # Carrega o modelo no dispositivo detectado (GPU ou CPU)
#         model = whisper.load_model(modelo_tamanho, device=device)
        
#         # Configurações otimizadas para evitar repetição
#         resultado = model.transcribe(
#             audio_path, 
#             fp16=torch.cuda.is_available(),  # Usa fp16 apenas se CUDA estiver disponível
#             language="pt",                   # Força português para melhor acurácia
            
#             # Parâmetros para evitar repetições e melhorar a qualidade
#             # Fornece uma lista de temperaturas. O Whisper tentará cada uma
#             # até encontrar uma que passe nos testes de compressão e probabilidade.
#             temperature=(0.0, 0.2, 0.4, 0.6, 0.8),
            
#             # Limiar para a taxa de compressão. Ajuda a evitar texto sem sentido e repetitivo.
#             compression_ratio_threshold=2.4,
            
#             # Limiar para a probabilidade média dos tokens. Evita frases com baixa confiança.
#             logprob_threshold=-0.8,
            
#             best_of=5,                      # Aumentado para mais robustez
#             verbose=True,                   # Mostra detalhes da transcrição
#             beam_size=5                     # Usa beam search para melhor precisão
#         )
        
#         return resultado
        
#     except Exception as e:
#         print(f"Ocorreu um erro durante a transcrição: {e}")
#         return {"text": f"Erro na transcrição: {e}"}

# # Exemplo de uso
# if __name__ == "__main__":
#     # Configurações
#     arquivo_audio = "testeWebcam.mp3"  # Caminho do seu arquivo de áudio
#     modelo = "base"               # Tamanho do modelo: tiny, base, small, medium, large
    
#     # Executa a transcrição
#     resultado = transcrever_com_nvidia(arquivo_audio, modelo)
    
#     # Extrai o texto transcrito
#     texto_transcrito = resultado.get("text", "").strip()

#     # Exibe os resultados
#     print("\n" + "="*50) # Separador visual
#     print("📄 TEXTO TRANSCRITO:") # Separador visual
#     print("="*50) # Separador visual
#     if texto_transcrito: # Verifica se há texto transcrito
#         print(texto_transcrito) # Exibe o texto transcrito
#     else: # Caso contrário, informa que não há texto
#         print("Nenhum texto foi transcrito ou ocorreu um erro.") # Mensagem de erro
        
#     # Salva a transcrição em um arquivo de texto se houver conteúdo
#     if texto_transcrito:
#         try:
#             # Cria um nome de arquivo com base na data e hora atuais para ser único
#             data_atual = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") # Formato: YYYY-MM-DD_HH-MM-SS
#             nome_arquivo = f"transcricao_{data_atual}.txt" # Nome do arquivo
#             # Salva o texto transcrito em um arquivo de texto
#             with open(nome_arquivo, "w", encoding="utf-8") as f: # Abre o arquivo para escrita
#                 f.write(texto_transcrito) # Escreve o texto no arquivo
                
#             print(f"\n✅ Transcrição salva com sucesso no arquivo: {nome_arquivo}") # Mensagem de sucesso
            
#         except Exception as e: # Captura erros ao salvar o arquivo
#             print(f"\n❌ Erro ao salvar o arquivo de transcrição: {e}") # Mensagem de erro ao salvar

# códido do chatgpt para pegar o ultimo arquivo da pasta downloads.
import whisper
import torch
import os
import subprocess
from datetime import datetime

# Função: pega o último arquivo baixado na pasta Downloads
def pegar_ultimo_downloads():
    pasta_downloads = os.path.join(os.path.expanduser("~"), "Downloads")
    arquivos = [os.path.join(pasta_downloads, f) for f in os.listdir(pasta_downloads)]
    arquivos = [f for f in arquivos if os.path.isfile(f)]
    if not arquivos:
        return None
    return max(arquivos, key=os.path.getctime)

# Função: converte vídeo em áudio MP3 usando ffmpeg
def converter_para_mp3(caminho_arquivo): 
    nome_base, _ = os.path.splitext(caminho_arquivo) # Remove a extensão original
    caminho_mp3 = nome_base + ".mp3" # Define o novo caminho com extensão .mp3
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", caminho_arquivo, "-vn", "-acodec", "mp3", caminho_mp3], # Comando ffmpeg para conversão
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print(f"🎵 Arquivo convertido para áudio: {caminho_mp3}")
        return caminho_mp3
    except Exception as e:
        print(f"❌ Erro ao converter o arquivo: {e}")
        return None

# Função para transcrever áudio usando Whisper com suporte a GPU NVIDIA
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
    arquivo = pegar_ultimo_downloads() # Pega o último arquivo da pasta Downloads
    
    if arquivo:
        print(f"📂 Último arquivo encontrado: {arquivo}") 

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
        print("❌ Nenhum arquivo encontrado na pasta Downloads.")
