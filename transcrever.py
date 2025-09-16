import whisper  # Biblioteca para transcrição de áudio
import torch  # Biblioteca para computação em GPU
import os  # Biblioteca para manipulação de arquivos e diretórios
from datetime import datetime  # Biblioteca para manipulação de datas

def transcrever_com_nvidia(audio_path, modelo_tamanho="base"):
    """
    Transcreve um áudio usando a GPU NVIDIA com o Whisper, com otimizações
    para evitar saídas repetitivas e de baixa qualidade.
    """
    print("Iniciando transcrição com GPU NVIDIA...")
    
    device = "cpu"  # Inicia com fallback para CPU
    # Tenta configurar o dispositivo para GPU NVIDIA
    if torch.cuda.is_available():
        print("GPU detectada. Usando CUDA.")
        device = "cuda"
    else:
        print("GPU NVIDIA não encontrada. Usando CPU.")

    try:
        # Carrega o modelo no dispositivo detectado (GPU ou CPU)
        model = whisper.load_model(modelo_tamanho, device=device)
        
        # Configurações otimizadas para evitar repetição
        resultado = model.transcribe(
            audio_path,
            fp16=torch.cuda.is_available(),  # Usa fp16 apenas se CUDA estiver disponível
            language="pt",                   # Força português para melhor acurácia
            
            # --- PRINCIPAIS MUDANÇAS AQUI ---
            # Fornece uma lista de temperaturas. O Whisper tentará cada uma
            # até encontrar uma que passe nos testes de compressão e probabilidade.
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8),
            
            # Limiar para a taxa de compressão. Ajuda a evitar texto sem sentido e repetitivo.
            compression_ratio_threshold=2.4,
            
            # Limiar para a probabilidade média dos tokens. Evita frases com baixa confiança.
            logprob_threshold=-0.8,
            
            # --- FIM DAS MUDANÇAS ---
            
            best_of=5,                      # Aumentado para mais robustez
            verbose=True,                   # Mostra detalhes da transcrição
            beam_size=5                     # Usa beam search para melhor precisão
        )
        
        return resultado
        
    except Exception as e:
        print(f"Ocorreu um erro durante a transcrição: {e}")
        return {"text": f"Erro na transcrição: {e}"}

# --- BLOCO PRINCIPAL (USO PRÁTICO) ---
if __name__ == "__main__":
    # Configurações
    arquivo_audio = "testeM.mp3"  # Caminho do seu arquivo de áudio
    modelo = "base"               # Tamanho do modelo: tiny, base, small, medium, large
    
    # Executa a transcrição
    resultado = transcrever_com_nvidia(arquivo_audio, modelo)
    
    # Extrai o texto transcrito
    texto_transcrito = resultado.get("text", "").strip()

    # Exibe os resultados
    print("\n" + "="*50)
    print("📄 TEXTO TRANSCRITO:")
    print("="*50)
    if texto_transcrito:
        print(texto_transcrito)
    else:
        print("Nenhum texto foi transcrito ou ocorreu um erro.")
        
    # --- LÓGICA DE SALVAMENTO SIMPLIFICADA ---
    # Salva a transcrição em um arquivo de texto se houver conteúdo
    if texto_transcrito:
        try:
            # Cria um nome de arquivo com base na data e hora atuais para ser único
            data_atual = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            nome_arquivo = f"transcricao_{data_atual}.txt"
            
            with open(nome_arquivo, "w", encoding="utf-8") as f:
                f.write(texto_transcrito)
                
            print(f"\n✅ Transcrição salva com sucesso no arquivo: {nome_arquivo}")
            
        except Exception as e:
            print(f"\n❌ Erro ao salvar o arquivo de transcrição: {e}")
            #teste