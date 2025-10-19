import os
import json
import shutil
from pathlib import Path
import google.generativeai as genai
from PIL import Image

# --- CONFIGURAÇÃO ---
# 1. Cole a sua chave de API do Gemini aqui
GOOGLE_API_KEY = "AIzaSyCFrs_EVe2AAZ63f7wTRFxcIEc8ztF9iu4" 

# 2. Configure as suas pastas. Use 'r' antes das aspas para evitar erros no Windows.
PASTA_ENTRADA = r"C:\Documentos\SARON\ImagensPreProcessadas"
PASTA_SAIDA = r"C:\Documentos\SARON\ImagensProcessadas"
# --------------------

try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    print(f"Erro ao configurar a API do Gemini: {e}")
    print("Verifique se a sua GOOGLE_API_KEY está correta.")
    exit()

# Modelo a ser usado. O 1.5 Flash é rápido e económico.
model = genai.GenerativeModel('gemini-2.5-flash')

def extrair_dados_com_gemini(img_path):
    """
    Envia uma imagem para a API Gemini e pede para extrair a referência e a cor.
    """
    print(f"  > Analisando com Gemini: {os.path.basename(img_path)}")
    try:
        # --- CORREÇÃO APLICADA AQUI ---
        # Usar 'with' garante que o ficheiro é fechado automaticamente
        with Image.open(img_path) as img:
            prompt = """
            Analise a imagem destes óculos. Procure por um texto na haste que siga o formato "Referência Tamanho[]Tamanho-Tamanho Cor".
            Se encontrar, extraia a Referência e a Cor.
            Responda APENAS com um objeto JSON no seguinte formato:
            {"referencia": "VALOR_DA_REFERENCIA", "cor": "VALOR_DA_COR"}
            Se não encontrar o texto nesse formato específico, responda com:
            {"referencia": null, "cor": null}
            """
            response = model.generate_content([prompt, img])
        # --- FIM DA CORREÇÃO ---
        
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
        
        return json.loads(cleaned_response)
    except Exception as e:
        print(f"  ! Erro na chamada à API Gemini para '{os.path.basename(img_path)}': {e}")
        return {"referencia": None, "cor": None}

def encontrar_imagens_similares_com_gemini(imagem_chave_path, lote_paths):
    """
    Envia a imagem chave e o resto do lote para o Gemini e pergunta quais são do mesmo par de óculos.
    """
    print(f"  > Verificando similaridade para a referência...")
    try:
        imagens_para_api = [Image.open(p) for p in lote_paths]
        nomes_ficheiros = [os.path.basename(p) for p in lote_paths]
        imagem_chave_nome = os.path.basename(imagem_chave_path)

        prompt = f"""
        A imagem de referência é "{imagem_chave_nome}". Das imagens fornecidas, identifique quais mostram EXATAMENTE o mesmo par de óculos.
        Considere a cor, o formato, o design e o material.
        Responda APENAS com uma lista JSON contendo os nomes dos ficheiros das imagens que correspondem (incluindo a imagem de referência).
        Os nomes dos ficheiros disponíveis são: {json.dumps(nomes_ficheiros)}
        """
        
        response = model.generate_content([prompt] + imagens_para_api)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
        
        return json.loads(cleaned_response)

    except Exception as e:
        print(f"  ! Erro na verificação de similaridade com Gemini: {e}")
        return [os.path.basename(imagem_chave_path)]


def processar_lotes_com_gemini(pasta_entrada, pasta_saida):
    """ Função principal que organiza e processa os lotes usando a API Gemini. """
    print("Iniciando o processamento com a API Gemini...")
    Path(pasta_saida).mkdir(parents=True, exist_ok=True)
    json_path = os.path.join(pasta_saida, "dados_oculos.json")
    dados_extraidos = []

    # --- INÍCIO DA NOVA LÓGICA ---
    # 1. Mapear todos os ficheiros já processados na pasta de saída.
    print("Verificando ficheiros já processados na pasta de saída...")
    processed_files_set = set()
    # A expressão `**/*` procura em todas as subpastas
    for file_path in Path(pasta_saida).glob('**/*'):
        if file_path.is_file():
            filename = file_path.name
            # O nome do ficheiro é "Cor_NomeOriginal.jpg". Extraímos o nome original.
            parts = filename.split('_', 1)
            if len(parts) > 1:
                original_name = parts[1]
                processed_files_set.add(original_name)

    if processed_files_set:
        print(f"Encontrados {len(processed_files_set)} ficheiros já processados. Serão ignorados.")
    # --- FIM DA NOVA LÓGICA ---

    tipos_de_imagem = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff')
    
    # 2. Obter a lista completa de ficheiros de entrada
    all_input_files = []
    for tipo in tipos_de_imagem:
        all_input_files.extend(Path(pasta_entrada).glob(tipo))
    
    # 3. Filtrar a lista, mantendo apenas os ficheiros que NÃO foram processados
    arquivos_imagem = sorted([
        str(p) for p in all_input_files if p.name not in processed_files_set
    ])
    
    total_input_files = len(list(all_input_files))
    total_a_processar = len(arquivos_imagem)

    print(f"Encontradas {total_input_files} imagens na pasta de entrada.")
    if total_input_files - total_a_processar > 0:
        print(f"Ignorando {total_input_files - total_a_processar} que já estão na pasta de saída.")
    
    if total_a_processar == 0:
        print("Nenhum ficheiro novo para processar. Encerrando.")
        return

    print(f"Total de imagens a processar: {total_a_processar}.")


    i = 0
    while i < len(arquivos_imagem):
        lote_paths = arquivos_imagem[i:i+5]
        if not lote_paths: break
        
        print(f"\nProcessando lote: {[os.path.basename(p) for p in lote_paths]}")

        imagem_chave_path = None
        dados_oculos = {}

        for img_path in lote_paths:
            resultado_api = extrair_dados_com_gemini(img_path)
            if resultado_api and resultado_api.get("referencia"):
                dados_oculos = resultado_api
                imagem_chave_path = img_path
                print(f"  > Padrão ENCONTRADO em '{os.path.basename(img_path)}': Ref: {dados_oculos['referencia']}, Cor: {dados_oculos['cor']}")
                break
        
        if imagem_chave_path:
            nomes_similares = encontrar_imagens_similares_com_gemini(imagem_chave_path, lote_paths)
            caminhos_a_mover = [p for p in lote_paths if os.path.basename(p) in nomes_similares]
            
            dados_extraidos.append(dados_oculos)
            
            # Limpa a referência para que seja um nome de pasta válido (substitui / por _)
            referencia_limpa = str(dados_oculos["referencia"]).replace('/', '_').replace('\\', '_')
            
            pasta_destino = Path(pasta_saida) / referencia_limpa
            pasta_destino.mkdir(exist_ok=True)

            print(f"  > Agrupando e movendo {len(caminhos_a_mover)} imagens para a pasta '{dados_oculos['referencia']}'.")
            
            for path_mover in caminhos_a_mover:
                nome_original = os.path.basename(path_mover)
                novo_nome = f"{dados_oculos['cor']}_{nome_original}"
                shutil.move(path_mover, pasta_destino / novo_nome)
            
            arquivos_imagem = [arq for arq in arquivos_imagem if arq not in caminhos_a_mover]
            continue 
        else:
            print("  ! Nenhuma imagem com o texto no formato esperado foi encontrada neste lote.")
            i += len(lote_paths)

    if dados_extraidos:
        # Carrega o JSON existente para adicionar novos dados em vez de sobrescrever
        dados_existentes = []
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                try:
                    dados_existentes = json.load(f)
                except json.JSONDecodeError:
                    print("Aviso: O ficheiro JSON existente está corrompido ou vazio. Será sobrescrito.")
        
        dados_existentes.extend(dados_extraidos)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(dados_existentes, f, ensure_ascii=False, indent=4)
        print(f"\nDados salvos com sucesso em '{json_path}'.")

    print("\nProcessamento concluído.")

if __name__ == '__main__':
    if not os.path.isdir(PASTA_ENTRADA):
        print(f"ERRO: A pasta de entrada '{PASTA_ENTRADA}' não foi encontrada.")
    elif "SUA_API_KEY_AQUI" in GOOGLE_API_KEY:
        print("ERRO: Por favor, substitua 'SUA_API_KEY_AQUI' pela sua chave de API do Gemini no script.")
    else:
        processar_lotes_com_gemini(PASTA_ENTRADA, PASTA_SAIDA)