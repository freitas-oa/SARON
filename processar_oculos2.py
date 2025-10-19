import os
import json
import re
import shutil
import time
from pathlib import Path
import google.generativeai as genai
from PIL import Image
import dotenv

# --- CONFIGURAÇÃO ---
# 1. Chave de API e Pastas (do seu script)

GOOGLE_API_KEY = dotenv.get_key('GOOGLE_API_KEY')
PASTA_SAIDA = Path(r"C:\Documentos\SARON\ImagensProcessadas")
PASTA_ENTRADA = Path(r"C:\Documentos\SARON\ImagensPreProcessadas")
# --------------------

# --- Constantes do Script ---
LOG_FILE = Path("processed_files.log")
DATA_FILE = Path("extracted_data.json")
ERROR_LOG_FILE = Path("error_batches.log")
BATCH_SIZE = 5
PAUSE_AFTER_BATCHES = 5

# --- Configuração da API Gemini ---
try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    print(f"Erro ao configurar a API do Gemini: {e}")
    print("Verifique se a sua GOOGLE_API_KEY está correta.")
    exit()

# Modelo a ser usado
try:
    model = genai.GenerativeModel('gemini-2.5-pro')
    print("Modelo 'gemini-2.5-pro' carregado com sucesso.")
except Exception as e:
    print(f"Erro ao carregar o modelo 'gemini-2.5-pro': {e}")
    print("Verifique se o nome do modelo está correto ou se a API Key tem acesso.")
    exit()

# --- Funções Auxiliares de Log e Dados ---

def load_processed_files() -> set:
    """Carrega os nomes dos arquivos já processados."""
    if not LOG_FILE.exists():
        return set()
    try:
        with open(LOG_FILE, 'r') as f:
            return set(line.strip() for line in f)
    except Exception as e:
        print(f"Aviso: Não foi possível ler o arquivo de log: {e}")
        return set()

def log_processed_file(filename: str):
    """Registra um arquivo como processado."""
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(f"{filename}\n")
    except Exception as e:
        print(f"Aviso: Não foi possível escrever no arquivo de log: {e}")

def log_error_batch(batch_files: list):
    """Registra um lote que falhou em encontrar uma imagem chave."""
    try:
        with open(ERROR_LOG_FILE, 'a') as f:
            f.write(f"Failed to find key image in batch: {', '.join([p.name for p in batch_files])}\n")
    except Exception as e:
        print(f"Aviso: Não foi possível escrever no log de erros: {e}")

def load_data() -> list:
    """Carrega o JSON de dados extraídos."""
    if not DATA_FILE.exists():
        return []
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Aviso: {DATA_FILE} está corrompido ou vazio. A criar um novo.")
        return []
    except Exception as e:
        print(f"Aviso: Não foi possível ler {DATA_FILE}: {e}")
        return []

import json

# Assumindo que DATA_FILE está definida em algum lugar, ex:
# DATA_FILE = 'seu_arquivo.json'

def save_data(data_list: list, new_entry: dict):
    """
    Salva os dados no JSON, transformando a estrutura plana em aninhada.
    Se a referência existir, mescla a cor e a imagem.
    Se for nova, adiciona a entrada na nova estrutura.
    """
    
    # Extrai dados da nova entrada (formato plano)
    new_ref = new_entry.get('reference')
    new_color = new_entry.get('color')
    new_img_key = new_entry.get('key_image_file')

    # Validação básica
    if not all([new_ref, new_color, new_img_key]):
        print("Aviso: Nova entrada incompleta. Faltando ref, cor ou imagem.")
        return

    found_ref = False
    
    # Procura pela referência na lista (que está no formato aninhado)
    for entry in data_list:
        if entry.get('reference') == new_ref:
            found_ref = True
            
            # Pega (ou cria se não existir) o dicionário 'images'
            images_dict = entry.setdefault('images', {})
            
            # Pega (ou cria se não existir) a lista de cores para esta imagem
            color_list = images_dict.setdefault(new_img_key, [])
            
            # Adiciona a nova cor apenas se ela não estiver na lista
            if new_color not in color_list:
                color_list.append(new_color)
            
            break # Referência encontrada e processada

    # Se a referência não foi encontrada, cria uma nova entrada
    if not found_ref:
        # Transforma a entrada plana na nova estrutura aninhada
        transformed_entry = {
            "reference": new_ref,
            "size1": new_entry.get('size1'),
            "size2": new_entry.get('size2'),
            "size3": new_entry.get('size3'),
            "images": {
                new_img_key: [new_color]
            }
        }
        data_list.append(transformed_entry)

    # Salva a lista completa (no novo formato aninhado)
    try:
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Aviso: Não foi possível salvar {DATA_FILE}: {e}")


def validate_image(image_path: Path) -> bool:
    """Verifica se a imagem é válida usando Pillow."""
    try:
        with Image.open(image_path) as img:
            img.verify() 
        with Image.open(image_path) as img:
            img.load() 
        return True
    except Exception as e:
        print(f"  [AVISO] Imagem corrompida ou inválida: {image_path.name}: {e}")
        log_processed_file(image_path.name) # Log para ignorar futuramente
        return False

# --- NOVA FUNÇÃO DE API ÚNICA ---

def call_gemini_process_batch(batch_paths: list[Path]) -> dict | None:
    """
    Envia o LOTE INTEIRO para a API e pede para:
    1. Encontrar a imagem chave (com texto).
    2. Extrair os dados dessa imagem.
    3. Encontrar todas as imagens similares (mesmo modelo, ignora cor).
    """
    print(f"  > Analisando lote de {len(batch_paths)} imagens com Gemini...")
    pil_images = []
    file_names = []
    
    try:
        for p in batch_paths:
            pil_images.append(Image.open(p))
            file_names.append(p.name)

        prompt = f"""
        Analise o lote de imagens de óculos fornecido. Os nomes de ficheiro são: {json.dumps(file_names)}

        Siga estas 3 etapas:

        1.  **Encontrar a Imagem Chave:** Examine todas as imagens. Encontre a UMA imagem que contém o texto de referência na haste no formato "Referência Tamanho[]Tamanho2-Tamanho3 Cor" (ex: "0037 54[]18-145 C4").

        2.  **Extrair Dados:** Da imagem chave encontrada na Etapa 1, extraia os 5 componentes: Referência, Tamanho, Tamanho2, Tamanho3, e Cor.

        3.  **Comparar Modelo:** Identifique quais imagens no lote (incluindo a imagem chave) mostram o *mesmo modelo (mesma referência)* de óculos. **Ignore a cor** para esta etapa; foque-se apenas no formato e design.

        Responda APENAS com um objeto JSON no seguinte formato:

        {{
            "key_image_name": "nome_do_ficheiro_com_texto.jpg",
            "data": {{
                "reference": "VALOR_REFERENCIA",
                "size1": "VALOR_TAMANHO1",
                "size2": "VALOR_TAMANHO2",
                "size3": "VALOR_TAMANHO3",
                "color": "VALOR_COR"
            }},
            "matched_filenames": ["ficheiro_chave.jpg", "ficheiro_similar_1.jpg", "ficheiro_similar_2.jpg"]
        }}

        Se NENHUMA imagem no lote contiver o texto de referência no formato exato, responda com:
        {{
            "key_image_name": null,
            "data": null,
            "matched_filenames": []
        }}
        """

        api_payload = [prompt] + pil_images
        
        response = model.generate_content(api_payload)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
        
        data = json.loads(cleaned_response)
        
        if data and data.get("data"):
            return data
        else:
            return None

    except Exception as e:
        print(f"  ! Erro na chamada à API Gemini (process_batch): {e}")
        return None
    finally:
        # Garante que todas as imagens PIL sejam fechadas
        [img.close() for img in pil_images]


# --- Lógica Principal (Atualizada) ---

def main():
    print("Iniciando script de processamento de óculos...")
    print(f"Pasta de Entrada: {PASTA_ENTRADA}")
    print(f"Pasta de Saída: {PASTA_SAIDA}")

    # 1. Setup inicial
    PASTA_SAIDA.mkdir(exist_ok=True)
    processed_files = load_processed_files()
    all_data = load_data()
    
    # 2. Encontrar arquivos para processar
    tipos_de_imagem = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff')
    all_files = []
    for tipo in tipos_de_imagem:
        all_files.extend(PASTA_ENTRADA.glob(tipo))

    unprocessed_files = sorted([p for p in all_files if p.name not in processed_files])
    
    if not unprocessed_files:
        print("Nenhum arquivo novo para processar.")
        return

    print(f"Encontrados {len(unprocessed_files)} arquivos novos para processar (de {len(all_files)} no total).")

    # 3. Criar lotes
    batches = [unprocessed_files[i:i + BATCH_SIZE] for i in range(0, len(unprocessed_files), BATCH_SIZE)]
    
    batch_counter = 0
    total_batches = len(batches)

    # 4. Processar lotes
    for i, current_batch in enumerate(batches):
        batch_counter += 1
        print(f"\n--- Processando Lote {batch_counter}/{total_batches} ---")
        
        # 4.1. Validar imagens ANTES de enviar à API
        valid_images_in_batch = []
        for img_path in current_batch:
            if validate_image(img_path):
                valid_images_in_batch.append(img_path)
            # 'validate_image' já loga o erro e o ficheiro como "processado"

        if not valid_images_in_batch:
            print("  [AVISO] Nenhuma imagem válida neste lote. A saltar.")
            continue # Os ficheiros corrompidos já foram logados
        
        # print the name of images being sent
        print("  Imagens válidas neste lote:")
        for img in valid_images_in_batch:
            print(f"    - {img.name}")

        # 4.2. Chamar a API (UMA SÓ VEZ)
        api_result = call_gemini_process_batch(valid_images_in_batch)

        # 4.3. Processar resultado da API
        if not api_result or not api_result.get("data"):
            print("  [FALHA] Nenhuma imagem chave encontrada pela API neste lote.")
            log_error_batch(current_batch) # Loga o lote original
            # Loga todos os ficheiros deste lote como "processados" para não tentar novamente
            for img_path in current_batch:
                log_processed_file(img_path.name)
            continue # Próximo lote

        # 4.4. Sucesso - Extrair dados da resposta
        parsed_data = api_result["data"]
        key_image_name = api_result["key_image_name"]
        
        # Encontra o Path completo da imagem chave (necessário para o log de dados)
        key_image_path = None
        for p in valid_images_in_batch:
            if p.name == key_image_name:
                key_image_path = p
                break
        
        if not key_image_path:
             print(f"  [ERRO] API retornou key_image '{key_image_name}' mas não foi encontrado no lote. A saltar.")
             log_error_batch(current_batch)
             continue

        print(f"  [SUCESSO] Imagem Chave encontrada pela API: {key_image_name}")
        print(f"  Dados extraídos: {parsed_data}")

        matched_filenames = api_result["matched_filenames"]
        matched_paths = [p for p in current_batch if p.name in matched_filenames]
        
        # 4.5. Organizar arquivos
        # Limpa a referência para que seja um nome de pasta válido
        reference_clean = str(parsed_data["reference"]).replace('/', '_').replace('\\', '_').strip()
        if not reference_clean:
                print(f"  [ERRO] Referência extraída está vazia. A ignorar lote.")
                log_error_batch(current_batch)
                continue
                
        target_dir = PASTA_SAIDA / reference_clean
        target_dir.mkdir(exist_ok=True)
        
        print(f"  Movendo {len(matched_paths)} arquivos para {target_dir}")
        
        i = 0
        for file_path in matched_paths:
            try:
                # Move o ficheiro com o nome original
                shutil.copy(str(file_path), str(target_dir / f"{parsed_data['reference']}_{parsed_data['color']}_{str(i)}.jpg"))
                log_processed_file(file_path.name) # Log de sucesso (Req 1)
            except Exception as e:
                print(f"  [ERRO] Falha ao mover {file_path.name}: {e}")
                if (target_dir / file_path.name).exists():
                        print(f"  Ficheiro já existia no destino. A registar como processado.")
                        log_processed_file(file_path.name)
            i += 1
        
        # 4.6. Salvar dados
        new_entry = {**parsed_data, "key_image_file": key_image_path.name}
        save_data(all_data, new_entry)
        
        # 4.7. Pausa (Req 4)
        if batch_counter % PAUSE_AFTER_BATCHES == 0 and batch_counter < total_batches:
            print(f"\n--- Pausado após {batch_counter} lotes. ---")
            try:
                input("Pressione Enter para continuar os próximos 5 lotes (ou CTRL+C para parar)...")
            except KeyboardInterrupt:
                print("\nProcessamento interrompido pelo usuário.")
                break

    print("\nProcessamento concluído.")

if __name__ == "__main__":
    if not PASTA_ENTRADA.is_dir():
        print(f"ERRO: A pasta de entrada '{PASTA_ENTRADA}' não foi encontrada.")
    elif "AIzaSyC" not in GOOGLE_API_KEY: # Check simples se a key foi mudada
        print("ERRO: Por favor, substitua 'AIzaSyC...' pela sua chave de API do Gemini no script.")
    else:
        main()