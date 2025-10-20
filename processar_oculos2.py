import os
import json
import re
import shutil
import time
from pathlib import Path
import google.generativeai as genai
from PIL import Image
import dotenv
import time

# --- CONFIGURAÇÃO ---
# 1. Chave de API e Pastas (do seu script)

dotenv.load_dotenv()  # Carrega variáveis do arquivo .env
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

PASTA_SAIDA = Path(r"C:\Documentos\SARON\ImagensProcessadas")
PASTA_ENTRADA = Path(r"C:\Documentos\SARON\ImagensPreProcessadas")
# --------------------

# --- Constantes do Script ---
LOG_FILE = Path("processing_events.log") # Único ficheiro de log
DATA_FILE = Path("extracted_data.json")
BATCH_SIZE = 5
# PAUSE_AFTER_BATCHES = 0

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

def log_event(status: str, message: str, filenames: list = None):
    """Registra um evento (sucesso, falha, aviso) no log único."""
    try:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] [{status:^15}] {message}\n")
            if filenames:
                for name in filenames:
                    f.write(f"                   - {name}\n")
    except Exception as e:
        print(f"Aviso: Não foi possível escrever no arquivo de log: {e}")


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

def load_processed_files_from_json(data_list: list) -> set:
    """Extrai todos os nomes de ficheiros do JSON de dados (sucessos)."""
    filenames = set()
    try:
        for entry in data_list:
            if "image_files" in entry:
                for img_group in entry["image_files"]:
                    if "key_file" in img_group and img_group["key_file"]:
                        filenames.add(img_group["key_file"])
                    if "additional_files" in img_group:
                        filenames.update(img_group["additional_files"])
    except Exception as e:
        print(f"Aviso: Não foi possível ler nomes de ficheiros do JSON: {e}")
        log_event("ERRO_JSON", f"Não foi possível ler nomes de ficheiros do JSON: {e}")
    return filenames

def save_data(data_list: list, new_data: dict, key_file: str, additional_files: list):
    """
    Salva os dados no JSON, usando a nova estrutura aninhada.
    Se a referência existir, mescla o novo grupo de cor/ficheiros.
    """
    
    new_ref = new_data.get('reference')
    new_color = new_data.get('color')
    
    if not all([new_ref, new_color, key_file]):
        print("Aviso: Nova entrada incompleta. Faltando ref, cor ou imagem chave.")
        log_event("ERRO_SAVE", "Nova entrada incompleta", [key_file])
        return

    # Cria a nova entrada de ficheiro (para a cor específica)
    new_file_entry = {
        "color": new_color,
        "key_file": key_file,
        "additional_files": additional_files
    }

    found_ref = False
    for entry in data_list:
        if entry.get('reference') == new_ref:
            found_ref = True
            
            # Garante que 'image_files' existe
            image_files_list = entry.setdefault('image_files', [])
            
            # Verifica se esta cor já foi processada
            found_color = False
            for img_group in image_files_list:
                if img_group.get('color') == new_color:
                    # A cor já existe. Apenas por segurança, atualiza os ficheiros.
                    img_group['key_file'] = key_file
                    img_group['additional_files'] = additional_files
                    found_color = True
                    break
            
            if not found_color:
                image_files_list.append(new_file_entry)
            
            break # Referência encontrada e processada

    # Se a referência não foi encontrada, cria uma nova entrada
    if not found_ref:
        transformed_entry = {
            "reference": new_ref,
            "size1": new_data.get('size1'),
            "size2": new_data.get('size2'),
            "size3": new_data.get('size3'),
            "image_files": [new_file_entry] # Adiciona a nova entrada de ficheiro
        }
        data_list.append(transformed_entry)

    # Salva a lista completa
    try:
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Aviso: Não foi possível salvar {DATA_FILE}: {e}")
        log_event("ERRO_SAVE", f"Não foi possível salvar {DATA_FILE}: {e}")


def validate_image(image_path: Path) -> bool:
    """Verifica se a imagem é válida usando Pillow."""
    try:
        with Image.open(image_path) as img:
            img.verify() 
        with Image.open(image_path) as img:
            img.load() 
        return True
    except Exception as e:
        print(f"   [AVISO] Imagem corrompida ou inválida: {image_path.name}: {e}")
        log_event("IMG_CORROMPIDA", f"Imagem corrompida: {e}", [image_path.name])
        return False

# --- NOVA FUNÇÃO DE API ÚNICA ---

def call_gemini_process_batch(batch_paths: list[Path]) -> dict | None:
    """
    Envia o LOTE INTEIRO para la API e pede para:
    1. Encontrar a imagem chave (com texto).
    2. Extrair os dados dessa imagem.
    3. Encontrar todas as imagens similares (mesmo modelo e cor).
    """
    print(f"   > Analisando lote de {len(batch_paths)} imagens com Gemini...")
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

        3.  **Comparar Modelo:** Identifique quais imagens no lote (incluindo a imagem chave) mostram o *mesmo modelo (mesma referência)* de óculos. **Incluir a cor** para esta etapa; foque-se no formato, design e cor.

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
        print(f"   ! Erro na chamada à API Gemini (process_batch): {e}")
        log_event("ERRO_API", f"Chamada à API falhou: {e}")
        return None
    finally:
        # Garante que todas as imagens PIL sejam fechadas
        [img.close() for img in pil_images]


# --- Lógica Principal (Atualizada) ---

def main():
    print("Iniciando script de processamento de óculos...")
    print(f"Pasta de Entrada: {PASTA_ENTRADA}")
    print(f"Pasta de Saída: {PASTA_SAIDA}")
    log_event("SCRIPT_START", "Script iniciado.")

    # 1. Setup inicial
    PASTA_SAIDA.mkdir(exist_ok=True)
    
    # Guarda falhas (corrompidas, falhas de API) APENAS desta sessão
    # para evitar loops infinitos dentro da mesma execução.
    # Não é lido do disco.
    failed_files_session = set()
    
    tipos_de_imagem = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff')
    batch_counter = 0

    # 4. Processar lotes (agora com um loop while)
    while True:
        # 2. Encontrar arquivos para processar (DENTRO DO LOOP)
        # Lê todos os ficheiros da pasta
        all_files = []
        for tipo in tipos_de_imagem:
            all_files.extend(PASTA_ENTRADA.glob(tipo))
            
        # Carrega os dados de SUCESSO (do JSON) a cada iteração
        all_data = load_data()
        files_in_json = load_processed_files_from_json(all_data)
        
        # Combina ficheiros de SUCESSO (do json) com falhas DESTA SESSÃO
        all_skipped_files = files_in_json.union(failed_files_session)

        # Filtra usando o set combinado 'all_skipped_files'
        unprocessed_files = sorted([p for p in all_files if p.name not in all_skipped_files])
        
        if not unprocessed_files:
            print("Nenhum arquivo novo para processar.")
            break # Sai do loop while

        total_files_remaining = len(unprocessed_files)
        total_batches_remaining = (total_files_remaining + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"\nEncontrados {total_files_remaining} arquivos novos para processar.")

        # 3. Criar lote dinâmico
        current_batch = unprocessed_files[:BATCH_SIZE]
        
        batch_counter += 1
        print(f"\n--- Processando Lote {batch_counter} (Aprox. {total_batches_remaining} lotes restantes) ---")
        
        # 4.1. Validar imagens ANTES de enviar à API
        valid_images_in_batch = []
        for img_path in current_batch:
            if validate_image(img_path):
                valid_images_in_batch.append(img_path)
            else:
                # Se a imagem é inválida, 'validate_image' já logou.
                # Adicionamos ao set em memória para a próxima iteração do while.
                failed_files_session.add(img_path.name) 

        if not valid_images_in_batch:
            print("   [AVISO] Nenhuma imagem válida neste lote. A saltar.")
            continue # Próxima iteração do loop while

        # print the name of images being sent
        print("   Imagens válidas neste lote:")
        for img in valid_images_in_batch:
            print(f"     - {img.name}")

        #time.sleep(30)
        # 4.2. Chamar a API (UMA SÓ VEZ)
        api_result = call_gemini_process_batch(valid_images_in_batch)

        # 4.3. Processar resultado da API
        if not api_result or not api_result.get("data"):
            print("   [FALHA] Nenhuma imagem chave encontrada pela API neste lote.")
            # Loga o lote original
            log_event("FALHA_API", "Nenhuma imagem chave encontrada no lote", [p.name for p in current_batch])
            
            # Adiciona todos os ficheiros deste lote ao set de falhas da sessão
            for img_path in current_batch:
                failed_files_session.add(img_path.name)
            continue # Próxima iteração do loop while

        # 4.4. Sucesso - Extrair dados da resposta
        parsed_data = api_result["data"]
        key_image_name = api_result["key_image_name"]
        
        key_image_path = None
        for p in valid_images_in_batch:
            if p.name == key_image_name:
                key_image_path = p
                break
        
        if not key_image_path:
             print(f"   [ERRO] API retornou key_image '{key_image_name}' mas não foi encontrado no lote. A saltar.")
             log_event("ERRO_INTERNO", f"API retornou '{key_image_name}' mas não foi encontrado", [p.name for p in current_batch])
             # Loga todos os ficheiros do lote como "processados" (falhados)
             for img_path in current_batch:
                 failed_files_session.add(img_path.name)
             continue

        print(f"   [SUCESSO] Imagem Chave encontrada pela API: {key_image_name}")
        print(f"   Dados extraídos: {parsed_data}")

        matched_filenames = api_result["matched_filenames"]
        matched_paths = [p for p in current_batch if p.name in matched_filenames]
        
        # 4.5. Organizar arquivos
        reference_clean = str(parsed_data["reference"]).replace('/', '_').replace('\\', '_').strip()
        if not reference_clean:
                print(f"   [ERRO] Referência extraída está vazia. A ignorar lote.")
                log_event("ERRO_DADOS", "Referência extraída está vazia", [p.name for p in current_batch])
                # Loga todos os ficheiros do lote como "processados" (falhados)
                for img_path in current_batch:
                    failed_files_session.add(img_path.name)
                continue
                
        target_dir = PASTA_SAIDA / reference_clean
        target_dir.mkdir(exist_ok=True)
        
        print(f"   Movendo {len(matched_paths)} arquivos para {target_dir}")
        
        i = 0
        for file_path in matched_paths:
            try:
                shutil.copy(str(file_path), str(target_dir / f"{parsed_data['reference']}_{parsed_data['color']}_{str(i)}.jpg"))
            except Exception as e:
                print(f"   [ERRO] Falha ao mover {file_path.name}: {e}")
                log_event("ERRO_MOVIMENTO", f"Falha ao mover {file_path.name}: {e}", [file_path.name])
                if (target_dir / file_path.name).exists():
                        print(f"   Ficheiro já existia no destino.")
            i += 1
        
        # 4.6. Salvar dados (no novo formato)
        # Prepara os ficheiros adicionais (todos os 'matched' exceto o 'key')
        additional_files = [p.name for p in matched_paths if p.name != key_image_name]
        save_data(all_data, parsed_data, key_image_name, additional_files)
        
        # Loga o sucesso
        log_event("SUCESSO", f"Ref {parsed_data['reference']} Cor {parsed_data['color']}", [p.name for p in matched_paths])

        
        # 4.7. Pausa (Req 4)
        # Verifica se a pausa é necessária e se ainda há ficheiros para processar
        # Recalcula os ficheiros restantes (sem os que acabaram de ser processados com sucesso)
        # remaining_after_success = len(unprocessed_files) - len(matched_paths)
        
        # if batch_counter % PAUSE_AFTER_BATCHES == 0 and remaining_after_success > 0:
        #     print(f"\n--- Pausado após {batch_counter} lotes. ---")
        #     try:
        #         input("Pressione Enter para continuar (ou CTRL+C para parar)...")
        #     except KeyboardInterrupt:
        #         print("\nProcessamento interrompido pelo usuário.")
        #         log_event("SCRIPT_STOP", "Script interrompido pelo usuário.")
        #         break # Sai do loop while

    print("\nProcessamento concluído.")
    log_event("SCRIPT_END", "Processamento concluído.")

if __name__ == "__main__":
    if not PASTA_ENTRADA.is_dir():
        print(f"ERRO: A pasta de entrada '{PASTA_ENTRADA}' não foi encontrada.")
    else:
        main()