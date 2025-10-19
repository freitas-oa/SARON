# test_api.py
import google.generativeai as genai

# !!! IMPORTANTE !!!
# COLE A SUA CHAVE DE API VÁLIDA AQUI
GOOGLE_API_KEY = "AIzaSyCFrs_EVe2AAZ63f7wTRFxcIEc8ztF9iu4"

try:
    genai.configure(api_key=GOOGLE_API_KEY)

    print("="*50)
    print("Conexão bem-sucedida! A listar modelos disponíveis...")
    print("="*50)

    # Lista todos os modelos que suportam o método 'generateContent'
    models = [m for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]

    if not models:
        print("\nNenhum modelo compatível encontrado para esta chave de API.")
        print("Verifique as permissões da sua chave no Google Cloud Console.")
    else:
        print("\nCopie um dos seguintes nomes de modelo para o seu script principal:")
        for model in models:
            # O nome a ser usado é a parte depois de "models/"
            nome_para_usar = model.name.split('/')[-1]
            print(f" -> '{nome_para_usar}'")

    print("\n" + "="*50)


except Exception as e:
    print("ERRO: A chave de API é inválida ou há um problema de conexão.")
    print("Detalhes do erro:", e)