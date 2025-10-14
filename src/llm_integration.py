from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_llmSolution(tsp_output):
    prompt = f"""
    Você é um assistente de análise de otimização de rotas.
    Abaixo estão os resultados brutos de um TSP com múltiplos veículos.
    Gere uma resposta em formato estruturado contendo:

    1. Resumo geral (número de veículos, total de cidades, distância total) (somar dados de todos os veículos)
    2. Ranking de eficiência (melhor e pior veículo)
    3. Métricas de balanceamento (diferença percentual e média)
    4. Tabela simples com cada veículo (distância, #cidades)
    5. Rotas completas (concisas, separadas por →)
    6. Um parágrafo final tipo relatório executivo (máx. 5 linhas)

    Dados:
    {tsp_output}
    """

    # response = client.chat.completions.create(
    #     model="gpt-4.1-nano",  
    #     messages=[
    #         {"role": "system", "content": "Você é um formatador de resultados de otimização de rotas."},
    #         {"role": "user", "content": prompt}
    #     ],
    #     temperature=0.1
    # )

    return "aasdasdasdasdas "