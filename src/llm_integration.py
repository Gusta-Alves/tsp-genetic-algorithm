from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_llmSolution(solution):
    system = (
        "Você é um assistente que explica soluções de otimização de rotas "
        "de forma clara, concisa e em português do Brasil. Evite jargão."
    )
    user = (
        "Explique a solução do problema do caixeiro-viajante com múltiplos caixeiros, "
        "em até 6 parágrafos curtos, destacando: lógica geral, custo total, "
        "resumo de cada rota e quaisquer observações/restrições relevantes. "
        "Se houver oportunidades de melhoria (ex.: balancear distâncias entre caixeiros), mencione."
        f"\n\nDados (JSON):\n{solution}"
    )

    response = client.chat.completions.create(
        model="gpt-4.1-nano",  
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}],
        temperature=0.2
    )
    print("Resposta do LLM:\n", response.choices[0].message.content)
    return response.choices[0].message.content