# TSP Solver com Múltiplos Veículos usando Algoritmo Genético

Este repositório contém uma implementação Python de um solver para o Problema do Caixeiro Viajante (TSP) com múltiplos veículos usando Algoritmo Genético (GA). O sistema utiliza clustering para dividir cidades entre veículos e otimiza as rotas simultaneamente com visualização em tempo real.

## 🎯 Visão Geral

O solver emprega um Algoritmo Genético para evoluir iterativamente uma população de soluções candidatas em direção a rotas ótimas ou quase-ótimas. O sistema suporta:

- **Múltiplos Veículos**: Até 5 veículos operando simultaneamente
- **Clustering Inteligente**: Divisão automática de cidades usando K-Means
- **Restrições Configuráveis**: Vias proibidas, cidades prioritárias e limitação de distância
- **Visualização em Tempo Real**: Interface gráfica interativa com Pygame
- **Elitismo Avançado**: Preservação dos 5 melhores indivíduos entre gerações

## 📁 Estrutura do Projeto

```
src/
├── tsp.py                  # Aplicação principal com interface Pygame
├── genetic_algorithm.py    # Operações do algoritmo genético (seleção, crossover, mutação)
├── city.py                 # Classe City para representação de cidades
├── draw_functions.py       # Funções de desenho e visualização
├── benchmark_att48.py      # Dataset benchmark ATT48 (48 cidades)
├── llm_integration.py      # Integração com LLM para análise de resultados (opcional)
└── ui.py                   # Componentes de UI (área de scroll, markdown)
```

## 🚀 Funcionalidades Principais

### Algoritmo Genético
- **População**: 100 indivíduos por veículo
- **Elitismo**: A melhor solução preservada por geração
- **Seleção**: Torneio
- **Crossover**: Order Crossover (OX)
- **Mutação**: Taxa de 50% com swap de cidades
- **Inicialização**: Heurística nearest-neighbor + população aleatória

### Restrições Operacionais
1. **Via Proibida**: Proíbe rotas específicas entre pares de cidades
2. **Cidade Prioritária**: Força visita prioritária a certas cidades logo após o depósito
3. **Limitação de Distância**: Adiciona paradas obrigatórias em postos de abastecimento quando a distância acumulada excede 900 unidades

### Interface Interativa
- **Painel Esquerdo**: Gráfico de evolução de fitness e tabela de informações dos veículos
- **Painel Direito**: Visualização do mapa com rotas coloridas por veículo
- **Controles**: 
  - Checkboxes para ativar/desativar restrições
  - Inputs para configurar número de veículos (1-5) e cidades (8-48)
  - Botões EDITAR e RESET para configuração em tempo real

## 📦 Dependências

```bash
numpy>=1.21.0
pygame>=2.0.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
openai>=1.0.0           # Opcional para integração LLM
python-dotenv>=1.0.0    # Opcional para integração LLM
screeninfo>=0.8.1
```

## 🔧 Instalação

1. Clone o repositório:
```bash
git clone https://github.com/Gusta-Alves/tsp-genetic-algorithm.git
cd tsp-genetic-algorithm
```

2. Crie e ative um ambiente virtual:
```bash
python -m venv .venv
source .venv/bin/activate  # No Windows: .venv\Scripts\activate
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

## 🎮 Como Usar

Execute o arquivo principal:
```bash
python src/tsp.py
```

### Controles da Interface

- **Checkboxes**: Clique para ativar/desativar restrições (disponível apenas para 4 veículos e 48 cidades)
- **Botão EDITAR**: Ativa modo de edição para modificar número de veículos e cidades
- **Botão RESET**: Restaura configurações padrão (4 veículos, 48 cidades, sem restrições)
- **Tecla Q**: Fecha a aplicação
- **Clique X**: Fecha a aplicação

### Configuração Personalizada

Durante o modo de edição, você pode:
- Definir de 1 a 5 veículos
- Definir de 8 a 48 cidades (mínimo = veículos × 2)
- Aplicar ou remover restrições operacionais

## 📊 Dataset

O projeto utiliza o benchmark **ATT48** com 48 cidades, um problema clássico de TSP amplamente usado para testes de algoritmos de otimização.

## 🎨 Visualização

- Cada veículo possui uma cor única
- O depósito central é calculado como o centroide de todas as cidades
- Rotas são desenhadas em tempo real mostrando a evolução do algoritmo
- Gráfico de fitness mostra a convergência ao longo das gerações
- Tabela exibe distância, número de cidades e última mudança por veículo

## 🤖 Integração LLM (Opcional)

O sistema pode integrar-se com modelos de linguagem (OpenAI) para análise e formatação de resultados. Para ativar:

1. Configure `_isllmintegrationEnabled = True` em `tsp.py`
2. Crie arquivo `.env` com sua chave API:
```
OPENAI_API_KEY=sua_chave_aqui
```

## 📈 Parâmetros Configuráveis

Edite as constantes em `tsp.py`:

```python
POPULATION_SIZE = 100        # Tamanho da população por veículo
MUTATION_PROBABILITY = 0.5   # Taxa de mutação
NUM_VEHICLES = 4             # Número de veículos (1-5)
NUM_CITIES = 48              # Número de cidades (8-48)
MAX_DISTANCE = 900           # Distância máxima sem reabastecimento
MAX_GENERATIONS = 100        # Número máximo de gerações
```

## 🏗️ Arquitetura

### Classe City
```python
@dataclass(frozen=True)
class City:
    name: str
    x: int
    y: int
```
Representação imutável de cidades com hash automático para uso em sets e dicionários.

### Fluxo Principal
1. **Inicialização**: Carrega cidades do benchmark ATT48 e aplica clustering K-Means
2. **Preparação**: Cria populações iniciais para cada veículo usando heurísticas
3. **Evolução**: Para cada geração, aplica seleção, crossover e mutação
4. **Elitismo**: Preserva os 5 melhores indivíduos de cada população
5. **Visualização**: Atualiza interface em tempo real a 30 FPS
6. **Restrições**: Aplica penalidades e ajustes conforme configurações ativas

## 🎓 Conceitos de Algoritmo Genético

- **Fitness**: Soma das distâncias da rota (menor é melhor)
- **Seleção por Torneio**: Escolhe os melhores indivíduos de subgrupos aleatórios
- **Order Crossover (OX)**: Preserva ordem relativa de cidades dos pais
- **Mutação por Swap**: Troca aleatória de posições de duas cidades
- **Elitismo**: Garante que as melhores soluções não sejam perdidas

## 🤝 Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para:
- Reportar bugs
- Sugerir melhorias
- Adicionar novas funcionalidades
- Melhorar a documentação

## 📄 Licença

Este projeto está licenciado sob a [MIT License](LICENSE).

## 👥 Autores

Desenvolvido como projeto de otimização combinatória utilizando algoritmos genéticos e técnicas de clustering para resolução do TSP com múltiplos veículos.

---

**Nota**: Este é um projeto educacional que demonstra a aplicação de algoritmos genéticos em problemas de otimização de rotas com restrições operacionais realistas.