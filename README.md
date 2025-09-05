project:
  title: "🎬 LH_CD_LUIZ_CARLOS — Desafio Cientista de Dados (IMDB)"
  author: "LUIZ CARLOS"
  summary: >
    Projeto desenvolvido como parte do Desafio Cientista de Dados da Indicium.
    O objetivo é realizar uma análise exploratória de dados (EDA), responder
    a perguntas de negócio e treinar um modelo preditivo capaz de estimar a
    nota do IMDB de filmes. A base de dados contém informações de filmes como
    título, ano, duração, gênero, overview, votos, receita, elenco etc.

structure:
  tree: |
    LH_CD_LUIZ_CARLOS/
    ├── README.md
    ├── requirements.txt
    ├── notebooks/
    │   └── LH_CD_LUIZ_CARLOS_EDA_Model.ipynb   # notebook executado (EDA + modelagem)
    ├── reports/
    │   └── EDA_IMDB.pdf                        # relatório exportado em PDF
    ├── models/
    │   └── model_imdb.pkl                      # modelo treinado salvo
    └── data/
        └── desafio_indicium_imdb.csv           # (opcional; se permitido compartilhar)

how_to_run:
  colab:
    description: "Opção recomendada, simples e reprodutível."
    steps:
      - "Abra o Google Colab."
      - "Carregue notebooks/LH_CD_LUIZ_CARLOS_EDA_Model.ipynb."
      - "Na primeira célula, faça upload de desafio_indicium_imdb.csv (ou use a pasta /data)."
      - "Execute todas as células em ordem (Runtime > Run all)."
      - "O modelo será treinado e salvo como models/model_imdb.pkl."
      - "Para baixar o modelo:"
      - |
        ```python
        from google.colab import files
        files.download("models/model_imdb.pkl")
        ```
      - "Exporte o PDF: Arquivo > Fazer download > Download .pdf, salve como reports/EDA_IMDB.pdf."
  local:
    description: "Execução em ambiente local (Python 3.10+)."
    commands: |
      git clone https://github.com/SEU_USUARIO/LH_CD_LUIZ_CARLOS.git
      cd LH_CD_LUIZ_CARLOS
      python -m venv .venv
      # Windows:
      .venv\Scripts\activate
      # Linux/Mac:
      source .venv/bin/activate
      pip install -r requirements.txt
      jupyter notebook notebooks/LH_CD_LUIZ_CARLOS_EDA_Model.ipynb

deliverables:
  - "Notebook executado: notebooks/LH_CD_LUIZ_CARLOS_EDA_Model.ipynb"
  - "Relatório em PDF: reports/EDA_IMDB.pdf"
  - "Modelo salvo: models/model_imdb.pkl"
  - "Arquivo de requisitos: requirements.txt"

eda_and_hypotheses:
  - "Distribuições de variáveis numéricas (notas, metascore, votos, receita)."
  - "Correlação entre metascore, votos, receita e IMDB_Rating."
  - "Tendência temporal das notas de filmes."
  - "Comparações por gênero e certificado."
  - "Principais diretores e atores."
  - "Hipóteses: popularidade influencia receita; crítica acompanha público;
     duração fora de ~100–140min pode afetar aceitação; overview contém sinais de gênero."

questions_answered:
  - "Qual filme recomendar: top 10 por score (nota × alcance via log(votos))."
  - "Principais fatores de faturamento: votos, metascore, ano de lançamento."
  - "Insights do Overview: possível inferir gêneros com TF-IDF + Regressão Logística."
  - "Previsão da nota IMDB: regressão com numéricas + categóricas + texto; métricas RMSE/MAE."
  - "Caso Shawshank: nota prevista próxima da real."

model:
  type: "HistGradientBoostingRegressor dentro de Pipeline"
  features:
    numeric: ["metascore","runtime_min","year","log_votes","log_gross"]
    categorical: ["Certificate","primary_genre","Director","Star1","Star2","Star3","Star4"]
    text: "Overview (TF-IDF com n-gramas 1–2)"
  metrics: ["RMSE", "MAE"]
  artifact: "models/model_imdb.pkl"

requirements:
  packages:
    - "pandas>=2.0.0"
    - "numpy>=1.24.0"
    - "scikit-learn>=1.3.0"
    - "matplotlib>=3.7.0"
    - "joblib>=1.3.0"
    - "jupyter>=1.0.0"

observations:
  - "O PDF deve ser gerado com o notebook executado (códigos, gráficos e saídas)."
  - "Se o dataset for sensível, não inclua em data/, apenas explique como carregar no Colab."
  - "Pipeline garante reprodutibilidade: mesmas transformações no treino e predição."
  - "Modelo salvo pode ser recarregado com:"
  - |
    ```python
    import joblib
    model = joblib.load("models/model_imdb.pkl")
    ```
