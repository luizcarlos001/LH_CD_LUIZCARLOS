# LH_CD_LUIZCARLOS — Desafio Cientista de Dados (IMDB)

Este repositório contém a solução do **Desafio Cientista de Dados da Indicium**.  
O objetivo é realizar uma **Análise Exploratória de Dados (EDA)**, responder a perguntas de negócio e treinar um **modelo preditivo** para estimar a nota do IMDB de filmes.  

---

## 📂 Estrutura do Projeto

LH_CD_LUIZCARLOS/
├── README.md
├── requirements.txt
├── notebooks/
│ └── LH_CD_LUIZ_CARLOS_EDA_Model.ipynb # notebook completo (EDA + modelagem)
├── reports/
│ └── EDA_IMDB.pdf # relatório exportado do notebook
├── models/
│ └── model_imdb.pkl # modelo salvo
└── data/
└── desafio_indicium_imdb.csv # (opcional, caso permitido)


---

## 🚀 Como Executar

### 🔹 Opção 1 — Google Colab (recomendado)
1. Abra o [Google Colab](https://colab.research.google.com/).  
2. Carregue `notebooks/LH_CD_LUIZ_CARLOS_EDA_Model.ipynb`.  
3. Na primeira célula, faça upload do arquivo `desafio_indicium_imdb.csv`.  
4. Execute todas as células na ordem (**Runtime > Run all**).  
5. O modelo será salvo em `models/model_imdb.pkl`. Para baixar:  
   ```python
   from google.colab import files
   files.download("models/model_imdb.pkl")

Exporte o relatório em PDF: Arquivo > Fazer download > Download .pdf e salve em reports/EDA_IMDB.pdf.

# Opção 2 

# clone o repositório
git clone https://github.com/luizcarlos001/LH_CD_LUIZCARLOS.git
cd LH_CD_LUIZCARLOS

# crie e ative o ambiente virtual
python -m venv .venv
## Windows:
.venv\Scripts\activate
## Linux/Mac:
source .venv/bin/activate

## instale dependências
pip install -r requirements.txt

## abra o notebook
jupyter notebook notebooks/LH_CD_LUIZ_CARLOS_EDA_Model.ipynb


### Conteúdo do Trabalho
EDA (Análise Exploratória)
---

Distribuições de variáveis numéricas (notas, metascore, votos, receita).

Correlações entre críticas, votos, receita e nota do IMDB.

Tendências temporais e comparações por gênero e certificado.

Destaque de diretores e atores mais recorrentes.

### Hipóteses Investigadas
---

Popularidade (número de votos) influencia receita e percepção de qualidade.

Avaliações da crítica (Meta_score) acompanham a opinião do público.

Filmes muito longos ou muito curtos podem reduzir aceitação.

O texto do Overview contém sinais que permitem inferir gênero.

 ### Respostas às Perguntas do Desafio
 ---

Qual filme recomendar para uma pessoa desconhecida?
Ranking imparcial por nota × alcance, resultando em um top 10 equilibrado.

Principais fatores de faturamento:
Votos, crítica especializada (Meta_score) e ano de lançamento têm forte correlação com receita.

O que o Overview revela? É possível inferir gênero?
Sim. Usando TF-IDF + Regressão Logística é possível prever gêneros predominantes.

Como prever a nota do IMDB?

Tipo de problema: regressão.

Variáveis utilizadas: numéricas (metascore, votos, receita, duração, ano), categóricas (gênero, certificado, diretor, estrelas) e texto (Overview).

Modelo escolhido: HistGradientBoostingRegressor em pipeline.

Métricas: RMSE e MAE.

Pontos fortes: lida bem com variáveis heterogêneas; pontos fracos: maior custo de treinamento.

Caso “The Shawshank Redemption”:
O modelo previu uma nota muito próxima da real, validando sua robustez.

### Modelo Final
---

Algoritmo: HistGradientBoostingRegressor (scikit-learn).

Pipeline:

Numéricas → imputação + padronização.

Categóricas → imputação + one-hot encoding.

Texto (Overview) → TF-IDF com n-gramas 1–2.

Métricas avaliadas: RMSE e MAE.

Artefato salvo: models/model_imdb.pkl.

Para recarregar o modelo:
import joblib
model = joblib.load("models/model_imdb.pkl")

### Requisitos
---

As dependências estão listadas em requirements.txt:
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
joblib>=1.3.0
jupyter>=1.0.0

### Observações
---

O relatório PDF deve ser gerado com o notebook executado (mostrando gráficos, tabelas e saídas).

Se o dataset não puder ser incluído, basta subir no Colab quando solicitado.

O pipeline garante que as mesmas transformações usadas no treino sejam aplicadas em predição.

---
