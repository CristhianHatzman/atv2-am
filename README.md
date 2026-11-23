# AT2 — Redes Neurais MLP e Deploy de API em Produção

Projeto acadêmico de classificação binária para prever o cancelamento (`Churn`) de clientes de telecomunicações. Ele documenta a exploração de dados, treina uma rede neural MLP em um pipeline Scikit-Learn e disponibiliza inferências por uma API FastAPI.

## Dataset

Foi utilizada a base original [Telco Customer Churn, de blastchar no Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), salva em `data/dataset.csv`. A base tem 7.043 registros e 21 colunas. O arquivo original foi baixado do Kaggle; `customerID` não participa do modelo por ser apenas identificador.

As oito features do pipeline são `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `InternetService`, `OnlineSecurity`, `TechSupport` e `PaymentMethod`. A variável-alvo é `Churn`.

## Tecnologias

Python, Pandas, NumPy, Matplotlib, Scikit-Learn, Joblib, FastAPI, Pydantic, Jupyter e UV.

## Estrutura

```text
am-at2-redes-neurais/
├── data/dataset.csv
├── models/modelo_mlp.joblib
├── notebooks/analise_treinamento.ipynb
├── src/
│   ├── __init__.py
│   ├── app.py
│   └── schemas.py
├── pyproject.toml
├── uv.lock
├── requisicoes.http
├── README.md
└── .gitignore
```

## Instalação e reprodução

É necessário Python 3.11 a 3.14 e o [UV](https://docs.astral.sh/uv/). Na raiz do projeto, execute:

```bash
uv sync
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/analise_treinamento.ipynb --ExecutePreprocessor.timeout=180
```

O segundo comando executa o notebook inteiro, cria/atualiza `models/modelo_mlp.joblib` e registra as saídas das análises. O notebook funciona quando iniciado na raiz ou a partir da pasta `notebooks`.

## Métricas obtidas

Execução realizada com divisão estratificada de teste de 20%, `random_state=42`, MLP `(64, 32)`, `early_stopping=True` e `max_iter=500`:

| Métrica | Resultado |
|---|---:|
| Acurácia | 0,7999 |
| Precision — No | 0,8271 |
| Recall — No | 0,9198 |
| F1 — No | 0,8710 |
| Precision — Yes | 0,6783 |
| Recall — Yes | 0,4679 |
| F1 — Yes | 0,5538 |

Foram executadas 90 iterações. A matriz de confusão e o relatório completo estão no notebook. Um falso positivo classifica um cliente como provável cancelador quando ele não cancela; um falso negativo deixa de sinalizar alguém que poderia cancelar.

## API

Inicie o servidor com o comando exigido:

```bash
uv run fastapi dev src/app.py
```

Abra a documentação Swagger em <http://127.0.0.1:8000/docs>.

- `GET /info` retorna o estado e metadados do serviço.
- `POST /predict` recebe as oito features listadas acima e retorna, por exemplo, `{"classe_predita":"Yes","probabilidade":0.65}`. A probabilidade é a confiança da classe efetivamente prevista.

O pipeline é carregado uma única vez no `lifespan`; a API nunca retreina nem relê o arquivo do modelo a cada chamada.

## Testes HTTP

Com o servidor em execução, abra `requisicoes.http` no VS Code e use a extensão REST Client em cada requisição. O arquivo inclui `/info`, dois perfis reais previamente validados com classes distintas (`No` e `Yes`) e uma entrada inválida que retorna HTTP 422. Todas as requisições enviam `User-Agent: Client-Avaliativo`.

## Limitações

O conjunto possui desbalanceamento de classes e o recall de `Yes` é moderado; portanto, parte dos cancelamentos reais não é identificada. O modelo não substitui uma decisão de retenção e deve ser reavaliado se o perfil de clientes ou as políticas comerciais mudarem.
