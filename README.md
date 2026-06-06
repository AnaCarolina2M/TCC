# TCC"

Código do Projeto de Conclusão de Curso referente ao artigo "Técnicas de Machine Learning na identificação do Transtorno do Espectro Autista em Adultos"

O Repositório está organizado na seguinte maneira:
- /data: Dados brutos
- /data/xai: resultado das análises realizadas pelo SHAP
- /experiment: classes orquestradoras do projeto
- /mlflow_log: logs de mlflow
- /mlruns: logs de mlflow
- /models: classes de implementação dos modelos
- /pipeline: pipeline de limpeza dos dados
- /strategies: definição das classes abstratasdo projeto

## Como executar o projeto:
Executar o comando "python -m experiment.main" no terminal

## Logs do MLflow
Executar o comando "mlflow ui" no terminal
