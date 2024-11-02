# CAS-ViT com Adapters

## Objetivo

Este repositório contém uma versão adaptada da rede neural [CAS-ViT](https://github.com/Tianfang-Zhang/CAS-ViT), com o objetivo de incorporar adapters em cada bloco de sua arquitetura. A rede CAS-ViT foi congelada, permitindo que apenas os adapters sejam treináveis.

## Tarefas e Etapas do Projeto

### 1. Construção do Dataset

- Coletar imagens utilizando um dispositivo móvel, com pelo menos 200 exemplos para cada classe escolhida (as classes são de livre escolha).
- Realizar a divisão dos dados em treino, validação e teste nas seguintes proporções:
  - **Treino:** 70%
  - **Validação:** 15%
  - **Teste:** 15%

### 2. Adaptação da Arquitetura do CAS-ViT

Foi realizada uma modificação na arquitetura do CAS-ViT para incluir adapters em cada bloco. Para isso, foi utilizada a implementação do modelo disponível em [CVPR24-Ease](https://github.com/sun-hailong/CVPR24-Ease), especificamente nas funções `forward_train` e `forward_test`, que foram adaptadas para suportar os adapters. Os seguintes arquivos foram alterados:

- **`engine.py` e `main.py`:** Atualizados para incluir o `classification_report`, proporcionando uma avaliação detalhada do modelo.
- **`datasets.py`:** Corrigido um bug de indentação que afetava o processo de carregamento de dados.
- **`rcvit.py`:** Modificado para adicionar os adapters na arquitetura da CAS-ViT.

### 3. Avaliação Experimental

A avaliação foi realizada em dois modelos:

1. **Modelo CAS-ViT:** Ajustado com o conjunto de dados criado.
2. **Modelo CAS-ViT + Adapter:** Ajustado com o mesmo conjunto de dados, mas com a arquitetura modificada para incluir adapters.

Foram realizadas duas avaliações:

- **Avaliação 1:** Teste no conjunto de dados personalizado e geração de um `classification_report` utilizando a função disponível em [sklearn.metrics.classification_report](https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.classification_report.html).
- **Avaliação 2:** Teste dos modelos no conjunto **ImageNet-A** disponível em [CVPR24-Ease](https://github.com/sun-hailong/CVPR24-Ease) e geração do `classification_report`.

Os resultados foram comparados e discutidos, observando o impacto do uso de adapters em uma rede pré-treinada e congelada.

## Resultados e Discussão

Uma análise dos resultados obtidos com ambos os modelos é apresentada neste repositório. Os detalhes de desempenho podem ser encontrados na pasta `Resultados/`.

### Referências

- Código original da CAS-ViT: [CAS-ViT GitHub](https://github.com/Tianfang-Zhang/CAS-ViT)
- Referência para implementação dos adapters: [CVPR24-Ease](https://github.com/sun-hailong/CVPR24-Ease)
