# SynthNET: Detecção de Deepfakes com Análise Híbrida Espacial e de Frequência

Este repositório contém o código, os dados e os resultados do projeto que propõe uma nova abordagem para a detecção de deepfakes. A arquitetura utiliza uma combinação de análise espacial e de frequência, explorando redes neurais convolucionais e características derivadas da transformada rápida de Fourier (FFT).

---

## Arquitetura da Rede

A arquitetura combina duas fontes de dados:
1. **Imagens**: Processadas por uma CNN para extração de características espaciais.
2. **Picos de frequência (FFT)**: Extraídos para capturar padrões específicos no domínio da frequência.

Os dados são fusionados em pontos específicos da rede para uma análise conjunta, seguidos por etapas de atenção espacial antes de prosseguir para convoluções adicionais.

<h4>Arquitetura da Rede</h4>
<img src="./assets/img/network_architecture.png" alt="Arquitetura da Rede" width="300">

---

## Resultados

Os experimentos foram conduzidos utilizando um subconjunto do dataset DFDC. A abordagem alcançou:
- **Acurácia de Validação**: 92,66%
- **Acurácia Balanceada no Conjunto de Teste**: 75,25%
- **ROC AUC no Conjunto de Teste**: 83,23%

Embora os resultados iniciais sejam promissores, foi observada uma tendência ao overfitting, indicando que ajustes no conjunto de dados e na arquitetura podem melhorar o desempenho.

![Resultados](./assets/img/synthnet_dfdc_result.png)

---

## Requisitos

- **Linguagem**: Python 3.8+
- **Bibliotecas**:
  - PyTorch
  - NumPy
  - Matplotlib
  - OpenCV
  - scikit-learn

Instale as dependências com o comando:

```bash
pip install -r requirements.txt
```
## Como Executar
### 1. Preparar os Dados

O conjunto de dados utilizado pode ser encontrado em https://www.kaggle.com/datasets/itamargr/dfdc-faces-of-the-train-sample.

Certifique-se de que os dados estão organizados conforme as pastas especificadas:

/data/train.csv
/data/eval.csv
/data/eval.csv

Nota: faça a leitura dos diretórios e organize os caminhos em 3 arquivos `.csv` contendo uma coluna path com os caminhos para as imagens.

### 2. Treinamento

Para treinar o modelo, execute:

```python
python3 train.py ./data/train.csv ./data/eval.csv --ckp ./experiments/exp_1/weights/synthnet-best.pth --b_size 64 --img_dim 224 --epochs 2 --out exp_2
```
### 3. Avaliação

Para avaliar o modelo treinado, use:
```python
python3 test.py ./data/test.csv --ckp ./experiments/exp_1/weights/synthnet-best.pth --b_size 128 --img_dim 224 --out exp_1
```
