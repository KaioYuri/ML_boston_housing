# Projeto de Machine Learning com TensorFlow

Este projeto utiliza TensorFlow para prever os preços das casas usando o conjunto de dados "Boston Housing".

## Estrutura do Projeto

- `notebooks/`: Jupyter notebooks para experimentação e visualização.
- `requirements.txt`: Lista de dependências do projeto.
- `README.md`: Documentação do projeto.

## Configuração

1. Clone o repositório:
    ```sh
    git clone https://github.com/KaioYuri/ML_boston_housing.git
    cd ML_boston_housing
    ```

2. Crie e ative um ambiente virtual:
    ```sh
    python -m venv myenv
    source myenv/bin/activate  # No Windows: .\myenv\Scripts\activate
    ```

3. Instale as dependências:
    ```sh
    pip install -r requirements-dev.txt
    ```

4. Adicione o kernel do ambiente virtual ao Jupyter:
    ```sh
    python -m ipykernel install --user --name=myenv
    ```

## Uso

Para treinar o modelo, execute o script principal:
```sh
python src/main.py
