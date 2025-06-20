# Quantum Mpemba

Este repositório ilustra uma simulação que reproduz parte dos resultados do artigo ``Thermodynamics of the quantum Mpemba effect'', onde discutimos uma receita para ativar o efeito Mpemba em sistemas quânticos.

## Estrutura do código
A hierarquia de módulos para implementar a simulação é a seguinte:
```
simulacao/
└── src
    ├── qtd
    │   ├── energies.py
    │   ├── entropy_and_coherences.py
    │   └── vectorized_lindbladian_and_davies_map.py
    └── qubit_ops
        └── qubit_operators.py
    ├── plot_frufrus
    │   ├── plot_bloch_sphere.py
    │   └── plot_sphere_and_free_energies_evolution.py
```


## Dependências
| :gift: Pacote | :white_check_mark: Versão | :exclamation: Dependências | :information_source: Instalação recomendada | 
| --- | --- | --- | --- |
| NumPy | 2.0 | Python (3+) | pip ou conda |
| Matplotlib | 3.9 | Python | pip ou conda |
| Jupyter | 7.2.2 | instaladas automaticamente | pip |
| QuTip | 4.5 | Python, NumPy, Scipy, Matplotlib, Cython, GCC, headers| pip ou conda|

**Observação**: a última versão do QuTip apresenta um conflito com versões recentes de matplotlib. Utilizamos a função da esfera de Bloch neste exemplo.

## Dados
É possível rodar um exemplo iterativamente e armazenar os resultados da simulação na pasta data/.

## Uso
Pode-se rodar o código ```single_qubit_gelato.py''' com argumentos na linha de comando. A notebook ```exemplo_mpemba_qubit.ipynb''', por sua vez, demonstra o passo a passo da simulação.
