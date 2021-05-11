from itertools import permutations
from typing import Any, List, Optional, Tuple, Dict
from functools import lru_cache

import numpy as np


def compute_permutation_distance(
    distance_matrix: np.ndarray, permutation: List[int]
) -> float:
    ind1 = permutation
    ind2 = permutation[1:] + permutation[:1]
    return distance_matrix[ind1, ind2].sum()


def solve_tsp_brute_force(
    distance_matrix: np.ndarray
) -> Tuple[Optional[List], Any]:

    # Exclude 0 from the range since it is fixed as starting point
    points = range(1, distance_matrix.shape[0])
    best_distance = np.inf
    best_solution = None

    for partial_permutation in permutations(points):
        # Remember to add the starting node before evaluating it
        permutation = [0] + list(partial_permutation)
        distance = compute_permutation_distance(distance_matrix, permutation)

        #verifica se a distancia calculada é menos qua a menor atual
        if distance < best_distance:
            best_distance = distance
            best_solution = permutation

    return best_solution, best_distance


def solve_tsp_dynamic_programming(
    distance_matrix: np.ndarray,
    maxsize: Optional[int] = None
) -> Tuple[List, float]:
    
    N = frozenset(range(1, distance_matrix.shape[0])) # ({1, 2, 3, 4, ... n-1})
    
    #define a estrutura que será salvo os caminhos memo(ry) 
    #a chave é uma tupla que guarda um inteiro
    #a chave será uma tupla de (int, range()) que guarda um inteiro. ex: [(1, {2,3,4,5,6..,n-1}), int]
    memo: Dict[Tuple, int] = {}

    # Step 1: get minimum distance
    @lru_cache(maxsize=None)
    def dist(ni: int, N: frozenset) -> float:

        #se for o último nó da permutação retorna o valor para o primeiro nó (para a formação do ciclo hamiltoniano)
        if not N:
            return distance_matrix[ni, 0]

        # guarda os custos de cada caminho recursivamente
        costs = []

        for nj in N:
            cost = (nj, distance_matrix[ni, nj] + dist(nj, N.difference({nj})))
            costs.append(cost)
        
        #função para pegar o custo mínimo
        nmin, min_cost = min(costs, key = lambda x: x[1])

        #salva o nó com de menor caminho total disponível (próxima cidade com menor caminho)
        #memo[(5, {1,2,3,4,6,7})] = 3
        memo[(ni, N)] = nmin

        #retona o valor da menor distância do caminho
        return min_cost

    #melhor distancia
    best_distance = dist(0, N)

    # Passo 2: pegar o caminho de menor tamanho
    ni = 0  # começa da origem (0, {1,2,3,4,5,6,7, ..., matrix.tamanho -1})
    solution = [0]
    while N:
        #pega memo[(0, {1,2,3,4,5,6..,n-1})] na primeira rodada
        ni = memo[(ni, N)]
        #adiciona o próximo nodo da sequencia de menor caminho
        solution.append(ni)
        #retira o numero da sequência da permutação
        N = N.difference({ni})

    #retorna a solução(vetor da sequencia) e o menor caminho
    return solution, best_distance

