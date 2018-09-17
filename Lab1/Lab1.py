import numpy as np
import pandas as pd

from typing import List, Tuple, NoReturn
from functools import partial
from concurrent.futures import ThreadPoolExecutor, wait


def gaussian_vector(size: int, mean: float = 0.0, std: float = 1.0) -> List[float]:
    return list(np.random.normal(loc=mean, scale=std, size=size))


def mnk_method(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    std_matrix = mat.T.dot(mat)
    if np.linalg.det(std_matrix) < 10 ** - 16:
        raise ValueError("Singular matrix!")
    return np.linalg.inv(std_matrix).dot(mat.T).dot(vec)


def get_variables() -> Tuple[List[float], List[float]]:
    with open('values.txt', 'r') as file:
        a_var = list(map(float, file.readline().split()))
        b_var = list(map(float, file.readline().split()))
    return a_var, b_var


def rmnk_gen(mat: np.ndarray, vec: np.ndarray, alpha: np.ndarray = None, betta: int = 10) -> np.ndarray:
    q_prev = np.zeros((mat.shape[1], 1)) if alpha is None else alpha
    p_prev = betta * np.eye(mat.shape[1])
    iter_ = 0
    while iter_ < mat.shape[0]:
        line_i = np.matrix(mat[iter_])
        p_new = p_prev - (p_prev.dot(line_i.T).dot(line_i.dot(p_prev))) / (1 + line_i.dot(p_prev).dot(line_i.T))
        q_new = q_prev + p_new.dot(line_i.T) * (vec[iter_] - line_i.dot(q_prev)).tolist()[0][0]
        yield q_new.T.tolist()[0]
        q_prev = q_new
        p_prev = p_new
        iter_ += 1


def rmnk_method(mat: np.ndarray, vec: np.ndarray) -> List[float]:
    return [_ for _ in rmnk_gen(mat,  vec)][-1]


def arma_model(time_: int, alpha: List[float], betta: List[float], v_k: List[float], white_noise: List[float],
               p: int=1, q: int=1, series: List[float] = None) -> float:
    
    alpha_ = np.array(alpha[1:p + 1])
    betta_ = np.array(betta[:q])
    res = alpha[0] + v_k[time_]
    if time_ >= q:
        eps_vector = v_k[time_ - q:time_]
        eps_vector = np.array(eps_vector[::-1])
        res += np.dot(betta_, eps_vector)
    if time_ >= p:
        x_vector = series[-p:]
        x_vector = np.array(x_vector)
        res += np.dot(alpha_, x_vector)
    return res + white_noise[time_]


def get_time_series() -> np.ndarray:
    with open('series.txt', 'r') as file:
        res = file.readlines()
    return np.array(list(map(float, res)))


def create_fisher_matrix(series: np.ndarray, v_k: List[float], p: int=1, q: int=1) \
        -> Tuple[np.ndarray, np.ndarray]:
    max_ = max(p, q)
    size = series.size - max_
    res = [[1 for _ in range(size)]]
    for i in range(p):
        res.append(series.tolist()[max_ - i - 1:max_ + size - i - 1])
    for i in range(q + 1):
        res.append(v_k[max_ - i:max_ + size - i])
    return np.array(res).T, series.tolist()[max_:]


def single_model(series_len: np.uint8, alpha: List[float], betta: List[float], v_k: List[float],
                 white_noise: List[float], p: int, q: int) -> NoReturn:
    
    arma_model_ = partial(arma_model, alpha=alpha, betta=betta, v_k=v_k, white_noise=white_noise)
    series = []
    for i in range(series_len):
        series.append(arma_model_(time_=i, p=p, q=q, series=series))
    # SPECIAL FOR IGOR
    parameters_1 = []
    parameters_2 = []
    r2 = []
    ika = []
    ep = []

    r2_r = []
    ika_r = []
    ep_r = []

    for i in range(max(p, q), series_len):
        try:
            x_mat, y = create_fisher_matrix(np.array(series[:i+1]), v_k, p, q)
            parameters_1.append(mnk_method(x_mat, y))
            parameters_2.append(rmnk_method(x_mat, y))
            # Epps Error
            a = mnk_method(x_mat, y)
            erro = [e(i + max(p, q), series, v_k, a, p, q) for i in range(series_len - max(p, q))]
            s1 = np.sum(list(map(lambda x: x ** 2, erro)))
            ep.append(s1)
            # R2
            our_values = series[:max(p, q)] +\
                [new_values(i + max(p, q), series, v_k, a, p, q) for i in range(series_len - max(p, q))]
            r2.append(var(our_values, np.mean(series)) / var(series, np.mean(series)))
            # ika
            ika.append(series_len * (np.log(s1)) + 2 * (p + q + 1))
            # ------------------------------------------------------------------------------
            a = rmnk_method(x_mat, y)
            erro = [e(i + max(p, q), series, v_k, a, p, q) for i in range(series_len - max(p, q))]
            s1 = np.sum(list(map(lambda x: x ** 2, erro)))
            ep_r.append(s1)
            # R2
            our_values = series[:max(p, q)] + \
                [new_values(i + max(p, q), series, v_k, a, p, q) for i in range(series_len - max(p, q))]
            r2_r.append(var(our_values, np.mean(series)) / var(series, np.mean(series)))
            # ika
            ika_r.append(series_len * (np.log(s1)) + 2 * (p + q + 1))
        except ValueError:
            continue
    
    pd.DataFrame(parameters_1, columns=[f'a{i}' for i in range(p + 1)] + ['I'] + [f'b{i}' for i in range(1, q + 1)]).\
        to_csv(f'MNK{p}_{q}_params.csv')
    pd.DataFrame(parameters_2, columns=[f'a{i}' for i in range(p + 1)] + ['I'] + [f'b{i}' for i in range(1, q + 1)]).\
        to_csv(f'RMNK{p}_{q}_params.csv')
    pd.DataFrame(ep).to_csv(f'MNK{p}_{q}_eps.csv')
    pd.DataFrame(r2).to_csv(f'MNK{p}_{q}_r2.csv')
    pd.DataFrame(ika).to_csv(f'MNK{p}_{q}_ika.csv')
    pd.DataFrame(ep_r).to_csv(f'RMNK{p}_{q}_eps.csv')
    pd.DataFrame(r2_r).to_csv(f'RMNK{p}_{q}_r2.csv')
    pd.DataFrame(ika_r).to_csv(f'RMNK{p}_{q}_ika.csv')

    file = open(f'Rez{p}_{q}.txt', 'w')
    printf = partial(print, file=file)
    x_mat, y = create_fisher_matrix(np.array(series), v_k, p, q)
    printf('MNK method . Params')
    # p a +1+ q b
    a = mnk_method(x_mat, y)
    printf(a)
    errors = [e(i + max(p, q), series, v_k, a, p, q) for i in range(series_len - max(p, q))]
    s = np.sum(list(map(lambda x: x ** 2, errors)))
    printf(f'Error (square): {s}')
    our_values = series[:max(p, q)] + \
        [new_values(i + max(p, q), series, v_k, a, p, q) for i in range(series_len - max(p, q))]
    printf(f' R^2: {var(our_values, np.mean(series)) / var(series, np.mean(series))}')
    printf(f'IKA : {series_len * (np.log(s)) + 2 * (p + q + 1)}')
    printf('RMNK method . Params')
    a = rmnk_method(x_mat, y)
    printf(a)
    errors = [e(i + max(p, q), series, v_k, a, p, q) for i in range(series_len - max(p, q))]
    s = np.sum(list(map(lambda x: x ** 2, errors)))
    printf(f'Error (square): {s}')
    our_values = series[:max(p, q)] + \
        [new_values(i + max(p, q), series, v_k, a, p, q) for i in range(series_len - max(p, q))]
    printf(f' R^2: {var(our_values, np.mean(series)) / var(series, np.mean(series))}')
    printf(f'IKA : {series_len * (np.log(s)) + 2 * (p + q + 1)}')
    file.close()


def e(k, y, v, a, p, q):
    return y[k] - new_values(k, y, v, a, p, q)


def new_values(k, y, v, a, p, q):
    v1 = np.array([1] + list(y[k - p:k])[::-1] + v[k - q:k + 1][::-1])
    return np.dot(v1, a)


def var(y_hat, y_mean):
    return np.sum(list(map(lambda x: (x - y_mean) ** 2, y_hat)))


def main() -> NoReturn:
    series_len = 1000
    alpha, betta = get_variables()
    v_k = gaussian_vector(series_len)
    white_noise = gaussian_vector(series_len, 0, 0.01)
    single_model_ = partial(single_model, series_len=series_len, alpha=alpha,
                            betta=betta, v_k=v_k, white_noise=white_noise)
    # for i in range(9):
    #     single_model_(p=i // 3 + 1, q=i % 3 + 1)
    with ThreadPoolExecutor(max_workers=9) as executor:
        fit_ = [executor.submit(single_model_, p=i // 3 + 1, q=i % 3 + 1) for i in range(9)]
        wait(fit_)


main()
