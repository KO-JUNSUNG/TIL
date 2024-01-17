import numpy as np
from scipy.stats import binom, poisson
from empiricaldist import Pmf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



# qs = 가능한 값, 인덱스
# ps = 가능한 값의 확률, pdf(pmf), 정규화되지(unnormalized) 않은 확률밀도함수 


def make_binomial(n:int,p:float)->Pmf:
    """이항 Pmf를 생성"""
    ks = np.arange(n+1)
    ps = binom.pmf(ks,n,p) # 확률변수 0~ n 까지의 이항확률값
    return Pmf(ps,ks)

def update_binomial(pmf:Pmf, data: list or tuple)->None:
    """이항분포를 사용한 pmf 갱신"""
    k,n = data
    xs = pmf.qs
    likelihood = binom.pmf(k,n,xs) # binom(k,n,p)
    pmf *= likelihood
    pmf.normalize()

def make_die(sides:int)->Pmf:
    outcomes = np.arange(1,sides+1)
    die = Pmf(1/sides,outcomes)
    return die

def add_dist_seq(seq):
    """seq = 여러 개의 dist를 가지는 리스트 구조"""
    total = seq[0]
    for other in seq[1:]:
        total = total.add_dist(other)
    return total


def plotting(target, x_label = 'Number', y_label = 'PMF')->None:
    # 그림의 가로 세로 비율 조절
    plt.figure(figsize=(10, 5))

    # Seaborn 스타일 설정
    sns.set(style="whitegrid")
    target.plot()

    # title, x label, y label 추가
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # 그래프 표시
    plt.show()

def make_mixture(pmf:Pmf, pmf_seq:list or tuple)->Pmf:
    """
    make mixture distribution
    각 주사위가 선택될 사전확률 pmf_dice
    각 주사위가 가능한 확률을 나타내는 Pmf_seq = likelihood
    """
    df = pd.DataFrame(pmf_seq).fillna(0).transpose()
    df *= np.array(pmf)
    total = df.sum(axis = 1)
    return Pmf(total)

def plt_default(x_label:str)->None:
    plt.xlabel(x_label)
    plt.ylabel("PMF")
    plt.legend()
    plt.show()


def make_poisson_pmf(lam: float, qs: list)->Pmf:
    """make_poisson_pmf 는 득점율 lam과 값의 배열 qs를 사용해서 포아송 PMF를 만든다. 결괏값으로는 Pmf 객체를 반환"""
    ps = poisson(lam).pmf(qs)
    pmf = Pmf(ps, qs)
    pmf.normalize()
    return pmf

from scipy.stats import gaussian_kde

def kde_from_sample(sample, qs):
    """make kde from sample data"""
    kde = gaussian_kde(sample)
    ps = kde(qs)
    pmf = Pmf(ps, qs)
    pmf.normalize()
    return pmf