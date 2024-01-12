from scipy.stats import binom
from empiricaldist import Pmf

def make_binomial(n,p):
    """이항 Pmf를 생성"""
    ks =np.arange(n+1)
    ps = binom.pmf(ks,n,p) # 확률변수 0~ n 까지의 이항확률값
    return Pmf(ps,ks)