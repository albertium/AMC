
import pandas as pd
from amc.helper import calc_implied_vols_from_prices, plot_implied_vols


# data = pd.read_csv('data/spx_opt.csv', parse_dates=['date', 'exdate'])
# new = calc_implied_vols_from_prices(data)
# plot_implied_vols(new)
# print(new[new.t_exp < 0.006])

class Base:
    def __init__(self, tenor, abc):
        self.mc = False
        self.fd = False
        self.tenor = tenor


class MC:
    def __init__(self, *args, **kwargs):
        super(MC, self).__init__(*args, **kwargs)
        self.mc = True


class FD:
    def __init__(self, abc, *args, **kwargs):
        print(abc)
        super(FD, self).__init__(*args, **kwargs)
        self.fd = True


class Yes(MC, FD, Base):
    def __init__(self):
        super(Yes, self).__init__(tenor=10)


yes = Yes()
print(yes.fd)
print(yes.mc)
print(yes.tenor)