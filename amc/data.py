

class Factor:
    def __init__(self, name: str, spot: float, atm_vol: float, is_normal: bool = False):
        self.name = name
        self.spot = spot
        self.atm_vol = atm_vol
        self.is_normal = is_normal


class VolatilityFactor(Factor):
    def __init__(self, name: str, spot: float, atm_vol: float):
        super(VolatilityFactor, self).__init__(name=name, spot=spot, atm_vol=atm_vol, is_normal=False)

    @property
    def vol(self):
        return self.spot


class EquityFactor(Factor):
    def __init__(self, name: str, spot: float, q: float, atm_vol: float):
        """

        :rtype: object
        """
        # TODO: to be expanded to include vol surface
        super(EquityFactor, self).__init__(name=name, spot=spot, atm_vol=atm_vol, is_normal=False)
        self.q = q
