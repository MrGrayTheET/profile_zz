import numpy as np
from screener.volatility_indicators import tr, atr
from sc_loader import sierra_charts as sc

sc = sc()

daily_config = {
    'ATR': 5,
    'RVOL': 10,
    'STD': 252,
}

class screener:

    def __init__(self, data , atr_len=5, rvol_len =10, std_len=252, atr_percentile=60, tr_ratio=1.5, over=True):

        self.config = {
                'ATR': atr_len,
                'RVOL': rvol_len,
                'STD': std_len,
                'TR_ratio': tr_ratio,
                'Over':over
            }

        self.data = data.dropna()
        self.ohlc = ['Open', 'High', 'Low', 'Close', 'Volume']
        self.true_indices = {'ATR':[],
                             'TR':[],
                             'TR>ATR':[]}

        self.add_screener_values()

        self.config.update({'ATR_thresh': np.percentile(self.data['ATR'], atr_percentile)})

        return

    def add_screener_values(self):
        self.data['ATR'] = atr(self.data, self.ohlc[1], self.ohlc[2], self.ohlc[3],length=self.config['ATR'], normalized=True)
        self.data['TR'] = tr(self.data, self.ohlc[1], self.ohlc[2], self.ohlc[3], normalized=True)
        self.data['Range'] = ((self.data['High'] - self.data['Low'])/self.data['Close'])
        self.data['Std Range'] = self.data.Range/self.data.Range.std()


    def screen(self, output_format='dash'):
        if self.config['Over']:
            tr_results = self.data.loc[self.data['TR'] > (self.data['ATR'] * self.config['TR_ratio'])]
            atr_results = self.data.loc[self.data['ATR'] > self.config['ATR_thresh']]
        elif not self.config['Over']:
            tr_results = self.data.loc[self.data['TR'] < (self.data['ATR'] * self.config['TR_ratio'])]
            atr_results =  self.data.loc[self.data['ATR'] < self.config['ATR_thresh']]

        if output_format == 'dash':
            res = {'TR': tr_results.index.date,
               'ATR': atr_results.index.date}

        return res












