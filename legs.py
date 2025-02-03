import pyximport;

pyximport.install()
from zigzag import zigzag as zz
import pandas as pd
import numpy as np
import datetime as dt
from matplotlib import pyplot as plt
from .sc_loader.sc_loader import sierra_charts as sc
from .screener.indicators import tr
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from plotly import graph_objects as go
import plotly.express as px
import plotly.io as pio

sch = sc('/content/profile_zz/profile_zz/sc_loader/data_config.toml')


class trend_zz:

    def __init__(self, df, zz_1=0.007, zz_2=0.012, zz_3=0.02, tick_size=0.1, vp_data=None):
        self.settings = {
            'minor': {
                'volatility': (zz_1, -zz_1),
                'markersize': 9,
                'linewidth': 2,
                'style': 'k-'
            },
            'intermediate': {
                'volatility': (zz_2, -zz_2),
                'markersize': 15,
                'linewidth': 4,
                'style': 'y--'
            },

            'primary': {
                'volatility': (zz_3, -zz_3),
                'markersize': 30,
                'linewidth': 5,
                'style': 'c-'
            },
            'ticks': tick_size,
        }

        self.confirmation_times = {}

        self.data = df
        self.data['minor'] = zz.peak_valley_pivots(df.Close.values, zz_1, -zz_1)
        self.data['intermediate'] = zz.peak_valley_pivots(df.Close.values, zz_2, -zz_2)
        self.data['primary'] = zz.peak_valley_pivots(df.Close.values, zz_3, -zz_3)
        self.vol_profiles = None

        return

    def leg_modes(self, leg_col='intermediate'):
        X = self.data.Close
        modes = zz.pivots_to_modes(self.data[leg_col].values)
        return pd.Series(X).pct_change().groupby(modes).describe()


    def indices(self, leg_col='intermediate'):
        indices = self.data[self.data[leg_col] != 0].index
        return indices

    def s_his(self, leg_col='intermediate'):
        return self.data[self.data[leg_col] == 1]

    def s_los(self, leg_col='intermediate'):
        return self.data[self.data[leg_col] == -1]

    def leg_returns(self, leg_col='intermediate'):
        rets = zz.compute_segment_returns(self.data.Close.values, self.data[leg_col].values)
        rets_index = self.data.loc[self.data[leg_col] != 0][1:].index
        rets_series = pd.Series(rets, rets_index)

        return rets_series

    def leg_stats(self, leg_col='intermediate', sample_tf='30min', BidAskVol=True):
        swing_index = self.data[self.data[leg_col] != 0].index
        leg_df = pd.DataFrame(index=swing_index,
                              columns=['leg_returns', 'leg_t', 'leg_volume', 'leg_volatility', 'leg_vol_2'])
        leg_df['Datetime'] = leg_df.index
        leg_df.loc[swing_index[1:], 'leg_returns'] = self.leg_returns(leg_col).values
        leg_df.loc[swing_index, 'leg_t'] = leg_df['Datetime'].loc[swing_index] - leg_df['Datetime'].loc[
            swing_index].shift(1)
        for i in range(len(swing_index)):
            if i + 1 < len(swing_index):
                ds = self.data[swing_index[i]:swing_index[i + 1]]
                leg_df.loc[swing_index[i + 1], 'leg_volume'] = ds.Volume.sum()
                if BidAskVol:
                    leg_df.loc[swing_index[i + 1], 'leg_sell_vol'] = ds.BidVolume.sum()
                    leg_df.loc[swing_index[i + 1], 'leg_buy_vol'] = ds.AskVolume.sum()
                    leg_df.loc[swing_index[i + 1], 'net_delta'] = ds.AskVolume.sum() - ds.BidVolume.sum()

                leg_df.loc[swing_index[i + 1], 'leg_density'] = self.leg_vbp(ds.Volume.sum(), ds.High.max(),
                                                                             ds.Low.min(),
                                                                             tick_size=self.settings['ticks'])
                ds = ds.resample(sample_tf).apply(sch.resample_logic)
                ds['TR'] = tr(ds, 'High', 'Low', 'Close', normalized=True)
                leg_df.loc[swing_index[i + 1], 'leg_volatility'] = ds['TR'].mean()
                leg_df.loc[swing_index[i + 1], 'leg_vol_2'] = ds['TR'].std()

            else:
                pass

        leg_df = leg_df.ffill().drop_duplicates()

        return leg_df


    def leg_vbp(self, leg_volume, high, low, tick_size=0.1):

        leg_range = (high - low) / self.settings['ticks']

        leg_vbp = leg_volume / leg_range

        return leg_vbp

    def sub_legs(self, leg_col='intermediate', sub_leg_col='minor'):
        main_legs = self.indices(leg_col)
        cols = ['mean', 'count', 'std', '25%', '50%', '75%', 'min', 'max']
        sub_leg_df = pd.DataFrame(index=pd.MultiIndex.from_product([['HL', 'LH'], main_legs]), columns=cols)

        for i in range(len(main_legs)):
            if i + 1 < len(main_legs):

                ds = self.data.loc[main_legs[i]:main_legs[i + 1]]
                sub_legs = ds[ds[sub_leg_col] != 0]
                modes = zz.pivots_to_modes(ds[sub_leg_col].values)

                if ds.Close[-1] > ds.Close[0]:
                    sub_leg_df.loc[('LH', main_legs[i + 1]), cols] = \
                    pd.Series(ds.Close).pct_change().groupby(modes).describe().loc[1, cols]
                else:
                    sub_leg_df.loc[('HL', main_legs[i + 1]), cols] = \
                    pd.Series(ds.Close).pct_change().groupby(modes).describe().loc[-1, cols]

        sub_leg_df = sub_leg_df.ffill()

        return sub_leg_df

    def insert_swings(self, leg_col='intermediate', data=[], return_data='locs'):
        s_his = self.data[self.data[leg_col] == 1].index
        s_los = self.data[self.data[leg_col] == -1].index
        swing_df = pd.DataFrame(index=self.data.index, columns=['s_hi', 's_lo'])

        s_hi_print_thresh = self.settings[leg_col][1]
        s_lo_print_thresh = self.settings[leg_col][0]
        low_confirmation = []
        high_confirmation = []

        for i in range(len(s_los)):
            if (i + 1 < len(s_los)) and (i + 1 < len(s_his)):
                lh_df = self.data.loc[s_los[i]:s_his[i + 1]]
                hl_df = self.data.loc[s_his[i]:s_los[i]]
                filtered_lhs = lh_df[
                    lh_df['Close'] > (lh_df['Close'].loc[s_los[i]] + lh_df['Close'] * self.settings[leg_col][0])].index
                filtered_hls = hl_df[
                    hl_df['Close'] < (hl_df['Close'].loc[s_his[i]] + hl_df['Close'] * self.settings[leg_col][1])].index

                if len(filtered_lhs) > 0:
                    s_lo_loc = filtered_lhs[0]
                    low_confirmation.append(s_lo_loc)
                    swing_df.loc[s_lo_loc, 's_lo'] = self.data['Low'].loc[s_los[i]]

                if len(filtered_hls) > 0:
                    s_hi_loc = filtered_hls[0]
                    high_confirmation.append(s_hi_loc)
                    swing_df.loc[s_hi_loc, 's_hi'] = self.data['High'].loc[s_his[i]]

        if return_data == 'locs':
            return {'s_lo': low_confirmation,
                    's_hi': high_confirmation}
        else:
            swing_df = swing_df.ffill()
            swing_df = pd.concat([self.data, swing_df], axis=1)

            return swing_df

    def plot_swings(self, start_date=None, end_date=dt.datetime.today(), pivot_cols=[], method='pyplot', ax=None):

        if start_date is not None:
            X = self.data.Close.loc[start_date:end_date]
        else:
            X = self.data.Close

        if ax is None:
            ax = plt.subplots(1, 1)[1]

        for pivots in pivot_cols:
            high_marks = np.repeat(self.settings[pivots]['markersize'], len(self.data[self.data[pivots] == 1]))
            low_marks = np.repeat(self.settings[pivots]['markersize'], len(self.data[self.data[pivots] == -1]))

            ax.set_xlim(0, len(X))
            ax.set_ylim(X.min() * 0.99, X.max() * 1.01)
            ax.plot(np.arange(len(X)), X, 'k:', alpha=0.5)
            ax.plot(np.arange(len(X))[self.data[pivots] != 0], X[self.data[pivots] != 0],
                    self.settings[pivots]['style'], linewidth=self.settings[pivots]['linewidth'])
            ax.scatter(np.arange(len(X))[self.data[pivots] == 1], X[self.data[pivots] == 1], color='g', s=high_marks)
            ax.scatter(np.arange(len(X))[self.data[pivots] == -1], X[self.data[pivots] == -1], color='r', s=low_marks)

        return


class vol_prof:

    def __init__(self, df, BidAskVol=True, kde_factor=0.05, num_samples=250, min_prom=0.4, prominence=True,
                 densities=True, tick_size=0.1, max_width_ticks=50, prominence_line=False, base_line=True):
        self.kx, self.ky, self.pkx, self.pky = [None] *4
        self.data = df
        self.vol_col = 'Volume'
        self.profile_settings = dict(kde_factor=kde_factor, num_samples=num_samples, min_prom=min_prom,
                                     prominence=prominence, densities=densities, tick_size=tick_size,
                                     max_width_ticks=max_width_ticks)
        self.plot_settings = dict(prominence_line=prominence_line, base_line=base_line)
        return

    def __getitem__(self, item):
        self.ds = self.data.loc[item]
        self.raw_vp = self.ds[self.vol_col].groupby(self.ds.Close)
        prom_df = self.get_levels(self.data.loc[item], *[k for i, k in self.profile_settings.items()])
        self.dist_fig = self.plot_prof(self.ds.Close, self.ds.Volume, self.kx, self.ky)
        if type(prom_df) == pd.DataFrame:
            if self.plot_settings['prominence_line']:
                line_x = self.pkx
                line_y0 = self.pky
                line_y1 = line_y0 - prom_df['prominences']

                for x, y0, y1, in zip(line_x, line_y0, line_y1):
                    self.dist_fig.add_shape(type='line', xref='x', yref='y',
                                         x0=x, y0=y0, x1=x, y1=y1, line=dict(color='red', width=2))

            if self.plot_settings['base_line']:
                for i, row in prom_df.iterrows():
                    self.dist_fig.add_shape(type='line', xref='x', yref='y',
                                        x0=row['x0'],y0=row['y'], x1=row['x1'], y1=row['y'],
                                        line= dict(color='red', width=2))



        return prom_df

    def get_levels(self, data, kde_factor=0.08, num_samples=250, min_prom=0.2,
                   prominence=True, densities=True, tick_size=0.025, max_width_ticks=50, return_top_sorted=False, top_n=5):

        volume = data['Volume']
        close = data['Close']
        kde = gaussian_kde(close, weights=volume, bw_method=kde_factor)
        xr = np.linspace(close.min(), close.max(), num_samples)  # Building x-axis for kde
        ticks_per_sample = (xr.max() - xr.min()) / num_samples
        kdy = kde(xr)
        min_prom = kdy.max() * min_prom
        width_range = (1, max_width_ticks * tick_size / ticks_per_sample)
        peaks, peak_props = find_peaks(kdy, min_prom, width=width_range)

        self.kx = xr
        self.ky = kdy

        if prominence:
            left_ips = peak_props['left_ips']
            right_ips = peak_props['right_ips']
            width_x0 = xr.min() + (left_ips * ticks_per_sample)
            width_x1 = xr.min() + (right_ips * ticks_per_sample)
            width_y = peak_props['width_heights']
            self.pkx = xr[peaks]
            self.pky = kdy[peaks]

            prominence_df = pd.DataFrame(dict(zip(['x0', 'x1', 'y'], [width_x0, width_x1, width_y])))
            prominence_df['prominences'] = peak_props['prominences']
            if densities:
                left_base = peak_props['left_bases']
                right_base = peak_props['right_bases']
                int_from = xr.min() + (left_base * ticks_per_sample)
                int_to = xr.min() + (right_base * ticks_per_sample)

                prominence_df['density'] = [kde.integrate_box_1d(x0, x1) for x0, x1 in zip(int_from, int_to)]

            if return_top_sorted:
                return prominence_df.iloc[np.argsort(prominence_df['density'])[::-1][:top_n]]

            else:
                return prominence_df

        else:
            return {'close': close, 'volume': volume, 'kx': self.kx, 'ky': self.ky}

    def plot_prof(self, close, volume, kx, ky, ):
        fig = go.Figure()
        fig.add_trace(go.Histogram(name='vol profile', x=close, y=volume,
                                   nbinsx=150, histfunc='sum', histnorm='probability density'))
        fig.add_trace(go.Scatter(name='KDE', x=kx, y=ky, mode='lines', marker_color='red'))
        if len(self.pky) > 0:
            fig.add_trace(go.Scatter(name='Peaks', x=self.pkx, y=self.pky, mode='markers', marker=dict(size=10)))

        return fig

    def plot_candlesticks(self):
        for i in self.dist_fig.data:
            i.xaxis, i.yaxis = 'x', 'y'

        plot_datas = self.ds.resample('1h').apply(sch.resample_logic)


        fig = go.Figure(data=[go.Candlestick(
            x=[x for x in range(len(plot_datas))],
            open=plot_datas['Open'].values,
            high = plot_datas['High'].values,
            low=plot_datas['Low'].values,
            close=plot_datas['Close'].values,
            xaxis='x2',
            yaxis='y2',
            visible=True,
            showlegend=False),*self.dist_fig.data],
            layout=go.Layout(dict(
                title_text='Candles',
                xaxis=go.layout.XAxis(side='top',
                           range = [0,300],
                           rangeslider=go.layout.xaxis.Rangeslider(visible=False),
                           showticklabels=False),

                yaxis=go.layout.YAxis(side='left',
                           range=[self.ds.Low.min(), self.ds.High.max()],
                           showticklabels=False),

                xaxis2=go.layout.XAxis(side='bottom',
                                       title='Date',
                                       rangeslider=go.layout.xaxis.Rangeslider(visible=False),
                                       overlaying="x"),

                yaxis2 = go.layout.YAxis(side="right",
                                         title='Price',
                                         range=[self.ds.Low.min(), self.ds.High.max()],
                                         overlaying="y"))
            )
        )
        return fig

