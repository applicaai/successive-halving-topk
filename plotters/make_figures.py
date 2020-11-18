import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd
import seaborn as sns

BENCHMARK_FILENAME = '../benchmark_log_16003623822_cuda:0.csv'

PLOTTED_K = (2, 8, 32, 128, 512, 2048)

BASELINE_NAME = 'Goyal et al.'
OUR_NAME = 'Our'

LOG_LOCATORS = (.01, .1, 1)
LINEAR_LOCATORS = (0, .5, 1)


def plot(df, filename, field='time', yscale='log'):
    grid = sns.FacetGrid(df,
                         col='k',
                         hue='pooler_name',
                         palette=sns.color_palette('Set2'),
                         col_wrap=3,
                         height=1.6,
                         legend_out=True)

    grid.map(sns.lineplot, 'n', field, marker='o', dashes=False)

    plt.xscale('log', basex=2)
    if yscale == 'log':
        plt.yscale('log', basey=10)

    ax = plt.gca()
    ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=2, numticks=6))
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(LOG_LOCATORS if yscale == 'log' else LINEAR_LOCATORS))
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

    plt.legend(loc='lower left', ncol=2, bbox_to_anchor=(.28, .05),
               bbox_transform=grid.fig.transFigure)

    grid.set(xlabel='', ylabel='')
    grid.fig.subplots_adjust(bottom=0.3)

    plt.savefig(filename)


def read_data():
    df = pd.read_csv(BENCHMARK_FILENAME)
    df = df[df['k'].isin(PLOTTED_K)]
    df = df.replace('our_topk', OUR_NAME)
    df = df.replace('iter_topk', BASELINE_NAME)
    return df


df = read_data()
plot(df, 'time_performance.pdf', field='time')
plot(df, 'approximation_quality.pdf', field='cosine_row', yscale='linear')
