###############################################################################
#                                                                             #
#    This program is free software: you can redistribute it and/or modify     #
#    it under the terms of the GNU General Public License as published by     #
#    the Free Software Foundation, either version 3 of the License, or        #
#    (at your option) any later version.                                      #
#                                                                             #
#    This program is distributed in the hope that it will be useful,          #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of           #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            #
#    GNU General Public License for more details.                             #
#                                                                             #
#    You should have received a copy of the GNU General Public License        #
#    along with this program. If not, see <http://www.gnu.org/licenses/>.     #
#                                                                             #
###############################################################################

from sciviso import Emapplot
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from wordcloud import WordCloud


def plot_cluster_ORA(filename, gene_ratio='GeneRatio', count_column='Count', padj='p.adjust', overlap_column='geneID',
                     id_column='ID', label_column='Description', gene_ratio_min=0.05, padj_max=0.05, title='',
                     label_font_size=9, figsize=(3, 3), axis_font_size=6, min_count=20, max_count=200, min_overlap=4,
                     save_fig=True):
    """

    Parameters
    ----------
    filename
    gene_ratio
    count_column
    padj
    overlap_column
    id_column
    label_column
    gene_ratio_min
    padj_max
    title
    label_font_size
    figsize
    axis_font_size
    min_count
    max_count
    min_overlap
    save_fig

    Returns
    -------

    """
    df = pd.read_csv(f'{filename}')
    # Convert gene ratio to a number
    gr = df[gene_ratio].values
    gene_ratios = []
    for g in gr:
        g = g.split('/')
        g0 = float(g[0])
        g1 = float(g[1])
        gene_ratios.append(g0 / g1)
    df[gene_ratio] = gene_ratios
    df = df[df[gene_ratio] > gene_ratio_min]
    df = df[df[padj] < padj_max]
    if len(df) > 1:
        eplot = Emapplot(df, size_column=count_column, color_column=padj, id_column=id_column,
                         label_column=label_column, overlap_column=overlap_column, overlap_sep='/', title=title,
                         config={'figsize': figsize, 'label_font_size': label_font_size,
                                 'axis_font_size': axis_font_size})
        eplot.build_graph(min_overlap=min_overlap)
        plt.title(title, fontsize=18, fontweight='bold')
        plt.gca().set_clip_on = False

        if save_fig:
            plt.savefig(f'{filename.replace(".csv", "")}_Network.svg', bbox_inches='tight', transparent=True)
        plt.show()

        x, y = np.ogrid[:300, :300]
        plt.rcParams['svg.fonttype'] = 'none'  # Ensure text is saved as text
        plt.rcParams['figure.figsize'] = figsize
        mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
        mask = 255 * mask.astype(int)
        wordfeqs = defaultdict(int)
        for g in df[overlap_column].values:
            for w in g.split('/'):
                w = w.replace(' ', '.')
                wordfeqs[w] += 1
        total_words = len(wordfeqs)
        for w in wordfeqs:
            wordfeqs[w] = wordfeqs[w] / total_words
        # Compute the frequency of each word (since there are duplicates sometimes...)
        wordcloud = WordCloud(background_color="white", mask=mask, repeat=False).generate_from_frequencies(wordfeqs)
        wordcloud_svg = wordcloud.to_svg(embed_font=True)
        if save_fig:
            f = open(f'{filename.replace(".csv", "")}_WordCloud.svg', "w+")
            f.write(wordcloud_svg)
            f.close()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()