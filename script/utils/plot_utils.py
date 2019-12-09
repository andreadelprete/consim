# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 09:16:56 2015

@author: adelpret
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_FONT_SIZE = 35;
DEFAULT_AXIS_FONT_SIZE = DEFAULT_FONT_SIZE;
DEFAULT_LINE_WIDTH = 4; #13;
DEFAULT_MARKER_SIZE = 4;
DEFAULT_FONT_FAMILY = 'sans-serif'
DEFAULT_FONT_SERIF = ['Times New Roman', 'Times','Bitstream Vera Serif', 'DejaVu Serif', 'New Century Schoolbook', 'Century Schoolbook L', 'Utopia', 'ITC Bookman', 'Bookman', 'Nimbus Roman No9 L', 'Palatino', 'Charter', 'serif'];
DEFAULT_FIGURE_FACE_COLOR = 'white'    # figure facecolor; 0.75 is scalar gray
DEFAULT_LEGEND_FONT_SIZE = DEFAULT_FONT_SIZE;
DEFAULT_AXES_LABEL_SIZE = DEFAULT_FONT_SIZE;  # fontsize of the x any y labels
DEFAULT_TEXT_USE_TEX = False;
LINE_ALPHA = 0.9;
SAVE_FIGURES = False;
FILE_EXTENSIONS = ['pdf', 'png']; #,'eps'];
FIGURES_DPI = 150;
SHOW_FIGURES = False;
FIGURE_PATH = './';

#axes.hold           : True    # whether to clear the axes by default on
#axes.linewidth      : 1.0     # edge linewidth
#axes.titlesize      : large   # fontsize of the axes title
#axes.color_cycle    : b, g, r, c, m, y, k  # color cycle for plot lines
#xtick.labelsize      : medium # fontsize of the tick labels
#figure.dpi       : 80      # figure dots per inch
#image.cmap   : jet               # gray | jet etc...
#savefig.dpi         : 100      # figure dots per inch
#savefig.facecolor   : white    # figure facecolor when saving
#savefig.edgecolor   : white    # figure edgecolor when saving
#savefig.format      : png      # png, ps, pdf, svg
#savefig.jpeg_quality: 95       # when a jpeg is saved, the default quality parameter.
#savefig.directory   : ~        # default directory in savefig dialog box,
                                # leave empty to always use current working directory
mpl.rcdefaults()
mpl.rcParams['lines.linewidth']     = DEFAULT_LINE_WIDTH;
mpl.rcParams['lines.markersize']    = DEFAULT_MARKER_SIZE;
mpl.rcParams['patch.linewidth']     = 1;
mpl.rcParams['font.family']         = DEFAULT_FONT_FAMILY;
mpl.rcParams['font.size']           = DEFAULT_FONT_SIZE;
mpl.rcParams['font.serif']          = DEFAULT_FONT_SERIF;
mpl.rcParams['text.usetex']         = DEFAULT_TEXT_USE_TEX;
mpl.rcParams['axes.labelsize']      = DEFAULT_AXES_LABEL_SIZE;
mpl.rcParams['axes.grid']           = True
mpl.rcParams['legend.fontsize']     = DEFAULT_LEGEND_FONT_SIZE;
mpl.rcParams['legend.framealpha']   = 0.5                           # opacity of of legend frame
mpl.rcParams['figure.facecolor']    = DEFAULT_FIGURE_FACE_COLOR;
mpl.rcParams['figure.figsize']      = 23, 18 #12, 9 #


def create_empty_figure(nRows=1, nCols=1, spinesPos=None,sharex=True):
    f, ax = plt.subplots(nRows,nCols,sharex=sharex);
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(50,50,1080,720);

    if(spinesPos!=None):
        if(nRows*nCols>1):
            for axis in ax.reshape(nRows*nCols):
                movePlotSpines(axis, spinesPos);
        else:
            movePlotSpines(ax, spinesPos);
    return (f, ax);

    
def movePlotSpines(ax, spinesPos):
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',spinesPos[0]))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',spinesPos[1]))

    
def setAxisFontSize(ax, size):
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(size)
        label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.65))

def plot_from_logger(logger, dt, streams, labels=None, titles=None, linestyles=None, ncols=1, xlabel='Time [s]', ylabel=None, yscale=None):
    nsubplots = len(streams)
    nrows = int(nsubplots/ncols)
    f, ax = plt.subplots(nrows, ncols, sharex=True);
    if(nsubplots>1):
        ax = ax.reshape(nsubplots)
    else:
        ax = [ax]
    
    if labels is None:
        labels = streams
    if linestyles is None:
        linestyles = [[None for j in i] for i in streams] 
        
    i = 0
    for (sp_streams, sp_labels, sp_linestyles) in zip(streams, labels, linestyles):
        if(sp_labels is None):
            sp_labels = [None,]*len(sp_streams)
        for (stream, label, ls) in zip(sp_streams, sp_labels, sp_linestyles):
            try:
                N = len(logger.get_streams(stream))
                time = np.arange(0.0, dt*N, dt)
                # sometimes the vector time has more elements than it should
                if(len(time)>N):
                    time = time[:N]
                if( ls is None):
                    ax[i].plot(time, logger.get_streams(stream), label=label)
                else:
                    ax[i].plot(time, logger.get_streams(stream), ls, label=label)
            except KeyError:
                print "[plot_from_logger] Could not find field %s"%(stream)
        ax[i].legend(loc='best')

        if titles is not None:
            if isinstance(titles, basestring):
                ax[0].set_title(titles)
            else:
                ax[i].set_title(titles[i])
                
        if ylabel is not None:
            if isinstance(ylabel, basestring):
                ax[i].set_ylabel(ylabel)
            else:
                ax[i].set_ylabel(ylabel[i])
        
        if yscale is not None:
            if isinstance(yscale, basestring):
                ax[i].set_yscale(yscale)
            else:
                ax[i].set_yscale(yscale[i])
        i += 1
    if xlabel is not None:
        ax[-1].set_xlabel(xlabel)
    return ax
    
def grayify_cmap(cmap):
    """Return a grayscale version of the colormap"""
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))
    
    # convert RGBA to perceived greyscale luminance
    # cf. http://alienryderflex.com/hsp.html
    RGB_weight = [0.299, 0.587, 0.114]
    luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
    colors[:, :3] = luminance[:, np.newaxis]
    
    return cmap.from_list(cmap.name + "_grayscale", colors, cmap.N)

    
def saveFigure(title):
    if(SAVE_FIGURES):
        for ext in FILE_EXTENSIONS:
            plt.gcf().savefig(FIGURE_PATH+title.replace(' ', '_')+'.'+ext, format=ext, dpi=FIGURES_DPI, bbox_inches='tight');

def plot_gain_stability(Kp_grid,Kd_grid,stab_grid):
    plt.contourf(Kp_grid,Kd_grid,stab_grid,1)
    plt.plot(Kp_grid.flatten(), Kd_grid.flatten(),'+')
    Kps = np.linspace(Kp_grid.min(), Kp_grid.max(),1000)
    Kds = 2.*np.sqrt(Kps)
    
    plt.xlabel('$K_p$')
    plt.ylabel('$K_d$')
    plt.plot(Kps,Kds)
    return plt


def main():
    num = 1527601104
    inDir = "./data/{}/".format(num)
    data = np.load(inDir + "stab.npz")
    stab_grid = data['stab_grid']
    Kp_grid   = data['Kp_grid']
    Kd_grid   = data['Kd_grid']
    plot_gain_stability(Kp_grid,Kd_grid,stab_grid)
    plt.savefig(inDir + "stab.png")
    plt.show()
    return 0

if __name__ == '__main__':
	main()
