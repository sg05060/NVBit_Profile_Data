import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.ticker import PercentFormatter

def parse_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument('--filename', default = '/home/kkh/pytorch/data/')
    return parser.parse_args()

def two_graphs(data, y1_min, y1_max,  y2_min, y2_max):
    # draw 2 graphs
    fig, (ax1,ax2) = plt.subplots(2,1, sharex=True)
    fig.subplots_adjust(hspace=0.05)
    ys, xs, bars = ax1.hist(data, range=(-14,15), weights=np.ones(len(data)) / (len(data)), histtype='bar', bins=29,  align='mid',  rwidth=0.8)
    ys, xs, bars = ax2.hist(data, range=(-14,15), weights=np.ones(len(data)) / (len(data)), histtype='bar', bins=29,  align='mid',  rwidth=0.8)

    for i in range(0,len(ys)):
        plt.text(x=xs[i]+0.2, y=ys[i],s='{:0>3.1f}%'.format(ys[i]),fontsize=3,color='blue',)


    ax2.set_xlim(-14, 15)

    ax1.set_ylim(y1_min, y1_max)
    ax2.set_ylim(y2_min, y2_max)


    # delete boundary line
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)
    ax2.xaxis.tick_bottom()

    return ax1, ax2


def main(args):
    #print(args.filename)
    data= np.load(args.filename).reshape(-1)

    # 1 graph
    plt.figure(figsize=(10,6))
    #plt.xlabel("FP32 Weights \n",fontsize=30)
    #plt.xlabel("INT16 Weights \n", fontsize=30)
    plt.xlabel("SYMBOL_Weights \n", fontsize=30)
    #plt.ylabel("counts",fontsize=25)
    plt.ylabel("counts(%)", fontsize=30)
    #plt.xticks(range(index.min(),index.max()))
    plt.ticklabel_format(axis = 'both', useOffset=False, style='plain')

    # percentage(FP32, INT16)
    ys, xs, bars = plt.hist(data,  weights=np.ones(len(data)) / len(data), histtype='bar', bins=100,  align='mid',  rwidth=1)

    # counts (FP32, INT16)
    #ys, xs, bars = plt.hist(data, histtype='bar', bins=100,  align='mid',  rwidth=1)

    # symbol
    # ys, xs, bars = plt.hist(data, range=(-15,16), histtype='bar', bins=31,  align='left',  rwidth=0.8)
    # ys, xs, bars = plt.hist(data, range=(-7,8), weights=np.ones(len(data)) / (len(data)), histtype='bar', bins=15,  align='left',  rwidth=0.8)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    #for i in range(0,len(ys)):
    #    if (ys[i] >= 0.01):
    #        plt.text(x=xs[i], y=ys[i],s='{:0>3.2f}%'.format(ys[i]),fontsize=8,color='blue',)
    # plt.xticks([(xs[i]+xs[i+1])/2 for i in range(0,len(xs)-1)], ["{:.1f} ~ {:.1f}".format(xs[i],xs[i+1]) if i<=3 for i in range(0, len(xs)-1)])

    xs_max = 0
    xs_min = 0

    if xs_max < max(xs) :
        xs_max = max(xs)
    if xs_min > min(xs) :
        xs_min = min(xs)
    # fp
    #plt.text(x=xs_max,y=0,s='{:0>3.5f}'.format(xs_max),fontsize=7, color='blue',)
    #plt.text(x=xs_min,y=0,s='{:0>3.5f}'.format(xs_min),fontsize=7, color='blue',)

    # int
    plt.text(x=xs_min,y=0,s='min {:0>3.0f}'.format(xs_min),fontsize=10, color='blue',)
    plt.text(x=xs_max,y=0,s='max {:>3.0f}'.format(xs_max),fontsize=10, color='blue',)

    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    plt.xlim(x_min,x_max+0.5)
    plt.ylim(y_min,y_max)

    plt.savefig(args.filename+".png",dpi=500)

    #############################################################################
    # # 2 graph
    # # draw 2 graphs
    # fig, (ax1,ax2) = plt.subplots(2,1, sharex=True)
    # fig.subplots_adjust(hspace=0.05)
    # ys, xs, bars = ax1.hist(data, range=(-14,15), weights=np.ones(len(data)) / (len(data)), histtype='bar', bins=29,  align='mid',  rwidth=0.8)
    # ys, xs, bars = ax2.hist(data, range=(-14,15), weights=np.ones(len(data)) / (len(data)), histtype='bar', bins=29,  align='mid',  rwidth=0.8)

    # for i in range(0,len(ys)):
    #     plt.text(x=xs[i]+0.2, y=ys[i],s='{:0>3.1f}%'.format(ys[i]),fontsize=3,color='blue',)


    # ax2.set_xlim(-14, 15)

    # ax1.set_ylim(10000000, 14000000)
    # ax2.set_ylim(0, 100000)


    # # delete boundary line
    # ax1.spines['bottom'].set_visible(False)
    # ax2.spines['top'].set_visible(False)
    # ax1.xaxis.tick_top()
    # ax1.tick_params(labeltop=False)
    # ax2.xaxis.tick_bottom()

    # plt.xlabel("SYMBOL_INT16 DATA \n")
    # plt.ylabel("counts(%)")
    # ax1.ticklabel_format(axis = 'both', useOffset=False, style='plain')
    # ax2.ticklabel_format(axis = 'both', useOffset=False, style='plain')

    # # ys, xs, bars = plt.hist(data, range=(-14,15), weights=np.ones(len(data)) / (len(data)), histtype='bar', bins=29,  align='mid',  rwidth=0.8)
    # # ys, xs, bars = ax2.hist(data, range=(-14,15), weights=np.ones(len(data)) / (len(data)), histtype='bar', bins=29,  align='mid',  rwidth=0.8)


    # # for i in range(0,len(ys)):
    # #     plt.text(x=xs[i]+0.2, y=ys[i],s='{:0>3.1f}%'.format(ys[i]),fontsize=3,color='blue',)

    # x_min, x_max = plt.xlim()
    # y_min, y_max = plt.ylim()
    # plt.xlim(x_min,x_max+0.5)
    # plt.ylim(y_min,y_max)

    # plt.savefig(args.filename+".png",dpi=500)




if __name__ == '__main__':
    args = parse_argument()
    main(args)
