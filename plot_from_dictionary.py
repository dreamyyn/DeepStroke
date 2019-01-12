import matplotlib.pyplot as plt
import csv

results = {'thres': [0.1, 0.2, 0.3, 0.4, 0.6, 0.7], 'auc': [0.8802070912637572, 0.8795920646524964, 0.8791452336260696, 0.8788930063319491, 0.8784131379442548, 0.8782998837903675], 'precision': [0.41484308745785864, 0.42908437237782765, 0.43851778546547093, 0.44761437986434754, 0.46270718866768334, 0.4675188485681173], 'recall': [0.642545210349806, 0.6290157886250328, 0.6196245052022227, 0.6110961891044838, 0.5953121781608802, 0.586682841320777], 'specificity': [0.9472502767866167, 0.9506536863287216, 0.953051184583586, 0.955068286654939, 0.9587603521085791, 0.9607237672029267], 'dice': [0.4382214001663466, 0.44220298127745994, 0.4448444306683406, 0.4466711179837892, 0.44959157369588826, 0.4507638251746023], 'volume_difference': [2154.7156862745096, 1662.7745098039215, 1338.8333333333333, 1071.549019607843, 598.3235294117648, 353.7647058823529], 'volume_predicted': [9230.196078431372, 8738.254901960785, 8414.313725490196, 8147.029411764706, 7673.803921568628, 7429.245098039216], 'f1_score': [0.4381829512195638, 0.44216417400031477, 0.4448053870947631, 0.4466319058689375, 0.4495520567890491, 0.4507241748464106]}


# x = results['thres']
# y1 = results['auc']
# y2 = results['dice']
# y3 = results['f1_score']
# y4 = [i * 0.008 for i in results['volume_difference']]
# y5 = results['recall']
# y6 = results['precision']
# print(y4)
#
# fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2,figsize= (12,15),sharex=False)
#
#
# ax1.set_xlabel('threshold of model')
# ax1.set_ylabel('auc')
# ax1.plot(x,y1,zorder=1, lw =1)
# ax1.scatter(x,y1)
# # ax2 = ax1.twinx()
# ax2.set_xlabel('threshold of model')
# ax2.set_ylabel('Volume difference (ml)')
# ax2.plot(x,y4, lw=1)
# ax2.scatter(x,y4)
# ax3.set_xlabel('threshold of model')
# ax3.set_ylabel('dice score')
# ax3.plot(x,y2,lw =1)
# ax3.scatter(x,y2)
# ax4.set_xlabel('threshold of model')
# ax4.set_ylabel('f1 score')
# ax4.plot(x,y3,lw =1)
# ax4.scatter(x,y3)
# ax5.set_xlabel('threshold of model')
# ax5.set_ylabel('recall')
# ax5.plot(x,y5,lw =1)
# ax5.scatter(x,y5)
# ax6.set_xlabel('threshold of model')
# ax6.set_ylabel('precision')
# ax6.plot(x,y6,lw =1)
# ax6.scatter(x,y6)
# plt.plot(x,y1,zorder=1,lw=1)
# plt.scatter(x,y1,zorder=2)
# plt.plot(x,y2,zorder=1,lw=1)
# plt.scatter(x,y2,zorder=2)
# plt.plot(x,y3,zorder=1,lw=1)
# plt.scatter(x,y3,zorder=2)
# plt.legend(results.keys())
# fig.tight_layout()
# fig.savefig('lineart.png')

with open('metrics.txt', 'w') as file:
  for i in range(len(results['thres'])):
    row = []
    for key in results.keys():
      # print(key,results[key][i])
      row += [results[key][i]]
      str1 = ','.join(str(n) for n in row) + '\n'
    print(str1)
    file.write(str1)