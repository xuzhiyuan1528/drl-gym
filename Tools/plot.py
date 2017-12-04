import matplotlib.pyplot as plt


def read_file(f_name):
    with open(f_name) as f_in:
        reward = []
        for row in f_in:
            if 'Sum_Reward' in row:
                reward.append(float(row.split(':')[-1]))

    plt.plot(reward)
    plt.show()


read_file('../11-29-pong.log')
