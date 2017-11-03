class History:
    def __init__(self, lr, drop):
        self.losses = []
        self.acc = []
        self.recall = []
        self.prec = []
        self.lr = lr
        self.drop = drop

    def update(self, loss, acc, recall, precision):
        self.losses += [loss]
        self.acc += [acc]
        self.recall += [recall]
        self.prec += [precision]

    def dump(self, file):
        with open(file, 'w') as f:
            f.write(str(self.lr) + '\n')
            f.write(str(self.drop) + '\n')
            f.write(str(self.losses)[1: -1] + '\n')   
            f.write(str(self.acc)[1: -1] + '\n')
            f.write(str(self.recall)[1: -1] + '\n')
            f.write(str(self.prec)[1: -1] + '\n')
