import datetime
import os
class Logger:
    def __init__(self, exp_name):
        if os.path.exists('./logs/Ablation/'):
            self.file = open('./logs/Ablation/{}.log'.format(exp_name), 'w')
        else:
            os.makedirs('./logs/Ablation/')
            self.file = open('./logs/Ablation/{}.log'.format(exp_name), 'w')

    def log(self, content):
        self.file.write(content + '\n')
        self.file.flush()



