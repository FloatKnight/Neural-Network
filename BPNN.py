import numpy as np
class backPropagtionNN:
    def __init__(self,size,eta=0.1):
        '''初始化 backPropagtion 模型
           size[0] 输入神经元个数
           size[-1] 输出神经元个数
           len(size) -2 隐层数
        '''
        #初始化权值
        self.W = None
        #初始化阈值
        self.thre = None
        #神经网络的大小
        self.size = size
        #学习率
        self.eta = eta

    def fit(self,X_train,y_train):
        '''根据X_train,y_train训练 backPropagtion 训练模型'''
        assert X_train.shape[0] == y_train.shape[0], \
            ' the size of X_train must be equal the size of Y_train '

        #初始化权值
        self.W = [ np.random.randn(self.size[i]*self.size[i+1]).reshape(self.size[i+1],self.size[i])  for i in range(len(self.size)-1) ]
        #初始化阈值
        self.thre = [ np.random.rand(self.size[i+1]) for i in range(len(self.size)-1) ]

        while self.score(X_train,y_train)<0.95:

            for x,y in zip(X_train,y_train):

                #计算当前样本各层的预测值   y_hat[-1] 是最后的预测值
                y_hat = list()
                y_hat.append(x)
                for i in range(len(self.size) -1):
                    y_hat.append(self.sigmoid( self.W[i].dot( y_hat[-1]) - self.thre[i] ))

                y_true = np.zeros(y_hat[-1].shape)
                y_true[y] = 1

                #保存 g 用于计算 各层的 权值 和 阈值的梯度
                df_g  = list()
                df_g.append( y_hat[-1]*(1-y_hat[-1])*( y_true - y_hat[-1] ) )
                #逆序遍历  y_hat  BP
                for y_hat_single,w in zip(y_hat[-2:0:-1],self.W[:0:-1]):
                    df_g.append( (y_hat_single*(1 - y_hat_single))*(w.T.dot(df_g[-1])) )

                df_g.reverse()

                #修改 权值 和 阈值
                for i in range(len(self.W)):
                    self.W[i] += self.eta * np.vstack([df_g[i]]*len(y_hat[i])).T * np.vstack([y_hat[i]] * len(df_g[i]))
                    self.thre[i] += -self.eta * df_g[i]
        return self

    def predict(self,X_predict):
        '''给定待预测数据集X_predict ,返回表示X_predict的结果向量'''
        assert self.W is not None, \
            'must fit before predict'
        assert X_predict.shape[1] == self.W[0].shape[1], \
            'the size of X_predict must be equal the size of self.W[0]'
        res = X_predict
        for i in range(len(self.size)-1):
            res = self.sigmoid(self.W[i].dot(res.T).T - self.thre[i])

        return np.argmax( res,axis=1 )

    def score(self,X_test,y_test):
        '''根据测试数据集 x_test,y_test确定当前模型的精准度
        计算y_true ,y_predict 之间的 R Square
        '''
        y_predict = self.predict(X_test)
        assert len(y_test) == len(y_predict), \
            'the size of y_true must be equal the size of y_predict'

        return 1- np.sum((y_predict-y_test)**2)/len(y_test)/np.var(y_test)


    def sigmoid(self,x):
        ''' sigmoid 激活函数 '''
        return 1/(1+np.exp(-x))

    def __repr__(self):
        return 'backPropagtionNN()'