from tensorflow.python.keras import layers, models


class IncepLayer(layers.Layer):
    def __init__(self,filters=32):
        super(IncepLayer, self).__init__()
        self.filters = filters
        self.c1 = layers.Conv2D(filters=self.filters,kernel_size=1,padding='same')
        self.c2 = layers.Conv2D(filters=self.filters,kernel_size=1,padding='same')
        self.c22 = layers.Conv2D(filters=self.filters,kernel_size=5,padding='same')
        self.c3 = layers.Conv2D(filters=self.filters,kernel_size=1,padding='same')
        self.c33 = layers.Conv2D(filters=self.filters,kernel_size=7,padding='same')

    def build(self,input_shape):
        super(IncepLayer,self).build(input_shape)

    def call(self, inputs):
        t1 = self.c1(inputs)
        t2 = self.c2(inputs)
        t2 = self.c22(t2)
        t3 = self.c3(inputs)
        t3 = self.c33(t3)
        inp_kernels = inputs.shape[-1].value
        concat = layers.concatenate([t1,t2,t3])
        print(concat.shape)
        return concat

inp = layers.Input((28,28,3))
x = IncepLayer()(inp)

m = models.Model(inp,x)
m.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
print(m.summary())