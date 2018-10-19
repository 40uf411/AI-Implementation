import numpy as np

x_data = [1.0, 2.0, 3.0, 5.0]
y_data = [2.0, 4.0, 6.0, 10.0]

w = 1 # random value

#forward pass
def forward(x):
    return  x * w

#loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

#compute gradient
def gradiant(x, y):
    return 2 * x * ( x * w - y) # d loss/ dw =

w_list = []
mse_list = []

# before training
print("predict (before training)", 4 , forward(4))

#training loop
for epoch in range(40):
    for x_val, y_val in zip(x_data, y_data):
        grad = gradiant(x_val,y_val)
        w = w - 0.01 * grad
        print("\tgrad: ", grad , "\tx_val: ",x_val,"\ty_val: ",y_val)
        l = loss(x_val,y_val)

    print("progress", epoch, "w= ",w,"loss= ",l)

print("predict (after training)", "4 hours",forward(4))


#
# old code
#
#for w in np.arange(0.0, 4.1, 0.1):
#    print("w=", w)
#    l_sum = 0
#    for x_val, y_val in zip(x_data, y_data):
#        y_pred_val = forward(x_val)
#        l = loss(x_val, y_val)
#        l_sum += l
#        print("\t", x_val, y_val, y_pred_val, l)
#
#    print("mse=", l_sum / 3 )
#    print("\n")
