hin = 8
win = 8
hout = 8
wout =8
kernel = 5

f=0
for stride in range(1,101): #1~100
    for padding in range(1,101):
        if hout == ((hin + 2 * padding - 1*(kernel - 1) - 1) / stride + 1):
            if wout == ((win + 2 * padding - (kernel - 1) - 1) / stride + 1):
                print("stride=", stride, " padding=", padding)
                f=1
                break
    if f==1:break