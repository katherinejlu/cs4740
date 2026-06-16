from p4 import main as rnn_main
from ffnn1fix import main as ffnn_main
import random
from data_loader import fetch_data

FLAG = 'RNN'


def main():
    if FLAG == 'RNN':
#        raise NotImplementedError
        hidden_dim = 100
        number_of_epochs = 10
        rnn_main(hidden_dim=hidden_dim, number_of_epochs=number_of_epochs)
    elif FLAG == 'FFNN':
        count1=0
        count2=0
        count3=0
        count12=0
        count23=0
        count31=0
        countall=0
        count0=0
        
        train_data, valid_data = fetch_data()
        random.shuffle(train_data)
        random.shuffle(valid_data)
        print('--------------32-------------')
        hidden_dim = 32
        number_of_epochs = 1
        out1=ffnn_main(train_data,valid_data,hidden_dim=hidden_dim, number_of_epochs=number_of_epochs)
        print('--------------64-------------')
        hidden_dim = 64
        number_of_epochs = 1
        out2=ffnn_main(train_data,valid_data,hidden_dim=hidden_dim, number_of_epochs=number_of_epochs)
        print('--------------128-------------')
        hidden_dim = 128
        number_of_epochs = 1
        out3=ffnn_main(train_data,valid_data,hidden_dim=hidden_dim, number_of_epochs=number_of_epochs)
        for i in range(len(out2)):
            if(out1[i]==0 and out2[i]==0 and out3[i]==0):
                count0+=1
                print('all wrong at ',i)
            if(out1[i]==1 and out2[i]==0 and out3[i]==0):
                count1+=1
                print('only 32 is right at ',i)
            if(out1[i]==0 and out2[i]==1 and out3[i]==0):
                count2+=1
                print('only 64 is right at ',i)
            if(out1[i]==0 and out2[i]==0 and out3[i]==1):
                count3+=1
                print('only 128 is right at ',i)
            if(out1[i]==1 and out2[i]==1 and out3[i]==0):
                count12+=1
                print('32 and 64 right at ',i)
            if(out1[i]==0 and out2[i]==1 and out3[i]==1):
                count23+=1
                print('64 and 128 right at ',i)
            if(out1[i]==1 and out2[i]==0 and out3[i]==1):
                count31+=1
                print('128 and 32 right at ',i)
            if(out1[i]==1 and out2[i]==1 and out3[i]==1):
                countall+=1
                print('all are right at ',i)
        print('count0: ',count0)
        print('count1: ',count1)
        print('count2: ',count2)
        print('count3: ',count3)
        print('count12: ',count12)
        print('count23: ',count23)
        print('count31: ',count31)
        print('countall: ',countall)


if __name__ == '__main__':
    main()