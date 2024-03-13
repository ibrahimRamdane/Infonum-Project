from Processing import data_yf, tab_fin, tab_fin_several
from Processing import *
from Metrics import *
from Visualization import *

# stocks1 =['AAPL','MSFT','TSM','ORCL','TXN','IBM','QCOM','MU','NTDOY','JD','HPQ']
# stocks2='AAPL'

if __name__ == "__main__":
    companies = input('liste entreprise : ')
    numbers = list(map(str, companies.split()))
    start_date=input('Date de début (XXXX-XX-XX) : ')
    end_date=input('Date de fin (XXXX-XX-XX) : ')
    data=data_yf(numbers,start_date,end_date)
    if int(len(data['Close'].index)/10)>=10:
        precision_recommande=int(len(data['Close'].index)/10)
    else :
        precision_recommande=10
    precision=int(input("Précision des segments ({} recomandé)) : ".format(precision_recommande)))
    visu=input('Visualisation des courbes ? (Y/N) : ')
    if len(numbers)==0:
        print("False")
    elif len(numbers)==1:
        print(cren_date(data,precision))
    elif len(numbers)>=2:
        print(cren_date_several(data,numbers,precision))
    if (len(numbers)==1) and (visu=='Y' or 'Yes' or 'y' or 'yes'):
        plot_graph(data)
        plot_segm(data,precision)
        # adding_image(plt.imread("logo hadamard stage.jfif"),2021)
        plt.show() 

    if (len(numbers)>=2) and (visu=='Y' or 'Yes' or 'y' or 'yes'):
        plot_segm_several(data,numbers,precision)
        # adding_image(plt.imread("logo hadamard stage.jfif"),2021)
        plt.show()