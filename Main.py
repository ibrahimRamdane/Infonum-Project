from TS.Processing import *
from TS.Metrics import *
from TS.Visualization import *
from NLP.news_analysis import NewsAnalyzer

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
        file = 'df embedding and entity and cluster and similarity- 1 janvier to 7 mars'
        df = pd.read_pickle(file)
        news_analyzer = NewsAnalyzer(df)
        #news_analyzer.plot_of_clusters()
        news_analyzer.add_df_similar_embedding(0.6)
        news_analyzer.add_df_similar_entity(0.5)

        index_1 = news_analyzer.get_i_most_similar_embedding_index(0.6, 4)
        index_2 = news_analyzer.get_i_most_similar_entity_index(0.5, 4)
        index_1_2 = set(index_1).union(set(index_2))
        creneau=cren_date(data,precision)[0]['creneau']
        close=cren_date(data,precision)[0]['Close']
        plot_graph(data,numbers)
        plot_segm(data,precision)
        L=[]
        dict={}
        i=0
        for x in creneau:
            start_date_change = str(x[0])
            end_date_change = str(x[-1])
            date=x[5]
            news = news_analyzer.most_important_news_embedding(0, start_date_change, end_date_change)
            L.append(news_analyzer.summarize(news))
        #adding_image(plt.imread("logo hadamard stage.jfif"),2021)
            plt.text(pd.Timestamp(date), close[i], L[i], fontsize=5, color='red')
            i=i+1
        plt.show() 
    
    elif len(numbers)>=2:
        # print(cren_date_several(data,numbers,precision))
        file = 'df embedding and entity and cluster and similarity- 1 janvier to 7 mars'
        df = pd.read_pickle(file)
        news_analyzer = NewsAnalyzer(df)
        #news_analyzer.plot_of_clusters()
        news_analyzer.add_df_similar_embedding(0.6)
        news_analyzer.add_df_similar_entity(0.5)

        index_1 = news_analyzer.get_i_most_similar_embedding_index(0.6, 4)
        index_2 = news_analyzer.get_i_most_similar_entity_index(0.5, 4)
        index_1_2 = set(index_1).union(set(index_2))
        creneau=cren_date(data,numbers,precision)[0]['creneau']
        L=[]
        for x in creneau:
            start_date = str(x[0])
            end_date = str(x[-1])
            news = news_analyzer.most_important_news_embedding(0, '2024-02-27', '2024-02-28')
            L.append(news_analyzer.summarize(news))
        print(L)
        
    # if (len(numbers)==1) and (visu=='Y' or 'Yes' or 'y' or 'yes'):
    #     plot_graph(data,numbers)
    #     plot_segm(data,precision)
    #     #adding_image(plt.imread("logo hadamard stage.jfif"),2021)
    #     plt.show() 

    if (len(numbers)>=2) and (visu=='Y' or 'Yes' or 'y' or 'yes'):
        plot_segm_several(data,numbers,precision)
        # adding_image(plt.imread("logo hadamard stage.jfif"),2021)
        plt.show()
    
    
    