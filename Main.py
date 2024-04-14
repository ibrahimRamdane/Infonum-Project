from TS.Processing import *
from TS.Metrics import *
from TS.Visualization import *
from NLP.news_analysis import NewsAnalyzer
import mplcursors
import textwrap


# stocks1 =['AAPL','MSFT','TSM','ORCL','TXN','IBM','QCOM','MU','NTDOY','JD','HPQ']
# stocks2='AAPL'

if __name__ == "__main__":
    companies = input('liste entreprise : ')
    numbers = list(map(str, companies.split()))
    start_date=input('Date de début (XXXX-XX-XX) : ')
    end_date=input('Date de fin (XXXX-XX-XX) : ')
    data=data_yf(numbers,start_date,end_date)
    if int(len(data['Close'].index)/10)>=1:
        precision_recommande=int(len(data['Close'].index)/10)
    else :
        precision_recommande=1
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
        date_r=[]
        Close_r=[]
        date_g=[]
        Close_g=[]
        i=0
        for x in creneau:
            start_date_change = str(x[0])
            end_date_change = str(x[-1])
            date=cren_date(data,precision)[0]['start'][i]
            news = news_analyzer.most_important_news_embedding(0, start_date_change, end_date_change)
            L.append(news_analyzer.summarize(news))
            if  cren_date(data,precision)[0]['metric'][i]<0:   
                Close_r.append(close[i])
                date_r.append(pd.Timestamp(date))
            else:
                Close_g.append(close[i])
                date_g.append(pd.Timestamp(date))
            i=i+1
        #adding_image(plt.imread("logo hadamard stage.jfif"),2021)
        points_r=plt.scatter(date_r,Close_r,color='red')
        points_g=plt.scatter(date_g,Close_g,color='green')
        points = plt.scatter(date_r+date_g, Close_r+Close_g, color='none')
        def on_hover(sel):
    # Associer le texte correspondant à chaque point
            if sel.target.index < len(L):
                width = max(4, len(L[sel.target.index]) // 6)
                wrapped_text = textwrap.fill(L[sel.target.index], width=width)
                sel.annotation.set_text(wrapped_text)
                if sel.target_[1] in Close_r:
                  sel.annotation.get_bbox_patch().set_facecolor('red')
                elif sel.target_[1] in Close_g:
                  sel.annotation.get_bbox_patch().set_facecolor('green')
                sel.annotation.get_bbox_patch().set(alpha=0.7)
        mplcursors.cursor(points, hover=True).connect("add", on_hover)
        print(cren_date(data,precision)[0])
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
    
    
    