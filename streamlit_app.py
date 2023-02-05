# -*- coding: utf-8 -*-
"""
abrir o anaconda navigator,
abrir terminal no machine learning,
cd - entrar na pasta onde esta o arquivo
chamar o streamlit com o arquivo.
#(MachineLearning) cd c:/users/tf/google drive/codigos py/app

#(MachineLearning) streamlit run ./streamlit_app.py

"""

import streamlit as st
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import fundamentus as fd

import statsmodels
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
import vectorbt as vbt
from plotly.subplots import make_subplots

from datetime import date, timedelta
import datetime

import warnings
import riskfolio as rp
from fpdf import FPDF
warnings.filterwarnings('ignore')


def home():
    st.markdown('---')
    col1, col2, col3 = st.columns([0.3,1,0.3])
    with col2:
        st.title('Visão de Mercado')
        url = 'https://images.jota.info/wp-content/uploads/2021/12/usa-g893b7a7d8-1920-1024x739.jpg'
        st.image(url)
    st.markdown('---')
        
def panorama():
    st.markdown('---')
    st.title('Panorama do Mercado')
    st.markdown(date.today().strftime('%d/%m/%Y'))
    st.subheader('Mercados pelo Mundo')
    tickers = ['^BVSP', '^GSPC', '^IXIC', '^GDAXI', '^FTSE', '^CL=F', 'GC=F', 'BTC-USD', 'ETH-USD']
    precos = yf.download(tickers, start='2020-01-01')['Adj Close']
        
    variacao = precos.pct_change() #((precos.iloc[-1]/precos.iloc[-2])-1)*100
    #st.dataframe(variacao, width = 700, height = 300)
    
    #--------------------------------------------
    # Dicionário de Indices x Ticker do YFinance
    dict_tickers = {
                 'Bovespa':'^BVSP', 
                 'S&P500':'^GSPC',
                 'NASDAQ':'^IXIC', 
                 'DAX':'^GDAXI', 
                 'FTSE 100':'^FTSE',
                 'Cruid Oil': 'CL=F',
                 'Gold':'GC=F',
                 'BITCOIN':'BTC-USD',
                 'ETHEREUM':'ETH-USD'
                 }
     # tradingcomdados
     # Montagem do Dataframe de informaçções dos indices
    df_info = pd.DataFrame({'Ativo': dict_tickers.keys(),'Ticker': dict_tickers.values()})
     
    df_info['Ult. Valor'] = ''
    df_info['%'] = ''
    count =0
    with st.spinner('Atualizando cotações...'):
         for ticker in dict_tickers.values():
             cotacoes = yf.download(ticker, period='5d', interval='1d')['Adj Close']
             variacao = ((cotacoes.iloc[-1]/cotacoes.iloc[-2])-1)*100
             #st.write(variacao) # use p checar
             df_info['Ult. Valor'][count] = round(cotacoes.iloc[-1],2)
             df_info['%'][count] =round(variacao,2)
             count += 1
    #st.write(df_info)
    
    #apresentar cada indice na tela
    col1, col2, col3 = st.columns(3)
       

    with col1:
        st.metric(df_info['Ativo'][0], value=df_info['Ult. Valor'][0], delta=str(df_info['%'][0]) + '%')
        st.metric(df_info['Ativo'][1], value=df_info['Ult. Valor'][1], delta=str(df_info['%'][1]) + '%')
        st.metric(df_info['Ativo'][2], value=df_info['Ult. Valor'][2], delta=str(df_info['%'][2]) + '%')

    with col2:
        st.metric(df_info['Ativo'][3], value=df_info['Ult. Valor'][3], delta=str(df_info['%'][3]) + '%')
        st.metric(df_info['Ativo'][4], value=df_info['Ult. Valor'][4], delta=str(df_info['%'][4]) + '%')
        st.metric(df_info['Ativo'][5], value=df_info['Ult. Valor'][5], delta=str(df_info['%'][5]) + '%')

    with col3:
        st.metric(df_info['Ativo'][6], value=df_info['Ult. Valor'][6], delta=str(df_info['%'][6]) + '%')
        st.metric(df_info['Ativo'][7], value=df_info['Ult. Valor'][7], delta=str(df_info['%'][7]) + '%')
        st.metric(df_info['Ativo'][8], value=df_info['Ult. Valor'][8], delta=str(df_info['%'][8]) + '%')
    
    st.markdown('---')   
    
    #------------------------------------------
   
    st.subheader(("Comportamento Durante o Dia"))
    
    lista_indice = ['IBOV', 'S&P500', 'NASDAQ']
   
    indice = st.selectbox('Selecione o Indice', lista_indice)
    
    if indice == 'IBOV':       
         indice_diario = yf.download('^BVSP', period='1d', interval='5m')
    if indice == 'S&P500':       
         indice_diario = yf.download('^GSPC', period='1d', interval='5m')
    if indice == 'NASDAQ':       
         indice_diario = yf.download('^IXIC', period='1d', interval='5m')     
            
    fig = go.Figure(data=[go.Candlestick(x=indice_diario.index,
                                         open=indice_diario['Open'],
                                         high=indice_diario['High'],
                                         low=indice_diario['Low'],
                                         close=indice_diario['Close']
                                         )]) 
    fig.update_layout(title=indice, xaxis_rangeslider_visible=False)

    st.plotly_chart(fig)

    # Seleção de Ações    
    lista_acoes = ['PETR4.SA', 'VALE3.SA', 'EQTL3.SA','AAPL']
    acao = st.selectbox('Selecione o ativo', lista_acoes)
    #chama o yahoo finance para pegar os dados
    hist_acao = yf.download(acao, period='1d', interval='5m')
    
    # Grafico de CandleStick
    fig = go.Figure(data=[go.Candlestick(x=hist_acao.index,
                                         open=hist_acao['Open'],
                                         high=hist_acao['High'],
                                         low=hist_acao['Low'],
                                         close=hist_acao['Close'])]) 
    fig.update_layout(title=acao, xaxis_rangeslider_visible=False)

    st.plotly_chart(fig)    
    
   #st.markdown('---')
    
def mapa_mensal():
    st.markdown('---')
  
    st.title('Analise de Retornos Mensais')
    with st.expander('Escolha', expanded=True):
        opcao = st.radio('Selecione', ['Indices', 'Ações'])
    if opcao == 'Indices':
        with st.form(key='form_Indice'):
            ticker = st.selectbox('Indice', ['^BVSP', '^IXIC', '^GSPC'])
            analisar = st.form_submit_button('Analisar')
    else:
        with st.form(key='form_acoes'):
            ticker = st.selectbox('Ações', ['PETR4.SA', 'VALE3.SA', 'EQTL3.SA'])
            analisar = st.form_submit_button('Analisar')            
     
    if analisar:
        data_inicial = '2000-01-01'
        data_final = '2022-12-12'
         
        if opcao == 'Indices':
            retornos = yf.download(ticker,start = data_inicial, end=data_final, interval='1mo')['Close']
            retornos = retornos.pct_change()
            retornos = retornos.dropna()
            
        if opcao == 'Ações':
            retornos = yf.download(ticker,start = data_inicial, end=data_final, interval='1mo')['Close']
            retornos=retornos.pct_change()
            retornos = retornos.dropna()

        #st.write(retornos)   # TABELA DE APOIO PARA VER O MEIO DO CAMINHO
       # Separar e agrupar os anos e meses
        retorno_mensal = retornos.groupby([retornos.index.year.rename('Year'), retornos.index.month.rename('Month')]).mean()
        # Criar matrix de retornos
        tabela_retornos = pd.DataFrame(retorno_mensal)
        tabela_retornos = pd.pivot_table(tabela_retornos, values='Close', index='Year', columns='Month')
        tabela_retornos.columns = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
        #st.write(tabela_retornos) #APOIO P ANALISAR O Q ESTA ACONTECENDO NO MEIO DO CAMINHO
        
        # Heatmap
        fig, ax = plt.subplots(figsize=(12, 9))
        cmap = sns.color_palette('RdYlGn', 50)
        sns.heatmap(tabela_retornos, cmap=cmap, annot=True, fmt='.2%', center=0, vmax=0.02, vmin=-0.02, cbar=False,
                    linewidths=1, xticklabels=True, yticklabels=True, ax=ax)
        ax.set_title(ticker, fontsize=18)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, verticalalignment='center', fontsize='12')
        ax.set_xticklabels(ax.get_xticklabels(), fontsize='12')
        ax.xaxis.tick_top()  # x axis em cima
        plt.ylabel('')
        st.pyplot(fig)    
        st.markdown('---')
        #estatisticas
        
        stats = pd.DataFrame(tabela_retornos.mean(), columns=['Média'])
        stats['Mediana'] = tabela_retornos.median()
        stats['Maior'] = tabela_retornos.max()
        stats['Menor'] = tabela_retornos.min()
        stats['Positivos'] = tabela_retornos.gt(0).sum()/tabela_retornos.count() #conta maior q zero,soma e divide pela qtd
        stats['Negativos'] = tabela_retornos.le(0).sum()/tabela_retornos.count()
 
        #st.write(stats) #apoio
        
        stats_a = stats[['Média', 'Mediana', 'Maior', 'Menor']]
        stats_a = stats_a.transpose() 
        st.markdown('---')
        # Heatmap
        fig, ax = plt.subplots(figsize=(12, 2))
        cmap = sns.color_palette('RdYlGn', 50)
        sns.heatmap(stats_a, cmap=cmap, annot=True, fmt='.2%', center=0, vmax=0.02, vmin=-0.02, cbar=False,
                    linewidths=1, xticklabels=True, yticklabels=True, ax=ax)
        #ax.set_title(ticker, fontsize=18) # nome da tabela
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, verticalalignment='center', fontsize='11')
        ax.set_xticklabels(ax.get_xticklabels(), fontsize='11')
       #ax.xaxis.tick_top()  # x axis em cima
        st.pyplot(fig)    

        #meses positivos e negativos
        stats_b = stats[['Positivos', 'Negativos']]
        stats_b = stats_b.transpose()

        fig, ax = plt.subplots(figsize=(12, 1.5))
        sns.heatmap(stats_b, annot=True, fmt='.2%', center=0, vmax=0.02, vmin=-0.02, cbar=False,
                    linewidths=1, xticklabels=True, yticklabels=True, ax=ax)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, verticalalignment='center', fontsize='11')
        st.pyplot(fig)
        
   
    st.markdown('---')
    
    
def Fundamentos():
    st.markdown('---')
    st.title('Informações sobre Fundamentos')
    st.markdown('---')
    
    lista_tickers = fd.list_papel_all()
    #st.write(lista_tickers) #verificando se veio a lista
    
    comparar = st.checkbox('Compare com um ativo')
    comparar2 = st.checkbox('Compare com um terceiro ativo')
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.expander('Ativo 1', expanded=True):
            papel1 = st.selectbox('Selecione o Ativo', lista_tickers)
            info_papel1 = fd.get_detalhes_papel(papel1)
            st.write('**Empresa:**', info_papel1['Empresa'][0])
            st.write('**Setor:** ', info_papel1['Setor'][0])
            st.write('**Subsetor:** ', info_papel1['Subsetor'][0])
            st.write('**Valor de Mercado:** ',  f"R$ {info_papel1['Valor_de_mercado'][0]:,.2f}")
            st.write('**Patrimonio Líquido:** ', f"R$ {float(info_papel1['Patrim_Liq'][0]):,.2f}")
            st.write('**Receita Liq. 12m:** ', f"R$ {float(info_papel1['Receita_Liquida_12m'][0]):,.2f}")
            st.write('**Divida Bruta:** ', f"R$ {float(info_papel1['Div_Bruta'][0]):,.2f}")
            st.write('**Divida Liquida:** ', f"R$ {float(info_papel1['Div_Liquida'][0]):,.2f}")
            st.write('**P\L:** ', f" {float(info_papel1['PL'][0]):,.2f}")
            st.write('**Dividend Yield:** ', f"{info_papel1['Div_Yield'][0]}")
        
        
        
    if comparar:    
        with col2:
            with st.expander('Ativo 2', expanded=True):            
                papel2 = st.selectbox('Selecione o 2º Ativo', lista_tickers)   
                info_papel2 = fd.get_detalhes_papel(papel2)
                st.write('**Empresa:**', info_papel2['Empresa'][0])
                st.write('**Setor:** ', info_papel2['Setor'][0])
                st.write('**Subsetor:** ', info_papel2['Subsetor'][0])
                st.write('**Valor de Mercado:** ',  f"R$ {info_papel2['Valor_de_mercado'][0]:,.2f}")
                st.write('**Patrimonio Líquido:** ', f"R$ {float(info_papel2['Patrim_Liq'][0]):,.2f}")
                st.write('**Receita Liq. 12m:** ', f"R$ {float(info_papel2['Receita_Liquida_12m'][0]):,.2f}")
                st.write('**Divida Bruta:** ', f"R$ {float(info_papel2['Div_Bruta'][0]):,.2f}")
                st.write('**Divida Liquida:** ', f"R$ {float(info_papel2['Div_Liquida'][0]):,.2f}")
                st.write('**P\L:** ', f" {float(info_papel2['PL'][0]):,.2f}")
                st.write('**Dividend Yield:** ', f"{info_papel2['Div_Yield'][0]}")
        
            if comparar2:            
                with col3:
                    with st.expander('Ativo 3', expanded=True):                        
                        papel3 = st.selectbox('Selecione o 3º Ativo', lista_tickers)
                        info_papel3 = fd.get_detalhes_papel(papel3)
                        st.write('**Empresa:**', info_papel3['Empresa'][0])
                        st.write('**Setor:** ', info_papel3['Setor'][0])
                        st.write('**Subsetor:** ', info_papel3['Subsetor'][0])
                        st.write('**Valor de Mercado:** ',  f"R$ {info_papel3['Valor_de_mercado'][0]:,.2f}")
                        st.write('**Patrimonio Líquido:** ', f"R$ {float(info_papel3['Patrim_Liq'][0]):,.2f}")
                        st.write('**Receita Liq. 12m:** ', f"R$ {float(info_papel3['Receita_Liquida_12m'][0]):,.2f}")
                        st.write('**Divida Bruta:** ', f"R$ {float(info_papel3['Div_Bruta'][0]):,.2f}")
                        st.write('**Divida Liquida:** ', f"R$ {float(info_papel3['Div_Liquida'][0]):,.2f}")
                        st.write('**P\L:** ', f" {float(info_papel3['PL'][0]):,.2f}")
                        st.write('**Dividend Yield:** ', f"{info_papel3['Div_Yield'][0]}")



def longshort():
    st.markdown('---')
    st.title('Teste de Long & Short')
    st.markdown('---')
    
    '''escolha dos ativos para gerar a analise de long short'''
    lista_tickers = fd.list_papel_all()
    escolha_ativo2 = st.checkbox('Após escolher o Ativo 1, selecione o Ativo 2:')
    col1, col2  = st.columns(2)
    with col1:
           ativo1 = st.selectbox('Selecione o Ativo 1', lista_tickers)
    with col2:
        if escolha_ativo2:
           ativo2 = st.selectbox('Selecione o Ativo 2', lista_tickers) 
    st.markdown('---')   
    
    '''escolha das datas iniciais e final'''
    data_inicial = st.date_input(
     "Qual a data inicial?",
    datetime.date(2019, 7, 6))
    st.write('A data escolhida é:', data_inicial)
    
    today = date.today()
    #data_final=today
    data_final = st.date_input(
     "Qual a data final?",
    (today))
    st.write('A data escolhida é:', data_final)
    
    #tratando as datas para o yf poder ler
    format(data_inicial, "%Y-%m-%d")
    format(data_final, "%Y-%m-%d")
    
    #VERFICANDO O FORMATO DA DATA
    #st.write('A data escolhida é:', data_inicial)
    #st.write('A data escolhida é:', data_final)
    st.markdown('---')
    seguir_calculo = st.checkbox('Após a escolha dos ativos, inicie os cáculos:')
    if seguir_calculo:
        with st.expander('seguir com os cálculos', expanded=True):
            #ACRESCENTANDO O '.SA' PARA YF ENTENDER OS ATIVOS
            tickers = [ativo1, ativo2]
            tickers = [i + '.SA' for i in tickers]
             #criando o dataframe - executando o download, colocando data, ativo1 e ativo2;
             #definindo o index como date.
            ativos = pd.DataFrame()
             
            for i in tickers:
             
               df = yf.download(i, start=data_inicial, end=data_final)['Adj Close']
               df.rename(i, inplace=True)
               ativos = pd.concat([ativos,df], axis=1)
               ativos.index.name='Date'
             
            st.dataframe(ativos)
            
            
            # O Gráfico fica dinamico 
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ativos.index, y=ativos[ativo1 + '.SA'], name=ativo1))
            fig.add_trace(go.Scatter(x=ativos.index, y=ativos[ativo2 + '.SA'], name=ativo2))
            fig.update_layout(title_text='Preços', template='simple_white')
            #plot(fig)
            #fig.to_image(format="png", engine="kaleido")
           # fig.write_image("images/fig1.png")
            st.plotly_chart(fig)  
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ativos[ativo1 + '.SA'], y=ativos[ativo2 + '.SA'], mode='markers'))
            fig.update_layout(title_text='Preços', showlegend=True, legend_title_text=ativo1+' x '+ativo2, template='simple_white')
            #fig.to_image(format="png", engine="kaleido")
            #fig.write_image("images/fig2.png")
            st.plotly_chart(fig)  
                
            #calculando os retornos    
            retornos = ativos.pct_change()
            
            #Testando Cointegração
            ativos.dropna(inplace=True)
            
            #facilitar a vida e testar se funcionaria esse enjambre rs
            ativoA=ativo1 + '.SA'
            ativoB=ativo2 + '.SA'
           
            #st.write((ativos[ativo1 + '.SA'], ativos[ativo2 + '.SA']))
            score, pvalue, _ = coint(ativos[ativoA],ativos[ativoB])
            #valor de pValue
            st.write('pValue é: ',pvalue)        
            
                
            #Calxulando o Spread
            X1 = ativos[ativoA]
            X2 = ativos[ativoB]
            
            X1 = sm.add_constant(X1)
            resultado = sm.OLS(X2, X1).fit()
            X1 = X1[ativoA]
            beta = resultado.params[ativoA]
            
            spread = X2 - beta*X1
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=spread.index, y=spread, name='Spread'))
            fig.update_layout(title_text='Spread', template='simple_white')
            #fig.to_image(format="png", engine="kaleido")
            #fig.write_image("images/fig_spread.png")
            #fig.show()
            st.plotly_chart(fig)  
            
                
            teste = adfuller(spread)
            st.write('---')
            st.write('esse é o valor do ADFtest', teste[1])
            st.write('---')
            
            #Métrica que nos diz quantos desvios o valor está distante em relação 
            #à média, facilitando a criação de um trigger para o trading de pares
            
            z_score = (spread - spread.mean())/np.std(spread)
                
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=z_score.index, y=z_score, name='Z_Score'))
            fig.update_layout(title_text='Z_Score', template='simple_white')
            #fig.to_image(format="png", engine="kaleido")
            #fig.write_image("images/fig_z_score.png")
            #fig.show()
            st.plotly_chart(fig)  
            
            #Ajustando para colocar os desvios padrões
            ativos['cte']=''
            ativos['cte2']=''
            ativos['cte']=2
            ativos['cte2']=-2
            
            
            #GERAÇÃO DE GRÁFICO DE ANALISE DE z_score E desvio padrão
            fig = make_subplots(rows=2, cols=1)
            fig.add_trace(go.Scatter(x=ativos.index, y=ativos[ativoA], name=ativoA),row=1, col=1)
            fig.add_trace(go.Scatter(x=ativos.index, y=ativos[ativoB], name=ativoB),row=1, col=1)
            #adaptando
            fig.add_trace(go.Scatter(x=ativos.index, y=ativos['cte'], name='dsv+2'),row=2, col=1)
            fig.add_trace(go.Scatter(x=ativos.index, y=ativos['cte2'],name='dsv-2'),row=2, col=1)
            fig.add_trace(go.Scatter(x=z_score.index, y=z_score, name='Z_Score'),row=2, col=1)
            fig.update_layout(title_text='Análise Z_Score', template='simple_white',height=500, width=600)
            st.plotly_chart(fig)  
               
            """
            Definir o threshold do z-score
            
            z-score > threshold = vender ação X2 e comprar ação X1 (SHORT THE SPREAD)
            
            z-score < threshold = comprar ação X2 e vender ação X1 (LONG THE SPREAD)
            """
            st.markdown('---')
            seguir_calculo2 = st.button('Prosseguir com os cáculos:')
            if seguir_calculo2:
                ativos_open = pd.DataFrame()
                
                for i in tickers:
                  df2 = yf.download(i, start=data_inicial, end=data_final)['Open']
                  df2.rename(i, inplace=True)
                  ativos_open = pd.concat([ativos_open,df2],axis=1)
                  ativos_open.index.name='Date'
                ativos_open
                
                    
                #Parametros PARA O BACKTEST
                
                CAIXA = 200000 #ABSOLUTO
                TAXA = 0.0001
                PCT_ORDEM1 = 0.10
                PCT_ORDEM2 = 0.10
                BANDA_SUPERIOR = 1.0
                BANDA_INFERIOR = -1.0
                
                
                
                #Gerar o sinal de long short
                
                ########fig_1, ax_1 = plt.subplots(figsize=(10,6))
                ######## testar a formula acima para fazer print do grafico
                ########
                
                vbt_sinal_short = (z_score > BANDA_SUPERIOR).rename('sinal_short')
                vbt_sinal_long  = (z_score < BANDA_INFERIOR).rename('sinal_long')
                
                pd.Series.vbt.signals.clean(
                 vbt_sinal_short, vbt_sinal_long, entry_first=False, broadcast_kwargs=dict(columns_from='keep'))
                
                #Garantir mesmo tamanho para os vetores de sinal
                vbt_sinal_short, vbt_sinal_long = pd.Series.vbt.signals.clean(
                  vbt_sinal_short, vbt_sinal_long, entry_first=False, broadcast_kwargs=dict(columns_from='keep')
                )
                
                
                tickers_coluna = pd.Index([ativoA, ativoB], name='tickers')
                vbt_ordem = pd.DataFrame(index=ativos.index, columns=tickers_coluna)
                vbt_ordem[ativoA] = np.nan
                vbt_ordem[ativoB] = np.nan
                vbt_ordem.loc[vbt_sinal_short, ativoA] = -PCT_ORDEM1
                vbt_ordem.loc[vbt_sinal_long, ativoB] = PCT_ORDEM1
                vbt_ordem.loc[vbt_sinal_short, ativoA] = PCT_ORDEM2
                vbt_ordem.loc[vbt_sinal_long, ativoB] = -PCT_ORDEM2
                
                
                vbt_ordem = vbt_ordem.vbt.fshift(1)
                
                #eu criei duas constantes para gerar o gráfico do desvio padrão anterior, precisa tirar para dar certo o calculo!!!
                ativos = ativos.drop(['cte','cte2'], axis=1)
                
                def portfolio_pairs_trading():
                
                    return vbt.Portfolio.from_orders(
                        ativos,
                        size=vbt_ordem,
                        price=ativos_open,
                        size_type='targetpercent',
                        val_price=ativos.vbt.fshift(1),
                        init_cash=CAIXA,
                        fees=TAXA,
                        cash_sharing=True,
                        group_by=True,
                        call_seq='auto',
                        freq='d'
                    )
                
                vbt_pf = portfolio_pairs_trading()
                
                st.write(vbt_pf.orders.records_readable)
                st.write(vbt_pf.stats())
                
                #grafico de dados acumulativo
                vbt_pf.plot().show();
                
                #plot_underwater
                vbt_pf.plot_underwater().show();
                
                #gráficos de drawdowns
                vbt_pf.drawdowns.plot(top_n=15).show();
                
                        
  


def otimiza():
    st.markdown('---')
    st.title('Realize a otimização de sua carteira')
    st.markdown('---')
    
    #Usa o cache para adiacionar os papeis na memoria e por na lista
    @st.cache(allow_output_mutation=True)
    def get_data():
        return []  
    
    #escolha efetiva do papel.
    st.markdown('---')
    st.subheader('Selecione os papeis')
    lista_tickers = fd.list_papel_all()
    ativo_selecionado = st.selectbox('Selecione o Ativo', lista_tickers)
    st.markdown('---')
    #Definir a quantidade de 0 até 100%
    qtde = st.slider("Peso%", 0, 100)
    
    #Se errou - ter como apagar
    if st.button("Apagar"):
        with st.spinner('Apagando a lista de ativos...'):
            get_data().clear()
    #Adição efetiva do papel e % no dataframe        
    if st.button("Adicionar"):
        with st.spinner('Atualizando ativos...'):
            get_data().append({"Ativo": ativo_selecionado, "Peso%": qtde})
    #mostra de resultado, avisa se passou de 100%
    if get_data() !=[]:
        st.write(pd.DataFrame(get_data()))
        ativos_df = pd.DataFrame(get_data())    
        st.write('A soma total está em ', ativos_df["Peso%"].sum(), '%') 
        if ativos_df["Peso%"].sum()>100:
            st.warning('Passou de 100%')           
        
        #adaptação do codigo para array de pesos
        peso_in = np.array(ativos_df["Peso%"])
         
        #Definir a data inicial para puxar os dados
        st.markdown('---')   
        
        '''escolha das datas iniciais e final'''
        
        data_inicial = st.date_input(
        "Qual a data inicial?",
        datetime.date(2015, 7, 6))
        st.write('A data escolhida é:', data_inicial)
        
        data_sugerida = data_inicial +  timedelta(days = 730)
        data_final = st.date_input(
            "Qual a data final?",
            (data_sugerida))
        st.write('A data escolhida é:', data_final)
           
           #tratando as datas para o yf poder ler
        format(data_inicial, "%Y-%m-%d")
        format(data_final, "%Y-%m-%d")
  
        #Acrescentando o .SA necessário para a leitura do yf
        tickers = [i + '.SA' for i in ativos_df['Ativo']]
        #criando o dataframe onde será armazenado
        ativos_carteira = pd.DataFrame()
        for i in tickers:
     
           df = yf.download(i, start=data_inicial, end=data_final)['Adj Close']
           df.rename(i, inplace=True)
           ativos_carteira = pd.concat([ativos_carteira,df], axis=1)
           ativos_carteira.index.name='Date'
        #checando o dataframe
        st.dataframe(ativos_carteira)
        
        #calculando os retornos     #Excluindo os NAN 
        retorno_carteira = ativos_carteira.pct_change().dropna()
        #Calcula o retorno e pega a covariancia
        cov_in = retorno_carteira.cov()
        #montando a matriz de pesos e ativos
        pesos_in = pd.DataFrame(data={'pesos_in':peso_in}, index=tickers)
       
        
        #"""Proxima etapa - carteira de comparação, a out"""
        
        #'''escolha das datas iniciais e final'''
        
        data_inicial2 =  st.date_input(
            "Qual a segunda data inicial?",
            (data_final))
        st.write('A data escolhida é:', data_final)               
        
        
        today = date.today()
           #data_final=today
        data_final2 = st.date_input(
            "Qual a data final?",
            (today))
        st.write('A data escolhida é:', data_final2)
           
           #tratando as datas para o yf poder ler
        format(data_inicial2, "%Y-%m-%d")
        format(data_final2, "%Y-%m-%d")
        
        # fim das datas da segunda carteira
        
        #inicio para o download da segunda etapa
        #Acrescentando o .SA necessário para a leitura do yf
       # tickers = [i + '.SA' for i in ativos_df['Ativo']]
        #criando o dataframe onde será armazenado
        ativos_carteira_out = pd.DataFrame()
        for i in tickers:
         
           df = yf.download(i, start=data_inicial2, end=data_final2)['Adj Close']
           df.rename(i, inplace=True)
           ativos_carteira_out = pd.concat([ativos_carteira_out,df], axis=1)
           ativos_carteira_out.index.name='Date'
        #checando o dataframe
        st.dataframe(ativos_carteira_out)
        
        #calculando os retornos     #Excluindo os NAN 
        retorno_carteira_out = ativos_carteira_out.pct_change().dropna()
        #Calcula o retorno e pega a covariancia
        cov_out = retorno_carteira_out.cov()
        
       
        
        #Hierarchical Risk Parity 
        #"Marcos López de Prado. 
        #Building diversified portfolios that outperform out of sample. 
        #The Journal of Portfolio Management, 42(4):59–69, 2016. 
        #URL: https://jpm.pm-research.com/content/42/4/59, 
        #arXiv:https://jpm.pm-research.com/content/42/4/59.full.pdf, 
        #doi:10.3905/jpm.2016.42.4.059.""
        
        pd.options.display.float_format = ' {:.4%}'.format
        
        portfolio = rp.HCPortfolio(returns=retorno_carteira)
        model = 'HRP'
        codependence = 'pearson'
        rm = 'MV'
        rf = 0
        linkage = 'single'
        leaf_order = True
        
        pesos = portfolio.optimization(model=model,
                                       codependence=codependence,
                                       rm=rm,
                                       rf=rf,
                                       leaf_order=leaf_order)
                    
        
        ax = rp.plot_dendrogram(returns=retorno_carteira,
                      codependence='pearson',
                      linkage='single',
                      k=None,
                      max_k=10,
                      leaf_order=True,
                      ax=None)
        
        plt.savefig('dendrogram')

        
        ax = rp.plot_network(returns=retorno_carteira, codependence="pearson",
                     linkage="ward", k=None, max_k=10,
                     alpha_tail=0.05, leaf_order=True,
                     kind='spring', ax=None)
        plt.savefig('network')
        
        #Retorno Acumulado out of sample
        fig_2, ax_2 = plt.subplots()
        rp.plot_series(returns=retorno_carteira_out, w=pesos, cmap='tab20', 
                       height=6, width=10,
                       ax=None)
        plt.savefig('cum_ret.png')
        
        #Gráfico de composição dos novos pesos antes da otimização
        fig_2, ax_2 = plt.subplots(figsize=(6,2))
        rp.plot_pie(w=pesos_in, title='Porfolio', height=6, width=10,
                    cmap='tab20', 
                    ax=None)
        plt.savefig('portfolio_pesos_iniciais')
        
        
        #Gráfico de composição dos novos pesos da carteira otimizada
        
        fig_3, ax_3 = plt.subplots(figsize=(6,2))
        rp.plot_pie(w=pesos, title='Portfolio',
                    height=6,
                    width=10,
                    cmap='tab20',
                    ax=None)
        plt.savefig('portfolio_pesos_otimizado')
        
        
        ### Contribuição de risco por ativo
        #Parâmetros do portfolio otimizado
        
        media_retorno = portfolio.mu
        covariancia = portfolio.cov
        retornos = portfolio.returns 

        #grafico de contribuição de medida de risco por ativo da cateira
        
        fig_4, ax_4 = plt.subplots(figsize=(6,2))
        rp.plot_risk_con(w=pesos, 
                         cov=cov_in,
                         returns=retorno_carteira,
                         rm=rm,
                         rf=0,
                         alpha=0.05,
                         color='tab:blue',
                         height=6,
                         width=10,
                         t_factor=252,
                         ax=None)
        plt.savefig('risk_contr_ativo_inicial')
        
        #Gráfico de contribuição de medida de risco por ativo carteira as is

        fig_5, ax_5 = plt.subplots(figsize=(6,2))
        
        rp.plot_risk_con(w=pesos, cov=cov_out, returns=retorno_carteira_out, rm=rm,
                              rf=0, alpha=0.05, color="tab:blue", height=6,
                              width=10, t_factor=252, ax=None)
        plt.savefig('risk_cont_ativo_otimizado.png')

        #Histograma dos retornos do portfolio
        
        fig_6, ax_6 = plt.subplots()
        
        rp.plot_hist(returns=retorno_carteira, w=pesos_in, alpha=0.05, bins=50, height=6,
                          width=10, ax=None)
        plt.savefig('pf_returns_in.png');
                
        #Histograma dos retornos do portfolio
        
        fig_7, ax_7 = plt.subplots()
        
        rp.plot_hist(returns=retorno_carteira_out,
                     w=pesos, alpha=0.05, bins=50, height=6,
                          width=10, ax=None);
        plt.savefig('pf_returns_out.png')
        
        fig_8, ax_8 = plt.subplots(figsize=(6,2))
        rp.plot_table(returns=retorno_carteira, w=pesos_in, MAR=0, alpha=0.05, ax=None)
        plt.savefig('table_in.png');
        
        fig_9, ax_9 = plt.subplots(figsize=(6,2))
        rp.plot_table(returns=retorno_carteira_out, w=pesos, MAR=0, alpha=0.05, ax=None)
        plt.savefig('table_out.png');
        
        # 1. Setup básico do PDF

        #Criar o pdf
        #DESCOBRIR COMO CENTRALIZAR OS DS GRÁFICOS E TEXTOS.... MENSURAR A ARQUITETURA ANTES
        pdf = FPDF()
        
        #Adiciona uma nova pagina
        pdf.add_page()
        
        #Setup da fonte
        pdf.set_font('Arial', 'B', 16)
        
                #2. Layout do PDF
        pdf.cell(40,10, 'Diagnóstico da sua Carteira')
        
        #Quabra de linha
        pdf.ln(20)
        
        #3. Tabela Performance
        pdf.cell(20,7,'Como sua carteira performou de {} até {}'.format(data_inicial, data_final))
        pdf.ln(8)
        pdf.image('table_in.png', w=100, h=200)
        pdf.ln(60)
        
        #4 tabela performance out-of-sample
        pdf.cell(20, 7,'Como sua carteira performou de {} até {}'.format(data_final, data_final2))
        pdf.ln(8)
        pdf.image('table_out.png', w=120, h=70)
        pdf.ln(10)
        
        #5 Retorno Acumuiado Carteira
        pdf.cell(20, 7, 'Retorno Acumulado da Carteira de {} até {}'.format(data_final,data_final2))
        pdf.ln(8)
        pdf.image('cum_ret.png', w=120, h=70)
        pdf.ln(10)
        
        #6 Pesos
        
        
        pdf.cell(20, 7, "Pesos da Carteira Atual")
        pdf.ln(8)
        pdf.image('portfolio_pesos_iniciais.png', w=100, h=60)
        pdf.ln(50)
        
        pdf.cell(20, 7, 'Pesos da Carteira Otimizada')
        pdf.ln(8)
        pdf.image('portfolio_pesos_otimizado.png', w=100, h=60)
        pdf.ln(30)          
                
        #7 Contribuição de risco por ativo
        
        pdf.cell(20, 7, "Contribuição de risco por ativo de {} ate {}".format(data_inicial, data_final))
        pdf.ln(8)
        pdf.image('risk_contr_ativo_inicial.png', w=150, h=70)
        pdf.ln(30)
        
        pdf.cell(20, 7, "Contribuição de risco, na carteira sugerida, por ativo de {} ate {}".format(data_final,data_final2))
        pdf.ln(8)
        pdf.image('risk_cont_ativo_otimizado.png', w=150, h=70)
        pdf.ln(30)
        
        #8 Histograma de retornos
        
        pdf.cell(20, 7, "Histograma de retornos de {} ate {}".format(data_inicial, data_final))
        pdf.ln(8)
        pdf.image('pf_returns_in.png', w=150, h=70)
        pdf.ln(20)
         
        pdf.cell(20, 7, "Histograma de retornos, na carteira sugerida, de {} ate {}".format(data_final,data_final2))
        pdf.ln(8)
        pdf.image('pf_returns_out.png', w=150, h=70)
        pdf.ln(30)
        
        
        # 9. Disclaimer
        pdf.set_font('Times', '', 6)
        pdf.cell(5, 2, 'Relatório construído com a biblioteca RiskFolio https://riskfolio-lib.readthedocs.io/en/latest/index.html')
        
        
        # 10. Output do PDF file
        pdf.output('diagnostico_de_carteira.pdf', 'F')
        
        with open("diagnostico_de_carteira.pdf", "rb") as pdf_file:
            PDFbyte = pdf_file.read()
        
        st.download_button(label="Exportar Relatório",
                            data=PDFbyte,
                            file_name="diagnostico_de_carteira.pdf",
                            mime='application/octet-stream')


def main():
    url = 'https://images.jota.info/wp-content/uploads/2021/12/usa-g893b7a7d8-1920-1024x739.jpg'
    st.sidebar.image(url, width=(200))
    st.sidebar.title('APP - Mercado Financeiro')
    st.sidebar.markdown('---')
    lista_menu = ['Home', 'Panorama do Mercado', 'Rentabilidade Mensais', 'Fundamentos', 'Long&Short', 'Otimizador de Carteira']
    escolha = st.sidebar.radio('Escolha a opção', lista_menu)
    
    if escolha == 'Home':
        home()
    if escolha == 'Panorama do Mercado':
        panorama()
    if escolha == 'Rentabilidade Mensais':
        mapa_mensal()
    if escolha == 'Fundamentos':
        Fundamentos()
    if escolha == 'Long&Short':
        longshort()
    if escolha == 'Otimizador de Carteira':
        otimiza()
        
    
main()
