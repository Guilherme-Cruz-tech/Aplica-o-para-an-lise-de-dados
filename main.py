import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('transacoes.csv', encoding='latin1')
df['valor'] = df.groupby('estado_cliente')['valor'].transform(lambda x: x.fillna(x.median()))
df['plataforma'] = 'Mobile'

df['data_transacao'] = pd.to_datetime(df['data_transacao'])
df['data_transacao'] = df['data_transacao'].dt.tz_localize('America/Sao_Paulo')
df['dia_semana'] = df['data_transacao'].dt.day_name()
df['mes'] = df['data_transacao'].dt.month
df = df.drop_duplicates(keep='first')

filtro = (
    (df['mes'] == 9) &
    ((df['estado_cliente'] == 'SP') | (df['estado_cliente'] == 'RJ')) &
    (df['valor'] > 5000.00)
)
df_filtrado = df[filtro]

risco_dict = {
    'C100': 'Baixo',
    'C101': 'Alto',
    'C102': 'Médio',
    'C103': 'Baixo',
    'C104': 'Alto'
}
df['nivel_risco'] = df['id_cliente'].map(risco_dict)

pivot_risco = pd.pivot_table(
    df,
    values='valor',
    index='mes',
    columns='nivel_risco',
    aggfunc='sum',
    margins=True,
    margins_name='Total'
)

def calcular_zscore_por_estado(dataframe):
    media = dataframe.groupby('estado_cliente')['valor'].transform('mean')
    desvio = dataframe.groupby('estado_cliente')['valor'].transform('std')
    return (dataframe['valor'] - media) / desvio

df['z_score'] = calcular_zscore_por_estado(df)
df_anomalias = df[df['z_score'] > 2.5]

df['data_dia'] = df['data_transacao'].dt.date
dados_diarios = df.groupby('data_dia')['valor'].sum().reset_index()
dados_diarios['media_movel_7d'] = dados_diarios['valor'].rolling(7).mean()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(dados_diarios['data_dia'], dados_diarios['valor'], label='Valor Total Diário')
ax.plot(dados_diarios['data_dia'], dados_diarios['media_movel_7d'], label='Média Móvel (7 Dias)')

ax.set_ylim(bottom=0)
ax.set_title('Desempenho Diário das Transações e Média Móvel')
ax.set_xlabel('Data')
ax.set_ylabel('Valor (R$)')
ax.legend()

plt.tight_layout()
plt.show()