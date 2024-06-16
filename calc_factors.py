import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import datetime
from sklearn.decomposition import FastICA
from statsmodels.tsa.stattools import adfuller
import arch
import warnings
warnings.filterwarnings('ignore')

financial_columns = ['close_B_1', 'close_B_2', 'close_B_3', 'close_B_4', 'close_B_5',\
       'close_AFKS', 'close_AGRO', 'close_BRENT', 'close_EUR_RUB__TOD',\
       'close_GAZP', 'close_IMOEX', 'close_LKOH', 'close_NLMK', 'close_NVTK',\
       'close_PLZL', 'close_ROSN', 'close_RTKM', 'close_RTSI', 'close_SBER',\
       'close_USD000000TOD', '%_0,25y', '%_0,5y', '%_0,75y', '%_1y', '%_2y',\
       '%_3y', '%_5y', '%_7y', '%_10y', '%_15y', '%_20y', '%_30y'
       # ,\
       # 'discounted_coupon_1', 'discounted_coupon_2', 'discounted_coupon_3',\
       # 'discounted_coupon_4', 'discounted_coupon_5'
       ]

dataset = pd.read_excel('dataset_coupons.xlsx')
def get_risk_factors(dt_end = datetime.date(2023 , 12, 3), dataset = dataset, financial_columns = financial_columns):
    dt_end_datetime = pd.to_datetime(dt_end)
    # Фильтрация DataFrame по условию dt < dt_end
    dataset = dataset[dataset['dt'] < dt_end_datetime].reset_index().drop(columns = ['index'])
    df_financial = dataset[financial_columns]

    # Инициализация модели ICA
    ica = FastICA(n_components=10, algorithm='parallel', whiten=True, fun='logcosh', max_iter=300)

    # подгонка модели к данным и извлечение независимых компонент
    ica_components = ica.fit_transform(df_financial)

    # создание нового DataFrame с независимыми компонентами как столбцами
    df_ica = pd.DataFrame(data=ica_components)

    # вывод результата
    df_ica.columns = [f'risk_factor_{i}' for i in range(len(df_ica.columns))]
    df_ica['dt'] = dataset['dt']
    df_ica = df_ica[['dt'] +[f'risk_factor_{i}' for i in range(len(df_ica.columns)-1)]]
    # Объясненная дисперсия для каждой компоненты
    dataset = dataset.merge(df_ica, how = 'left', on = 'dt')

    return dataset

def garch_simulation(df, risk_factor_col, steps_forward, num_samples):
    model = arch.arch_model(df[risk_factor_col], mean = 'AR', vol='Garch', p=10, o=0, q=10)
    res = model.fit(disp='off')
    sim_data = res.forecast(horizon=steps_forward, method='simulation')
    means = sim_data.mean.iloc[0].values
    variances = sim_data.variance.iloc[0].values
    # Симуляция нормального распределения
    simulated_data = np.zeros((num_samples, len(means)))
    for i in range(len(means)):
        std_dev = np.sqrt(variances[i])
        simulated_data[:, i] = np.random.normal(means[i], std_dev, num_samples)
    return simulated_data[:, :]

def calc_and_simulate_risk_factors(dt_end = datetime.date(2023 , 12, 3), steps_forward = 1, num_samples = 1000, dataset = dataset, financial_columns = financial_columns):
    df_risks = get_risk_factors(dt_end = dt_end, dataset = dataset, financial_columns = financial_columns)
    """
    Вычисляет и симулирует факторы риска.

    Args:
        dt_end (datetime.date): Дата, к которой требуется провести расчет рисков. По умолчанию - 3 декабря 2023 года.
        steps_forward (int): Шаги вперед для прогнозирования. По умолчанию - 1.
        num_samples (int): Количество сэмплов для симуляции. По умолчанию - 1000.
        dataset (pd.DataFrame): Набор данных с рисками и финансовыми данными.
        financial_columns (list): Список финансовых колонок для рассмотрения.

    Returns:
        (pd.DataFrame, dict): Датафрейм с факторами риска и словарь результатов симуляции рисков по каждому фактору.
    """
    df = df_risks[[f'risk_factor_{i}' for i in range(10)]].copy()
    df.index = df_risks['dt']
    dict_of_risk_simulations = dict()
    for risk_factor_col in df.columns:
        dict_of_risk_simulations[risk_factor_col] = garch_simulation(df, risk_factor_col, steps_forward = steps_forward, num_samples = num_samples)
    return df_risks, dict_of_risk_simulations