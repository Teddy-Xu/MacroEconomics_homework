import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu
import os

file_path = 'CDIS_10-20-2024_03-06-30-04_timeSeries.csv'
data = pd.read_csv(file_path)


def safe_filename(filename):
    # Define a function to make the filename safe
    return filename.replace(", ", "_").replace(":", "_").replace(" ", "_")


# 创建results文件夹如果不存在
if not os.path.exists('results'):
    os.makedirs('results')

countries = [
    ('China, P.R.: Mainland', 'Vietnam'),
    ('China, P.R.: Mainland', 'Mexico'),
    ('China, P.R.: Mainland', 'United States'),
    ('China, P.R.: Mainland', 'India'),
    ('China, P.R.: Mainland', 'Japan'),
    ('China, P.R.: Mainland', 'Germany'),
    ('China, P.R.: Mainland', 'United Kingdom'),
    ('China, P.R.: Mainland', 'Russian Federation'),
    ('China, P.R.: Mainland', 'Indonesia'),
]

years = [str(year) for year in range(2009, 2023)]

# 将非数值（如'C'）替换为NaN
data[years] = data[years].apply(pd.to_numeric, errors='coerce')


def calculate_growth_rates(values):
    return [
        (values[i] - values[i - 1]) / values[i - 1] if values[i - 1] != 0 else np.nan
        for i in range(1, len(values))
    ]


def clean_zero_data(values, years):
    valid_years = []
    valid_values = []
    for year, value in zip(years, values):
        if not np.isnan(value) and value != 0:
            valid_years.append(year)
            valid_values.append(value)
    return valid_years, valid_values


with open('results/result.txt', 'w') as f:
    for country, counterpart in countries:
        # 对外投资分析
        outward_data = data[(data['Country Name'] == country)
                            & (data['Counterpart Country Name'] == counterpart) &
                            (data['Indicator Name'].str.contains('Outward Direct Investment'))]

        p_value = None
        if not outward_data.empty:
            outward_sum = outward_data[years].sum()
            valid_years, valid_values = clean_zero_data(outward_sum.values, years)
            growth_rates = calculate_growth_rates(valid_values)

            midpoint = len(growth_rates) // 2
            pre_2020_growth = [gr for gr in growth_rates[:midpoint] if not np.isnan(gr)]
            post_2020_growth = [gr for gr in growth_rates[midpoint:] if not np.isnan(gr)]

            if len(pre_2020_growth) > 1 and len(post_2020_growth) > 1:
                u_statistic, p_value = mannwhitneyu(
                    pre_2020_growth, post_2020_growth, alternative='two-sided'
                )
                f.write(f'Outward Investment {country} -> {counterpart}: p-value = {p_value:.5f}\n')

        plt.figure(figsize=(7, 4))
        plt.plot(valid_years, valid_values, label=f'{country} -> {counterpart} Outward', marker='o')
        plt.axvline(x='2019', color='grey', linestyle='--', linewidth=1)
        plt_title = f'Outward Investment Trend\n{country} -> {counterpart}'
        if p_value is not None:
            plt_title += f'\np-value = {p_value:.5f}'
        plt.title(plt_title)
        plt.xlabel('Year')
        plt.ylabel('Investment (USD)')
        plt.legend()
        # 保存外向投资图
        plt.savefig(
            os.path.join(
                'results', f'Outward_{safe_filename(country)}_{safe_filename(counterpart)}.png'
            )
        )
        plt.close()

        # 对内投资分析
        inward_data = data[(data['Country Name'] == country)
                           & (data['Counterpart Country Name'] == counterpart) &
                           (data['Indicator Name'].str.contains('Inward Direct Investment'))]

        p_value = None
        if not inward_data.empty:
            inward_sum = inward_data[years].sum()
            valid_years, valid_values = clean_zero_data(inward_sum.values, years)
            growth_rates = calculate_growth_rates(valid_values)

            midpoint = len(growth_rates) // 2
            pre_2020_growth = [gr for gr in growth_rates[:midpoint] if not np.isnan(gr)]
            post_2020_growth = [gr for gr in growth_rates[midpoint:] if not np.isnan(gr)]

            if len(pre_2020_growth) > 1 and len(post_2020_growth) > 1:
                u_statistic, p_value = mannwhitneyu(
                    pre_2020_growth, post_2020_growth, alternative='two-sided'
                )
                f.write(f'Inward Investment {country} <- {counterpart}: p-value = {p_value:.5f}\n')

        plt.figure(figsize=(7, 4))
        plt.plot(valid_years, valid_values, label=f'{country} <- {counterpart} Inward', marker='o')
        plt.axvline(x='2019', color='grey', linestyle='--', linewidth=1)
        plt_title = f'Inward Investment Trend\n{country} <- {counterpart}'
        if p_value is not None:
            plt_title += f'\np-value = {p_value:.5f}'
        plt.title(plt_title)
        plt.xlabel('Year')
        plt.ylabel('Investment (USD)')
        plt.legend()
        # 保存内向投资图
        plt.savefig(
            os.path.join(
                'results', f'Inward_{safe_filename(country)}_{safe_filename(counterpart)}.png'
            )
        )
        plt.close()

print("Analysis complete.")
