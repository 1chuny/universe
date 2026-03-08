import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# ==========================================
# ПАРАМЕТРИ ДЛЯ НАЛАШТУВАННЯ
# ==========================================
n = 32
COHORTS_TO_PREDICT = list(range(1, n))
FORECAST_WEEKS = 52
# ==========================================

def clean_currency(x):
    if pd.isna(x) or str(x).strip() == '': return 0.0
    return float(str(x).replace('$', '').replace('\xa0', '').replace(' ', '').replace(',', '.'))


def power_law(x, a, b):
    """Модель для прогнозування: y = a * (x + 1)^b"""
    return a * np.power(x + 1, b)


def run_forecast():
    # 1. Завантаження даних
    installs_df = pd.read_csv('Analytics part.xlsx - Task1. Installs.csv')
    installs_df.columns = ['Cohort', 'Installs']

    rev_raw = pd.read_csv('Analytics part.xlsx - Task 1. Revenue Cohort.csv', header=None)
    revenue_data = rev_raw.iloc[2:, :].copy()
    revenue_data = revenue_data.iloc[:, [0] + list(range(2, 47))]
    rev_cols = ['Cohort'] + [str(i) for i in range(45)]
    revenue_data.columns = rev_cols

    # 2. Очищення даних
    for col in rev_cols[1:]:
        revenue_data[col] = revenue_data[col].apply(clean_currency)
    revenue_data['Cohort'] = pd.to_numeric(revenue_data['Cohort'])
    installs_df['Cohort'] = pd.to_numeric(installs_df['Cohort'])
    merged = pd.merge(revenue_data, installs_df, on='Cohort')

    # 3. Розрахунок середнього маржинального доходу (Marginal ARPU) для навчання моделі
    weeks = list(range(45))
    marginal_arpus = []
    for w in weeks:
        col_name = str(w)
        # Беремо когорти, які мають дані за цей тиждень (C + w <= 45)
        active_cohorts = merged[merged['Cohort'] <= (45 - w)]
        avg_rev = active_cohorts[col_name].sum() / active_cohorts['Installs'].sum()
        marginal_arpus.append(avg_rev)

    # 4. Навчання моделі (Fitting)
    popt, _ = curve_fit(power_law, weeks, marginal_arpus)
    print(f"--- Модель навчена. Параметри кривої: a={popt[0]:.4f}, b={popt[1]:.4f} ---\n")

    # 5. Прогнозування для вибраних когорт
    results = []
    total_revenue_all_cohorts = 0  # Для збору всього доходу
    total_installs_all_cohorts = 0  # Для збору всіх юзерів

    for cohort_id in COHORTS_TO_PREDICT:
        cohort_data = merged[merged['Cohort'] == cohort_id]
        if cohort_data.empty:
            continue

        installs = cohort_data['Installs'].values[0]
        weeks_actual = 45 - cohort_id + 1
        actual_rev_cols = [str(w) for w in range(min(weeks_actual, 45))]
        total_actual_rev = cohort_data[actual_rev_cols].sum(axis=1).values[0]

        projected_rev = 0
        if FORECAST_WEEKS > weeks_actual:
            for w in range(weeks_actual, FORECAST_WEEKS):
                projected_rev += power_law(w, *popt) * installs

        # Накопичуємо загальні суми
        total_revenue_all_cohorts += (total_actual_rev + projected_rev)
        total_installs_all_cohorts += installs

        # LTV конкретної когорти
        cohort_ltv = (total_actual_rev + projected_rev) / installs

        results.append({
            'Когорта': cohort_id,
            'Встановлення': installs,
            'Фактичний дохід': round(total_actual_rev, 2),
            'Прогноз доходу (додатково)': round(projected_rev, 2),
            'Фінальний LTV (прогноз)': round(cohort_ltv, 4)
        })

    # 6. Фінальний розрахунок
    avg_ltv_total = total_revenue_all_cohorts / total_installs_all_cohorts

    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))

    print("\n" + "=" * 50)
    print(f"СЕРЕДНІЙ ПОКАЗНИК ДЛЯ КОГОРТ {min(COHORTS_TO_PREDICT)}-{max(COHORTS_TO_PREDICT)}:")
    print(f"Компанія заробляє у середньому за {FORECAST_WEEKS} тижнів: {avg_ltv_total:.2f}$ з одного користувача")
    print("=" * 50)


if __name__ == "__main__":
    run_forecast()