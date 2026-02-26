import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

def generate_synthetic_data(n=3000):
    data = {}

    data["candidate_id"] = range(1, n+1)

    data["response_delay_hours"] = np.random.exponential(scale=24, size=n)
    data["number_of_active_offers"] = np.random.poisson(0.8, size=n)
    data["past_job_switches"] = np.random.randint(0, 6, size=n)
    data["avg_tenure_months"] = np.random.randint(6, 48, size=n)
    data["notice_period_days"] = np.random.choice([0, 15, 30, 60, 90], size=n)

    data["salary_expectation_lpa"] = np.random.uniform(4, 25, size=n)
    data["salary_offered_lpa"] = data["salary_expectation_lpa"] * np.random.uniform(0.9, 1.3, size=n)
    data["salary_hike_percent"] = ((data["salary_offered_lpa"] - data["salary_expectation_lpa"]) / data["salary_expectation_lpa"]) * 100

    data["interview_sentiment"] = np.random.uniform(0, 1, size=n)
    data["communication_score"] = np.random.randint(30, 100, size=n)
    data["technical_score"] = np.random.randint(40, 100, size=n)

    data["background_verified"] = np.random.choice([0, 1], size=n, p=[0.1, 0.9])

    df = pd.DataFrame(data)

    # ----------- Backout Probability Logic ------------
    prob = np.ones(n) * 0.9

    prob -= np.where(df["response_delay_hours"] > 48, 0.25, 0)
    prob -= np.where(df["number_of_active_offers"] >= 2, 0.30, 0)
    prob -= np.where(df["communication_score"] < 40, 0.15, 0)
    prob -= np.where(df["technical_score"] < 50, 0.10, 0)
    prob -= np.where(df["salary_hike_percent"] > 40, 0.20, 0)

    prob = np.clip(prob, 0, 1)

    df["joined"] = np.random.binomial(1, prob)

    return df


if __name__ == "__main__":
    df = generate_synthetic_data()
    df.to_csv("data/udyogaa_synthetic.csv", index=False)
    print("Synthetic dataset generated successfully!")
