# HR Analytics: Employee Promotion Prediction

## Overview

This project focuses on HR analytics, specifically addressing the challenge of identifying suitable candidates for promotion (up to the manager position) within a large multinational corporation. The goal is to streamline the promotion cycle by predicting potential promotees at a specific checkpoint, allowing the company to expedite the evaluation process and transition employees to new roles efficiently.

## Problem Statement

The company faces delays in the promotion cycle as final promotions are only announced after evaluation. This project aims to predict whether a potential promotee at a checkpoint in the test set will be promoted or not after the evaluation process. Key features include employee demographics, past and current performance metrics, and other relevant attributes.

## Features

- **employee_id:** Unique ID for each employee
- **department:** Department of the employee
- **region:** Region of employment (unordered)
- **education:** Education level of the employee
- **gender:** Gender of the employee
- **recruitment_channel:** Channel of recruitment for the employee
- **no_of_trainings:** Number of other trainings completed in the previous year on soft skills, technical skills, etc.
- **age:** Age of the employee
- **previous_year_rating:** Employee rating for the previous year
- **length_of_service:** Length of service in years
- **awards_won?:** 1 if awards were won during the previous year, else 0
- **avg_training_score:** Average score in current training evaluations
- **is_promoted:** (Target) Recommended for promotion (1 for promoted, 0 for not promoted)

## Objectives

1. Explore and understand the provided HR analytics dataset.
2. Preprocess and clean the data for machine learning.
3. Train machine learning models to predict whether an employee will be promoted after the evaluation process.
4. Evaluate model performance using appropriate metrics.
5. Provide insights and recommendations based on the model results.

## Implementation

1. **Data Exploration:**
   - Analyze and understand the distribution of features.
   - Identify patterns, trends, and potential correlations.

2. **Data Preprocessing:**
   - Handle missing values, if any.
   - Encode categorical variables and preprocess numerical features.

3. **Model Training:**
   - Train machine learning models to predict promotion outcomes.
   - Explore various algorithms and select the most suitable ones.

4. **Evaluation:**
   - Assess model performance using metrics such as accuracy, precision, recall, and F1 score.

5. **Insights and Recommendations:**
   - Provide actionable insights based on model predictions.
   - Offer recommendations for optimizing the promotion process.
