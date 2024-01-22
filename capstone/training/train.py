#!/usr/bin/env python
# coding: utf-8

import polars as pl
import polars.selectors as cs
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
import pickle
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

input_file = "./data/creditcard_2023.csv"
output_file = "./mlmodels/pipeline.pkl"


def extract_data(input_file):
    df = (
        pl.scan_csv(input_file)
        .select(pl.all().shrink_dtype().name.to_lowercase())
        .rename({"class": "is_fraud"})
    ).collect()
    df_full_train, _ = train_test_split(
        df, test_size=0.2, shuffle=True, random_state=11
    )
    return df_full_train


def prepare_dictionaries(df_full_train):
    y_full_train = df_full_train.select("is_fraud").to_pandas().values
    df_full_train = df_full_train.drop("is_fraud")
    categorical = df_full_train.select(cs.string()).columns
    numerical = df_full_train.select(cs.numeric()).columns
    dicts_full_train = df_full_train.select(categorical + numerical).to_dicts()
    return dicts_full_train, y_full_train


def train_model(dicts_full_train, y_full_train):
    params = {
        "n_estimators": 170,
        "max_depth": 36,
        "random_state": 11,
    }
    pipeline = make_pipeline(
        DictVectorizer(sparse=False), RandomForestClassifier(**params)
    )
    pipeline.fit(dicts_full_train, y_full_train.ravel())
    return pipeline


def save_model(artifact, output_file):
    with open(output_file, "wb") as file_out:
        pickle.dump(artifact, file_out)


def main():
    logging.info(f"reading the data from the following path: {input_file}...")
    df_full_train = extract_data(input_file)
    logging.info("preparing the data...")
    dicts_full_train, y_full_train = prepare_dictionaries(df_full_train)
    logging.info("training the model...")
    pipeline = train_model(dicts_full_train, y_full_train)
    logging.info(f"saving the results to the following path: {output_file}...")
    save_model(pipeline, output_file)


if __name__ == "__main__":
    main()
